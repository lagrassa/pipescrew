from frankapy import FrankaArm
import rospy
import time
#rospy.init_node("planorparam")
import numpy as np
from franka_action_lib.srv import GetCurrentRobotStateCmd
from autolab_core import RigidTransform
import GPy as gpy
from autolab_core import RigidTransform, YamlConfig
from visualization import Visualizer3D as vis3d

from perception_utils.apriltags import AprilTagDetector
from perception_utils.realsense import get_first_realsense_sensor

from perception import Kinect2SensorFactory, KinectSensorBridged
from perception.camera_intrinsics import CameraIntrinsics

from frankapy import FrankaArm
fa = FrankaArm()
class Robot():
    def __init__(self):
        robot_state_server_name = '/get_current_robot_state_server_node_1/get_current_robot_state_server'
        rospy.wait_for_service(robot_state_server_name)
        self._get_current_robot_state = rospy.ServiceProxy(robot_state_server_name, GetCurrentRobotStateCmd)
        self.contact_threshold = -1.5 # less than is in contact
        self.bad_model_states = []
        self.good_model_states = []
        self.shape_to_goal_loc = {}
        self.shape_type_to_id = {Rectangle:0}
        self.shape_goal_type_to_id = {Rectangle:1}
        self.setup_perception()

    def setup_perception(self):
        self.cfg = YamlConfig("april_tag_pick_place_azure_kinect_cfg.yaml")
        self.T_camera_world = RigidTransform.load(self.cfg['T_k4a_franka_path'])
        self.sensor = Kinect2SensorFactory.sensor('bridged', self.cfg)  # Kinect sensor object
        self.sensor.start()
        self.april = AprilTagDetector(self.cfg['april_tag'])
        # intr = sensor.color_intrinsics #original
        self.intr = CameraIntrinsics('k4a', 972.31787109375, 971.8189086914062,
                                1022.3043212890625, 777.7421875, height=1536, width=2048)

    def detect_ar_world_pos(self,id, straighten=True):
        T_tag_camera = self.april.detect(self.sensor, self.intr, vis=self.cfg['vis_detect'])[id]
        T_tag_world = self.T_camera_world * T_tag_camera
        if straighten:
            T_tag_world  = straighten_transform(T_tag_world)
        return T_tag_world
    """
    Goes to shape center and then grasps it 
    """
    def grasp_shape(self,T_tag_world, shape_type):
       self.holding_type = shape_type
       x_offset = 0.024
       z_offset = 0.03
       T_tag_tool = RigidTransform(rotation=np.eye(3), translation=[x_offset, 0, z_offset], from_frame=T_tag_world.from_frame,
                                   to_frame="franka_tool")
       T_tool_world = T_tag_world * T_tag_tool.inverse()
       fa.open_gripper()
       start = fa.get_pose()
       path = self.linear_interp_planner(start, T_tool_world)
       self.follow_traj(path)
       fa.close_gripper()

    """
    assumes shape is already held 
    """
    def insert_shape(self):
        #find corresponding hole and orientation
        T_tag_world = self.get_shape_goal_location( self.holding_type)
        T_tag_tool = RigidTransform(rotation=np.eye(3), translation=[0, 0, 0.02],
                                    from_frame=T_tag_world.from_frame,
                                    to_frame="franka_tool")
        T_tool_world = T_tag_world * T_tag_tool.inverse()
        start = fa.get_pose()
        path = self.linear_interp_planner(start, T_tool_world)
        self.follow_traj(path)

    def follow_traj(self, path):
        for pt, i in zip(path, range(len(path))):
            new_pos = pt
            expect_contact = False
            if new_pos.translation[2] < 0.04:
                expect_contact = True

            import ipdb; ipdb.set_trace()
            #fa.goto_pose_with_cartesian_control(new_pos, cartesian_impedances=[2000, 2000, 1000, 300, 300, 300]) #one of these but with impedance control? compliance comes from the matrix so I think that's good enough
            fa.goto_pose(new_pos) #one of these but with impedance control? compliance comes from the matrix so I think that's good enough
            #consider breaking this one up to make it smoother
            force = self.feelforce()
            model_deviation = False
            if np.linalg.norm(fa.get_pose().translation-new_pos.translation) > pose_thresh:
                model_deviation = True
                print("Farther than expected. Expected "+str(np.round(new_pos.translation,2))+" but got "+
                      str(np.round(fa.get_pose().translation,2)))
            if force < self.contact_threshold and not expect_contact:
                print("unexpected contact")
                model_deviation = True
            elif force > self.contact_threshold and expect_contact:
                print("expected contact")
                model_deviation = True
            if model_deviation:
                self.bad_model_states.append(path[i-1])
            else:
                self.good_model_states.append(path[i-1])
            np.save("data/bad_model_states.npy", self.bad_model_states)
            np.save("data/good_model_states.npy", self.good_model_states)
    def train_model(self):
        self.high_state = [0.4,0.4,0.05]
        self.low_state = [-0.4, -0.4, 0.00]
        lengthscale = (self.high_state-self.low_state) * 0.05
        k = gpy.kern.Matern52(self.high_state, ARD=True, lengthscale=lengthscale)
        states = np.vstack([self.bad_model_states, self.good_model_states])
        labels = np.vstack([-np.ones(len(self.bad_model_states)), np.ones(len(self.good_model_states))])
        self.model = gpy.models.GPRegression(states, labels, k)
        self.model['.*variance'].constrain_bounded(1e-1,2., warning=False)
        self.model['Gaussian_noise.variance'].constrain_bounded(1e-4,0.01, warning=False)
        # These GP hyper parameters need to be calibrated for good uncertainty predictions.
        self.model.optimize(messages=False)


    def get_shape_goal_location(self, shape_type):
        goal_id = self.shape_goal_type_to_id[shape_type]
        if shape_type not in self.shape_to_goal_loc.keys():
            import ipdb; ipdb.set_trace()
            goal_loc = self.detect_ar_world_pos(goal_id)
            self.shape_to_goal_loc[shape_type] = goal_loc
        return self.shape_to_goal_loc[shape_type]

    def get_shape_location(self, shape_type):
        shape_id = self.shape_type_to_id[shape_type]
        return self.detect_ar_world_pos(shape_id)
    """
    moves arm back to pose where it's not in the way of the camera
    """
    def reset_arm(self):
        fa.reset_joints()

    def feelforce(self):
        ros_data = self._get_current_robot_state().robot_state
        force = ros_data.O_F_ext_hat_K
        return force[2]

    """
    Linear interpolation of poses, including quaternion
    """
    def linear_interp_planner(self, start, goal, n_pts = 2):
        return start.linear_trajectory_to(goal, n_pts)
    def keyboard_teleop(self):
        print("WASD teleop space for up c for down q and e for spin. k to increase delta, j to decrease ")
        rt = fa.get_pose()
        delta = 0.01
        angle_delta = 0.1
        while True:
            val = input()
            if val == "a":
                rt.translation[0] -= delta
            elif val == "d":
                rt.translation[0] += delta
            elif val == "s":
                rt.translation[1] -= delta
            elif val == "w":
                rt.translation[1] += delta
            elif val == " ":
                rt.translation[-1] += delta
            elif val == "c":
                rt.translation[-1] -= delta
            elif val == "q":
                rt_rot = RigidTransform.z_axis_rotation(0.1)
                rt = rt.apply(rt_rot)
            elif val == "e":
                rt_rot = RigidTransform.z_axis_rotation(-0.1)
                rt = rt.apply(rt_rot)
            elif val == 'k':
                delta *= 2
                angle_delta *= 2
            elif val == "j":
                delta /= 2
                angle_delta /=2
            
            #fa.goto_pose_with_cartesian_control(rt, cartesian_impedances=[600, 600, 400, 300, 300, 300]) #one of these but with impedance control? compliance comes from the matrix so I think that's good enough
            fa.goto_pose(rt)
            time.sleep(0.05)

def straighten_transform(rt):
    angles = rt.euler_angles
    roll_off = angles[0]
    pitch_off = angles[1]
    roll_fix = RigidTransform(rotation = RigidTransform.x_axis_rotation(np.pi-roll_off),  from_frame = rt.from_frame, to_frame=rt.from_frame)
    pitch_fix = RigidTransform(rotation = RigidTransform.y_axis_rotation(pitch_off), from_frame = rt.from_frame, to_frame=rt.from_frame)
    new_rt = rt*roll_fix*pitch_fix
    return new_rt

class Circle:
    pass

class Square:
    pass
class Rectangle:
    pass

if __name__ == "__main__":
    robot = Robot()
    shape_center = robot.get_shape_location(Rectangle)
    print("goal loc", np.round(robot.get_shape_goal_location(Rectangle).translation,2))
    robot.grasp_shape(shape_center, Rectangle)
    robot.insert_shape()
    robot.keyboard_teleop()
