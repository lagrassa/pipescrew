from frankapy import FrankaArm
import rospy
import cv_bridge
import time
#rospy.init_node("planorparam")
import numpy as np
from franka_action_lib.srv import GetCurrentRobotStateCmd
from franka_action_lib.msg import RobotState
from autolab_core import RigidTransform
import GPy as gpy
from autolab_core import RigidTransform, YamlConfig
from visualization import Visualizer3D as vis3d

from perception_utils.apriltags import AprilTagDetector
from perception_utils.realsense import get_first_realsense_sensor

from perception import Kinect2SensorFactory, KinectSensorBridged
from sensor_msgs.msg import Image
from perception.camera_intrinsics import CameraIntrinsics

from frankapy import FrankaArm
fa = FrankaArm()

class Circle:
    pass

class Square:
    pass
class Rectangle:
    def __init__(self):
        pass
    @staticmethod
    def symmetries():
        return [-np.pi, 0, np.pi]
    """
    returns the intervals of rotation that are identical, at least enough to be useful. 
    The triangle can't be rotated but the rectangle can be inserted at any interval of 3.14
    """
    @staticmethod
    def tforms_to_pose(ids, tforms,goal=False):
        if goal:
            rectangles_ids = [4,5,6,7]
        else:
            rectangles_ids = [0,1,2,3]
        relevant_ids = [ar_id for ar_id in ids if ar_id in rectangles_ids]
        tforms = np.array(tforms)[np.array(relevant_ids)]

        avg_rotation = tforms[0]
        for i in range(1, len(tforms)):
            avg_rotation = avg_rotation.interpolate_with(tforms[i], 0.5)

        if len(relevant_ids) == 4:
            translation = np.mean(np.vstack([T.translation for T in tforms]), axis=0)
        elif 0 in relevant_ids and 2 in relevant_ids:
            translation = np.mean(np.vstack([T.translation for T in tforms]), axis=0)
        elif 1 in relevant_ids and 3 in relevant_ids:
            translation = np.mean(np.vstack([T.translation for T in tforms]), axis=0)
        else:
            print("Not enough detections to make accurate pose estimate")

        return RigidTransform(rotation = avg_rotation.rotation, translation = translation, to_frame = tforms[0].to_frame, from_frame = "peg_center")

class Robot():
    def __init__(self):
        robot_state_server_name = '/get_current_robot_state_server_node_1/get_current_robot_state_server'
        rospy.wait_for_service(robot_state_server_name)
        self._get_current_robot_state = rospy.ServiceProxy(robot_state_server_name, GetCurrentRobotStateCmd)
        self.contact_threshold = -3.5 # less than is in contact
        self.bad_model_states = []
        self.good_model_states = []
        self.bridge = cv_bridge.CvBridge()
        self.shape_to_goal_loc = {}
        self.shape_type_to_ids = {Rectangle:(0,1,2,3)}
        self.shape_goal_type_to_ids = {Rectangle:1}
        self.setup_perception()

    def setup_perception(self):
        self.cfg = YamlConfig("april_tag_pick_place_azure_kinect_cfg.yaml")
        self.T_camera_world = RigidTransform.load(self.cfg['T_k4a_franka_path'])
        self.sensor = Kinect2SensorFactory.sensor('bridged', self.cfg)  # Kinect sensor object
        prefix = "/overhead"
        self.sensor.topic_image_color  = prefix+self.sensor.topic_image_color
        self.sensor.topic_image_depth  = prefix+self.sensor.topic_image_depth
        self.sensor.topic_info_camera  = prefix+self.sensor.topic_info_camera
        self.sensor.start()
        self.april = AprilTagDetector(self.cfg['april_tag'])
        # intr = sensor.color_intrinsics #original
        self.intr = CameraIntrinsics('k4a', 970.4990844726562,970.1990966796875, 1025.4967041015625, 777.769775390625, height=1536, width=2048)
                                 

    def detect_ar_world_pos(self,ids, straighten=True, shape_class = Rectangle, goal=False):
        #O, 1, 2, 3 left hand corner. average [0,2] then [1,3]
        T_tag_cameras = []
        detections = self.april.detect(self.sensor, self.intr, vis=self.cfg['vis_detect'])

        detected_ids = []
        for new_detection in detections:
            detected_ids.append(int(new_detection.from_frame.split("/")[1])) #won't work for non-int values
            T_tag_cameras.append(new_detection)
        T_tag_camera = shape_class.tforms_to_pose(detected_ids, T_tag_cameras, goal=goal) #as if there were a tag in the center
        T_tag_world = self.T_camera_world * T_tag_camera
        if straighten:
            T_tag_world  = straighten_transform(T_tag_world)
        print("detected pose", np.round(T_tag_world.translation,2))

        return T_tag_world
    """
    Goes to shape center and then grasps it 
    """
    def grasp_shape(self,T_tag_world, shape_type):
       self.holding_type = shape_type
       x_offset = 0
       z_offset = 0.025
       start = fa.get_pose()
       T_tag_tool = RigidTransform(rotation=np.eye(3), translation=[x_offset, 0, z_offset], from_frame="peg_center",
                                   to_frame="franka_tool")
       T_tool_world = T_tag_world * T_tag_tool.inverse()
       self.original_object_rotation = T_tool_world.copy() #save this for when we insert it 
       T_tool_world.rotation = start.rotation
       fa.open_gripper()
       path = self.linear_interp_planner(start, T_tool_world)
       self.follow_traj(path)
       fa.close_gripper()

    """
    assumes shape is already held 
    """
    def insert_shape(self):
        #find corresponding hole and orientation
        T_tag_world = self.get_shape_goal_location( self.holding_type)
        T_tag_tool = RigidTransform(rotation=np.eye(3), translation=[0, 0, 0.05],
                                    from_frame=T_tag_world.from_frame,
                                    to_frame="franka_tool")
        T_tool_world = T_tag_world * T_tag_tool.inverse()
        start = fa.get_pose()
        #set rotation to closest rotation in symmetry to pose
        start_yaw = self.original_object_rotation.euler_angles[-1] #yaw we're at right now
        planned_yaw = T_tool_world.euler_angles[-1] #yaw we need to go to 
        symmetries = self.holding_type.symmetries()
        best_symmetry_idx = np.argmin([np.linalg.norm((planned_yaw+sym)-start_yaw ) for sym in symmetries]) #wraparound :c
        best_correct_yaw = symmetries[best_symmetry_idx]
        good_rotation = RigidTransform.z_axis_rotation((planned_yaw+best_correct_yaw) -start_yaw)
        T_tool_world.rotation = np.dot(start.rotation,good_rotation)
        path = self.linear_interp_planner(start, T_tool_world)
        self.follow_traj(path, cart_gain = 1200, z_cart_gain = 1200, rot_cart_gain=150)
        z_down_offset = 0.02
        T_tool_world.translation[-1] -= z_down_offset
        self.follow_traj([T_tool_world], cart_gain = 300, z_cart_gain = 300, rot_cart_gain=150)
        

    def follow_traj(self, path, cart_gain = 2500, z_cart_gain = 2500, rot_cart_gain = 300):
        for pt, i in zip(path, range(len(path))):
            new_pos = pt
            expect_contact = False
            if new_pos.translation[2] < 0.01:
                expect_contact = True

            fa.goto_pose_with_cartesian_control(new_pos, cartesian_impedances=[cart_gain, cart_gain, z_cart_gain,rot_cart_gain, rot_cart_gain, rot_cart_gain]) #one of these but with impedance control? compliance comes from the matrix so I think that's good enough
            #fa.goto_pose(new_pos) #one of these but with impedance control? compliance comes from the matrix so I think that's good enough
            #consider breaking this one up to make it smoother
            force = self.feelforce()
            model_deviation = False
            pose_thresh = 0.02
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
            if model_deviation:
                print("Ending MB policy due to model deviation")
                return
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
        goal_ids = self.shape_goal_type_to_ids[shape_type]
        if shape_type not in self.shape_to_goal_loc.keys():
            goal_loc = self.detect_ar_world_pos(goal_ids, shape_class = Rectangle, goal=True, straighten=True)
            self.shape_to_goal_loc[shape_type] = goal_loc
        return self.shape_to_goal_loc[shape_type]

    def get_shape_location(self, shape_type):
        shape_ids = self.shape_type_to_ids[shape_type]
        return self.detect_ar_world_pos(shape_ids)
    """
    moves arm back to pose where it's not in the way of the camera
    """
    def reset_arm(self):
        fa.reset_joints()

    def feelforce(self):
        ros_data = self._get_current_robot_state().robot_state
        force = ros_data.O_F_ext_hat_K
        return force[2]
    def kinesthetic_teaching(self, prefix="test"):
        time = 4
        do_intel = False
        do_kinect = True
        self.ee_infos = []
        def callback_ee(data):
            ee_state = data.O_T_EE
            self.ee_infos.append(ee_state)
        if do_intel:
            intel_data = None
            self.intel_camera_images = []
            def callback_intel(data):
                image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
                self.intel_camera_images.append(image)
            self.intel_subscriber = rospy.Subscriber("/camera/color/image_raw", Image, callback_intel)
        if do_kinect:
            kinect_data = None
            self.kinect_camera_images = []
            def callback_kinect(data):
                image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
                self.kinect_camera_images.append(image)
                ee_data= rospy.wait_for_message("/robot_state_publisher_node_1/robot_state", RobotState)
                callback_ee(ee_data) #helps in syncing
            self.kinect_subscriber = rospy.Subscriber("/frontdown/rgb/image_raw", Image, callback_kinect)


        input("Beginning kinesthetic teaching. Ready?")
        fa.apply_effector_forces_torques(time, 0, 0, 0)
        np.save("data/"+str(prefix)+"ee_data.npy", self.ee_infos)
        if do_intel:
            np.save("data/"+str(prefix)+"intel_data.npy", self.intel_camera_images)

            self.intel_subscriber.unregister()
        if do_kinect:
            np.save("data/"+str(prefix)+"kinect_data.npy", self.kinect_camera_images)
            self.kinect_subscriber.unregister()

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
                rt_rot = RigidTransform.z_axis_rotation(angle_delta)
                rt.rotation = np.dot(rt.rotation, rt_rot)
            elif val == "e":
                rt_rot = RigidTransform.z_axis_rotation(-angle_delta)
                rt.rotation = np.dot(rt.rotation, rt_rot)
            elif val == 'k':
                delta *= 2
                angle_delta *= 2
            elif val == "j":
                delta /= 2
                angle_delta /=2
            
            fa.goto_pose_with_cartesian_control(rt, cartesian_impedances=[800, 800, 500, 200, 200, 200]) #one of these but with impedance control? compliance comes from the matrix so I think that's good enough
            #fa.goto_pose(rt)
            time.sleep(0.05)

def straighten_transform(rt):
    angles = rt.euler_angles
    roll_off = angles[0]
    pitch_off = angles[1]
    roll_fix = RigidTransform(rotation = RigidTransform.x_axis_rotation(np.pi-roll_off),  from_frame = rt.from_frame, to_frame=rt.from_frame)
    pitch_fix = RigidTransform(rotation = RigidTransform.y_axis_rotation(pitch_off), from_frame = rt.from_frame, to_frame=rt.from_frame)
    new_rt = rt*roll_fix*pitch_fix
    return new_rt

def run_insert_exp(robot, prefix):
    fa.open_gripper()
    fa.reset_joints()
    input("reset scene. Ready?")
    shape_center = robot.get_shape_location(Rectangle)
    robot.grasp_shape(shape_center, Rectangle)
    robot.insert_shape()
    robot.kinesthetic_teaching(prefix)


if __name__ == "__main__":
    n_exps = 3
    robot = Robot()
    fa.open_gripper()
    fa.reset_joints()
    print("goal loc", np.round(robot.get_shape_goal_location(Rectangle).translation,2))
    for i in [3,4,5]:
        run_insert_exp(robot, i)
    
