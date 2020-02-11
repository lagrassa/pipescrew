from frankapy import FrankaArm
import rospy
import time
#rospy.init_node("planorparam")
import numpy as np
from franka_action_lib.srv import GetCurrentRobotStateCmd
from autolab_core import RigidTransform
import GPy as gpy
fa = FrankaArm()
class Robot():
    def __init__(self):
        robot_state_server_name = '/get_current_robot_state_server_node_1/get_current_robot_state_server'
        rospy.wait_for_service(robot_state_server_name)
        self._get_current_robot_state = rospy.ServiceProxy(robot_state_server_name, GetCurrentRobotStateCmd)
        self.contact_threshold = -1.5 # less than is in contact
        self.shape_type_to_goal_pose = {Square: (0,0,0.07), Circle: (0,0,0.02)} #eventually comes from perception
        self.bad_model_states = []
        self.good_model_states = []
        self.shape_type_to_goal_pose[Circle] = RigidTransform(rotation=np.array([1,0,0,0]), translation=[0.1,0,0.02],from_frame='franka_tool', to_frame='world')

        self.shape_type_to_goal_pose[Square] = RigidTransform(rotation=np.array([1,0,0,0]), translation=[0.1,0,0.02],from_frame='franka_tool', to_frame='world')

    """
    Goes to shape center and then grasps it 
    """
    def grasp_shape(self,shape_center, shape_type):
       self.holding_type = shape_type 
       fa.open_gripper()
       start = fa.get_pose().translation
       path = self.linear_interp_planner(start, shape_center)
       self.follow_traj(path)
       fa.close_gripper()

    """
    assumes shape is already held 
    """
    def insert_shape(self):
        #find corresponding hole and orientation
        shape_loc = self.get_shape_goal_location( self.holding_type)
        goal_loc_gripper = shape_loc + [0,0,0.02] #transform to robot gripper
        start = fa.get_pose()
        path = self.linear_interp_planner(start.translation, goal_loc_gripper)
        self.follow_traj(path)

    def follow_traj(self, path):
        for pt, i in zip(path, len(path)):
            new_pos = RigidTransform(rotation=pt[3:], translation=pt[0:3],from_frame='franka_tool', to_frame='world')
            if new_pos.translation[2] < 0.04:
                expect_contact = True
            fa.goto_pose_with_cartesian_control(new_pos, cartesian_impedances=[2000, 2000, 500, 300, 300, 300]) #one of these but with impedance control? compliance comes from the matrix so I think that's good enough
            #consider breaking this one up to make it smoother
            force = self.feelforce()
            model_deviation = False
            if force < self.contact_threshold and not expect_contact:
                print("unexpected contact")
                model_deviation = True
            elif force > self.contact_threshold and expect_contact:
                print("expected contact")
                model_deviation = True
            if model_deviation:
                self.bad_model_states.append(path[i-1])
            else:
                self.good_model_state.append(path[i-1])
            np.save("data/bad_model_states.npy")
            np.save("data/good_model_states.npy")
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
        return self.shape_type_to_goal_pose[shape_type]
    def get_shape_location(self, shape_type):
        if shape_type == Circle:
            return RigidTransform(rotation=np.array([1,0,0,0]), translation=[0.1,0,0.02],from_frame='franka_tool', to_frame='world')
        else:
            return RigidTransform(rotation=np.array([1,0,0,0]), translation=[-0.1,0,0.2],from_frame='franka_tool', to_frame='world')
    def feelforce(self):
        ros_data = self._get_current_robot_state().robot_state
        force = ros_data.O_F_ext_hat_K
        return force

    """
    Linear interpolation of poses, including quaternion
    """
    def linear_interp_planner(self, start, goal, n_pts = 8):
        return np.linspace(start, goal, n_pts)
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
            
            fa.goto_pose_with_cartesian_control(rt, cartesian_impedances=[600, 600, 400, 300, 300, 300]) #one of these but with impedance control? compliance comes from the matrix so I think that's good enough
            time.sleep(0.05)



class Circle:
    pass

class Square:
    pass

if __name__ == "__main__":
    robot = Robot()
    shape_center = robot.get_shape_location(Circle)
    robot.grasp_shape(shape_center, Circle)
    robot.insert_shape()
    robot.keyboard_teleop()
        



