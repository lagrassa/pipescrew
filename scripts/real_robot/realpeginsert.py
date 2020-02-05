from frankapy import FrankaArm
import rospy
import time
#rospy.init_node("planorparam")
import numpy as np
from franka_action_lib.srv import GetCurrentRobotStateCmd
from autolab_core import RigidTransform

fa = FrankaArm()
class Robot():
    def __init__(self):
        shape_type_to_pose = {Square: (0,0,0.07), Circle: (0,0,0.02)}
        robot_state_server_name = '/get_current_robot_state_server_node_1/get_current_robot_state_server'
        rospy.wait_for_service(robot_state_server_name)
        self._get_current_robot_state = rospy.ServiceProxy(robot_state_server_name, GetCurrentRobotStateCmd)
        self.contact_threshold = -1.5 # less than is in contact
        pass
    """
    Goes to shape center and then grasps it 
    """
    def grasp_shape(self,shape_center, shape_type):
       self.holding_type = shape_type 
       fa.open_gripper()
       path = self.linear_interp(start, shape_center)
       fa.close_gripper()

    """
    assumes shape is already held 
    """
    def insert_shape(self):
        #find corresponding hole and orientation
        shape_loc = self.get_shape_location(location, self.holding_type)
        goal_loc_gripper = shape_loc + [0,0,0.02] #transform to robot gripper
        path = self.linear_interp(start, goal_loc_gripper)
        self.follow_traj(path)

    def follow_traj(self, path):
        for pt in parth:
            new_pos = RigidTransform(rotation=pt[3:], translation=pt[0:3],from_frame='franka_tool', to_frame='world')
            if new_pos.translation[2] < 0.04:
                expect_contact = True
            fa.goto_pose_with_cartesian_control(new_pos, cartesian_impedances=[2000, 2000, 500, 300, 300, 300]) #one of these but with impedance control? compliance comes from the matrix so I think that's good enough
            #check for contact


    def get_shape_location(self, shape_type):
        return shape_type_to_pose[shape_type]
    def feelforce(self):
        ros_data = self._get_current_robot_state().robot_state
        force = ros_data.O_F_ext_hat_K
        return force

    """
    Linear interpolation of poses, including quaternion
    """
    def linear_interp(self, start, goal, n_pts = 10):
        return np.linspace(start, goal, n_pts)
    def keyboard_teleop(self):
        print("WASD teleop space for up c for down")
        rt = fa.get_pose()
        delta = 0.01
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

            fa.goto_pose_with_cartesian_control(rt, cartesian_impedances=[600, 600, 400, 300, 300, 300]) #one of these but with impedance control? compliance comes from the matrix so I think that's good enough
            
            time.sleep(0.05)
     


class Circle:
    pass

class Square:
    pass

if __name__ == "__main__":
    robot = Robot()        
    robot.keyboard_teleop()
        



