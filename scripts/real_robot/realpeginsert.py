from frankapy import FrankaArm
import numpy as np
from autolab_core import RigidTransform

fa = FrankaArm()
class Robot():
    def __init__(self):
        shape_type_to_pose = {Square: (0,0,0.07), Circle: (0,0,0.02)}
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
    def insert_shape(self, ):
        #find corresponding hole and orientation
        shape_loc = self.get_shape_location(location, self.holding_type)
        goal_loc_gripper = shape_loc + [0,0,0.02] #transform to robot gripper
        path = self.linear_interp(start, goal_loc_gripper)
        self.follow_traj(path)

    def follow_traj(self, path)
        for pt in parth:
            new_pos = RigidTransform(rotation=pt[3:], translation=pt[0:3],from_frame='franka_tool', to_frame='world')
            if new_pos.translation[2] < 0.04:
                expect_contact = True
            fa.goto_pose_with_cartesian_control(new_pos, cartesian_impedances=[2000, 2000, 500, 300, 300, 300]) #one of these but with impedance control? compliance comes from the matrix so I think that's good enough
            #check for contact


    def get_shape_location(self, shape_type):
        return shape_type_to_pose[shape_type]
    """
    Linear interpolation of poses, including quaternion
    """
    def linear_interp(self, start, goal, n_pts = 10):
        return np.linspace(start, goal, n_pts)


class Circle:

class Square:

        
        



