from frankapy import FrankaArm
import numpy as np
from autolab_core import RigidTransform

fa = FrankaArm()
trans, rot = np.load("above_pose.npy")
#fa.close_gripper()
above_position = RigidTransform(rotation=rot, translation=trans,from_frame='franka_tool', to_frame='world')

fa.goto_pose_with_cartesian_control(above_position, cartesian_impedances=[3000, 3000, 100, 300, 300, 300])
fa.open_gripper()
input("Add gripper")
fa.close_gripper()
trans, rot = np.load("insert_pose.npy")
insert_position= RigidTransform(rotation=rot, translation=trans,from_frame='franka_tool', to_frame='world')
insert_position.translation[-1] -=0.01 # manual adjustment 

fa.goto_pose_with_cartesian_control(insert_position, cartesian_impedances=[20, 20, 2000, 300, 300, 300]) #one of these but with impedance control? compliance comes from the matrix so I think that's good enough

