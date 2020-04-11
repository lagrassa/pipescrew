from frankapy import FrankaArm
print("imported franka arm")
import numpy as np
fa = FrankaArm()
print("made franka arm")
input("Bring arm to where it should take the gripper:")
fa.apply_effector_forces_torques(15, 0, 0, 0)
rt = fa.get_pose()
print(rt)
np.save("data/grasp_pipe_pose_trans.npy", rt.translation)
#input("Bring arm to where insertion point:")
#rt = fa.get_pose()
#np.save("insert_pose_trans.npy", rt.translation)


