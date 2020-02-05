from frankapy import FrankaArm
import numpy as np

fa = FrankaArm()
fa.apply_effector_forces_torques(args.time, 0, 0, 0)
input("Bring arm to where it should take the gripper:")
rt = fa.get_pose()
np.save("above_pose.npy", (rt.translation,rt.rotation))
input("Bring arm to where insertion point:")
rt = fa.get_pose()
np.save("insert_pose.npy", (rt.translation, rt.rotation))


