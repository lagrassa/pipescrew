from frankapy import FrankaArm
import numpy as np

fa = FrankaArm()
import ipdb; ipdb.set_trace()
for i in range(30):
    fa.apply_effector_forces_torques(1, 0, 0, 0)
    print(fa.get_pose().euler_angles)


input("Bring arm to where it should take the gripper:")
rt = fa.get_pose()
np.save("above_pose.npy", (rt.translation,rt.rotation))
input("Bring arm to where insertion point:")
rt = fa.get_pose()
np.save("insert_pose.npy", (rt.translation, rt.rotation))


