from frankapy import FrankaArm
import numpy as np

fa = FrankaArm()
for i in range(20):
    input("Prepare to move arm")
    fa.apply_effector_forces_torques(3, 0, 0, 0)
    rt = fa.get_pose()
    print(rt)



