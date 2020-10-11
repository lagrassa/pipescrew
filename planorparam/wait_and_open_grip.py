import time
import numpy as np
from frankapy import FrankaArm
fa = FrankaArm()

time.sleep(6)
fa.open_gripper()

