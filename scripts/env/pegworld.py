import pybullet as p
import pybullet_tools.utils as ut
import os
from  pybullet_tools.utils import create_cylinder, set_point, set_pose, simulate_for_duration
from make_pipe import make_cylinder


"""purely simulated world
ONLY TO BE USED FOR COLLISION DETECTION
I am not focusing on making the dynamics of this at all realistic. 
"""
class PegWorld():
    def __init__(self, visualize=False, bullet=None, handonly=False):
        if visualize:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.handonly = handonly
        p.setGravity(0,0,-9.8)
        self.setup_robot()
        self.steps_taken = 0
        self.setup_workspace()


    """
    spawns a franka arm, eventually a FrankaArm object
    """
    def setup_robot(self):
        if not self.handonly:
            self.robot = p.loadURDF(os.environ["HOME"]+"/ros/src/franka_ros/franka_description/robots/model.urdf") #fixme, point somewhere less fragile
            set_point(self.robot, (-0.4,0,0.005))
            p.changeDynamics(self.robot, -1, mass=0)
        else:
            self.robot = p.loadURDF(os.environ["HOME"]+"/ros/src/franka_ros/franka_description/robots/hand.urdf") #fixme, point somewhere less fragile
            init_pos = (0,0,0.35)
            init_quat = (1,0,0,0)
            set_pose(self.robot,(init_pos, init_quat))


    """
    table with a hollow and solid cylinder on it
    """
    def setup_workspace(self):
        self.floor = p.loadURDF("../models/short_floor.urdf")
        #make board
        width = 0.2
        length = 0.1
        height = 0.04
        block_height = 0.03
        self.board = ut.create_box(width, length, height) 
        #make circle
        radius = 0.1
        self.circle = ut.create_cylinder(radius, block_height)
        #make square
        square_side = 0.08
        self.square =  ut.create_box(square_side, block_height)

    """
    Trajectory from current pose to goal considering attachments
    Avoids obstacles. This is our planner. At the moment it does not consider any transition model uncertainty. Will update after experimental results
    """
    def make_traj(self, goal):
        pass
        


if __name__ == "__main__":
    pw = PegWorld(visualize=True)
