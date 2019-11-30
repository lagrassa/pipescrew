import pybullet as p
import os
from  pybullet_tools.utils import create_cylinder, set_point, set_pose, simulate_for_duration
from make_pipe import make_cylinder


"""purely simulated world"""
class PipeWorld():
    def __init__(self, visualize=False, bullet=None, handonly=True):
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
            self.cid = p.createConstraint(self.robot, -1, -1, -1, p.JOINT_FIXED, [0, 0, 1], [0, 0, 0], [0, 0, 0],[0, 0, 0, 1], [0, 0, 0, 1])
            p.changeConstraint(self.cid, init_pos, init_quat, maxForce=20)


    """
    table with a hollow and solid cylinder on it
    """
    def setup_workspace(self):
        self.floor = p.loadURDF("../models/short_floor.urdf")
        p.changeDynamics(self.floor, -1, mass=0)
        #self.hollow = p.loadURDF("../models/hollow.urdf", (0,0,0), globalScaling=0.020)
        length =0.04
        sec_width =  0.009
        thick = 0.001
        angle_correction = -0.25 #not a nice hack to make the pipe look better
        self.hollow = make_cylinder(12,sec_width,length,thick, angle_correction)
        p.changeDynamics(self.hollow, -1, mass=0)
        set_pose(self.hollow, ((0.0,0,0.0),(0,0.8,0.8,0)))
        self.pipe = create_cylinder(0.01, 0.1, mass=1, color=(0, 0, 1, 1))
        p.changeDynamics(self.pipe, -1, mass=0.3, lateralFriction=0.99, rollingFriction=0.99, spinningFriction=0.99, restitution=0.05)
        p.changeDynamics(self.robot, 9, lateralFriction=0.99, rollingFriction=0.99, spinningFriction=0.99, restitution=0.05)
        p.changeDynamics(self.robot, 10, lateralFriction=0.99, rollingFriction=0.99, spinningFriction=0.99, restitution=0.05)
        set_point(self.pipe, (0,0.08,0.015))
    def shift_t_joint(self, new_x, new_y):
        old_pos, old_orn = p.getBasePositionAndOrientation(self.hollow)
        
        set_pose(self.hollow, ((new_x,new_y,old_pos[2]),old_orn))



if __name__ == "__main__":
    pw = PipeWorld(visualize=True)
