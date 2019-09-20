import pybullet as p
from  pybullet_tools.utils import create_cylinder, set_point, set_pose, simulate_for_duration


"""purely simulated world"""
class PipeWorld():
    def __init__(self, visualize=False):
        if visualize:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setGravity(0,0,-9.8)
        self.setup_robot()
        self.setup_workspace()
    """
    spawns a franka arm, eventually a FrankaArm object
    """
    def setup_robot(self):
        self.robot = p.loadURDF("/usr0/home/alagrass/ros/src/franka_ros/franka_description/robots/model.urdf") #fixme, point somewhere less fragile
        set_point(self.robot, (-0.3,0,0.005))
        p.changeDynamics(self.robot, -1, mass=0)

    """
    table with a hollow and solid cylinder on it
    """
    def setup_workspace(self):
        self.floor = p.loadURDF("../models/short_floor.urdf")
        p.changeDynamics(self.floor, -1, mass=0)
        self.hollow = p.loadURDF("../models/hollow.urdf", (0,0,0), globalScaling=0.015)
        p.changeDynamics(self.hollow, -1, mass=0)
        set_pose(self.hollow, ((0.0,0,0.02),(0,0.5,0.5,0)))
        self.pipe = create_cylinder(0.01, 0.1, mass=1, color=(0, 0, 1, 1))
        p.changeDynamics(self.pipe, -1, mass=0.3)
        set_point(self.pipe, (0,0.08,0.015))


if __name__ == "__main__":
    pw = PipeWorld(visualize=True)
