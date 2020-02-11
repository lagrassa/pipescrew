import pybullet as p
import numpy as np
import utils as ut
import os
from  utils import create_cylinder, set_point, set_pose, simulate_for_duration


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
        self.in_hand = []
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
            start_joints = (0.09186411075857098, 0.02008522792588543, 0.03645461729775788, -1.9220854528910314, 0.213232566443952983, 1.647271913704007, 0.0, 0.0, 0.0)
            self.grasp_joint = 9
            p.changeDynamics(self.robot, -1, mass=0)
            ut.set_joint_positions(self.robot, ut.get_movable_joints(self.robot), start_joints)
        else:
            self.robot = p.loadURDF(os.environ["HOME"]+"/ros/src/franka_ros/franka_description/robots/hand.urdf") #fixme, point somewhere less fragile
            init_pos = (0,0,0.35)
            init_quat = (1,0,0,0)
            self.grasp_joint = 0#kinda sketchy tbh, try to just use the whole robot
            set_pose(self.robot,(init_pos, init_quat))


    """
    table with a hollow and solid cylinder on it
    """
    def setup_workspace(self):
        self.floor = p.loadURDF("../../models/short_floor.urdf")
        #make board
        width = 0.3
        length = 0.2
        height = 0.02
        block_height = 0.01
        self.board = ut.create_box(width, length, height) 
        #make circle
        radius = 0.02
        self.circle = ut.create_cylinder(radius, block_height)
        #make square
        square_side = 0.03
        self.square =  ut.create_box(square_side, square_side, block_height, color=(0,0,1,1))
        square_loc = (0,-0.2,0)
        circle_loc = (0,0.1,0)
        ut.set_point(self.square, square_loc)
        ut.set_point(self.circle, square_loc)
        self.shape_name_to_shape = {}
        self.shape_name_to_shape['circle'] = self.circle
        self.shape_name_to_shape['square'] = self.square

    """
    Trajectory from current pose to goal considering attachments
    Avoids obstacles. This is our planner. At the moment it does not consider any transition model uncertainty. Will update after experimental results
    """
    def make_traj(self, goal):
        #quat =  p.getLinkState(self.robot, self.grasp_joint)[1]
        quat = [1,0,0,0]
        #all of this only really makes sense with a full robot
        goal_pose = (goal+self.grasp[0], quat)
        joints_to_plan_for = ut.get_movable_joints(self.robot)[:-2] #all but the fingers

        rest =  np.mean(np.vstack([ut.get_min_limits(self.robot, joints_to_plan_for), ut.get_max_limits(self.robot, joints_to_plan_for)]), axis=0)
        rest = [ut.get_joint_positions(self.robot, ut.get_movable_joints(self.robot))]
        lower = ut.get_min_limits(self.robot, joints_to_plan_for)
        upper = ut.get_max_limits(self.robot, joints_to_plan_for)
        ranges = 10*np.ones(len(joints_to_plan_for))
        null_space = [lower, upper, ranges, [rest]]
        null_space = None

        end_conf = ut.inverse_kinematics(self.robot, self.grasp_joint, goal_pose, movable_joints=joints_to_plan_for, null_space=null_space) #end conf to be in the goal loc
        end_conf = end_conf[:len(joints_to_plan_for)]
        for attachment in self.in_hand:
            attachment.assign()
        traj = ut.plan_joint_motion(self.robot, joints_to_plan_for, end_conf, obstacles=[self.board, self.square], attachments=self.in_hand,
                      self_collisions=True, disabled_collisions=set(self.in_hand),
                      weights=None, resolutions=None)
        
        return traj
    def attach_shape(self, shape_name, grasp_pose):
        self.grasp = grasp_pose
        attachment = ut.Attachment(self.robot, self.grasp_joint, grasp_pose, self.shape_name_to_shape[shape_name])
        new_obj_pose = np.array(p.getLinkState(self.robot, self.grasp_joint)[0])+self.grasp[0]
        #ut.set_point(self.shape_name_to_shape[shape_name], new_obj_pose)
        attachment.assign()
        self.in_hand = [attachment]

    def detach_shape(self, shape_name):
        self.in_hand = []

if __name__ == "__main__":
    pw = PegWorld(visualize=True, handonly = False)
    pw.attach_shape('circle', (np.array([0,0,0.05]), np.array([1,0,0,0])))
    traj = pw.make_traj(np.array([0,0.05,0.3]))
    print("trajectory", traj)
