import pybullet as p
import numpy as np
import env.pb_utils as ut
import os
from  env.pb_utils import create_cylinder, set_point, set_pose, simulate_for_duration


"""purely simulated world
ONLY TO BE USED FOR COLLISION DETECTION
I am not focusing on making the dynamics of this at all realistic. 
"""
class PegWorld():
    def __init__(self, visualize=False, bullet=None, handonly=False, rectangle_loc =None, circle_loc = None, board_loc = None, obstacle_loc=None):
        if visualize:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.handonly = handonly
        p.setGravity(0,0,-9.8)
        self.in_hand = []
        self.setup_robot()
        self.steps_taken = 0
        self.setup_workspace(rectangle_loc=rectangle_loc, circle_loc=circle_loc, board_loc=board_loc, obstacle_loc=obstacle_loc)


    """
    spawns a franka arm, eventually a FrankaArm object
    """
    def setup_robot(self):
        if not self.handonly:
            #self.robot = p.loadURDF(os.environ["HOME"]+"/ros/src/franka_ros/franka_description/robots/model.urdf") #fixme, point somewhere less fragile
            self.robot = p.loadURDF("../../models/robots/model.urdf")
            set_point(self.robot, (0,0,0.005))
            start_joints = (0.09186411075857098, 0.02008522792588543, 0.03645461729775788, -1.9220854528910314, 0.213232566443952983, 1.647271913704007, 0.0, 0.0, 0.0)
            start_joints = [ 2.78892560e-04, -7.85089406e-01,  4.81729135e-05, -2.35613802e+00,
                            4.95981896e-04,  1.57082514e+00,  7.85833531e-01, 0, 0]
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
    def setup_workspace(self, rectangle_loc = ([0.562, 0.151, 0.016], [-0.029, -0.   ,  0.   ,  1.   ]),
            circle_loc = ([0.425, 0.101, 0.01 ],[-0.03, -0.  ,  0.  ,  1.  ]),
            obstacle_loc = ([ 0.53172045, -0.03062703,  0.07507126], [-0.028, -0.   ,  0.   ,  1.   ]), board_loc = ((0.479, 0.0453, -0.015),[0.707, 0.707, 0.   , 0.   ])):
        import ipdb; ipdb.set_trace()
        #RigidTransform(rotation=np.array([[-5.78152806e-02, -9.98327119e-01,  4.84639353e-07],
        #       [-9.98327425e-01,  5.78157598e-02,  3.97392158e-08],
        #              [ 4.07518811e-07, -6.59092487e-08, -9.99999635e-01]]), translation=np.array([ 0.53810962,  0.08998347, -0.00768057]), from_frame='peg_center', to_frame='world')
        #hole_pose = (np.array([ 0.53810962,  0.08998347, -0.00768057]), np.array([-2.89198577e-02, -9.19833769e-08,  1.37694750e-07,  9.99581685e-01]))
        #self.floor = p.loadURDF("../../models/short_floor.urdf")
        #make board
        width = 0.4
        length = 0.3
        height = 0.02
        block_height = 0.01
        self.board = ut.create_box(width, length, height, color = (1,0.7,0,1)) 
        ut.set_pose(self.board, board_loc)
        #make circle
        radius = 0.078/2
        self.circle = ut.create_cylinder(radius, block_height, color = (1,1,0,1))
        #make rectangle
        self.rectangle =  ut.create_box(0.092, 0.069, block_height, color=(0.5,0,0.1,1))
        self.obstacle = ut.create_box(0.08, 0.04, 0.08, color = (0.5,0.5,0.5,1))
        rectangle_loc[0][-1] -= 0.5*block_height
        circle_loc[0][-1] -= 0.5*block_height
        obstacle_loc[0][-1] -= 0.5*0.08
        ut.set_pose(self.rectangle, rectangle_loc)
        ut.set_pose(self.circle, circle_loc)
        ut.set_pose(self.obstacle, obstacle_loc)
        self.shape_name_to_shape = {}
        self.shape_name_to_shape['circle'] = self.circle
        self.shape_name_to_shape['rectangle'] = self.rectangle

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
        traj = ut.plan_joint_motion(self.robot, joints_to_plan_for, end_conf, obstacles=[self.board, self.obstacle, self.rectangle], attachments=self.in_hand,
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
    def set_joints(self, joint_vals):
        ut.set_joint_positions(self.robot, ut.get_movable_joints(self.robot), joint_vals)
        for attachment in self.in_hand:
            attachment.assign()


if __name__ == "__main__":
    board_loc = ((0,0,0),(1,0,0,0))
    circle_loc = ((0.3,0.2,0),(1,0,0,0)) #acquire from darknet
    obstacle_loc = ((0.2,0.1,0),(1,0,0,0)) #TODO acquire from perception
    rectangle_loc = ((0.3,0.5,0),(1,0,0,0)) 
    pw = PegWorld(rectangle_loc=rectangle_loc, circle_loc=circle_loc, board_loc=board_loc, obstacle_loc=obstacle_loc, visualize=True, handonly=False)
    import ipdb; ipdb.set_trace()
    #pw = PegWorld(visualize=False, handonly = False)
    pw.attach_shape('rectangle', (np.array([0,0,0.05]), np.array([1,0,0,0])))
    traj = pw.make_traj(np.array([0,0.05,0.3]))
    print("trajectory", traj)
