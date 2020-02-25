import pybullet as p
import numpy as np
from pyquaternion import Quaternion
from transformations import quaternion_about_axis
from real_robot.shapes import *
import env.pb_utils as ut
import os
from  env.pb_utils import create_cylinder, set_point, set_pose, simulate_for_duration


"""purely simulated world
ONLY TO BE USED FOR COLLISION DETECTION
I am not focusing on making the dynamics of this at all realistic. 
"""
class PegWorld():
    def __init__(self, visualize=False, bullet=None, handonly=False, rectangle_loc = [[0.562, -0.121, 0.016], [1, 0.   ,  0.   ,  0  ]],circle_loc = [[0.425, 0.101, 0.01 ],[1, -0.  ,  0.  ,  0  ]],
            obstacle_loc = [[ 0.53172045, -0.03062703,  0.07507126], [1, -0.   ,  0.   ,  0   ]], board_loc = [[0.479, 0.0453, -0.015],[0.707, 0.707, 0.   , 0.   ]], hole_goal =  ((0.55,0.08, 0), (1,0,0,0))):
        if visualize:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.handonly = handonly
        p.setGravity(0,0,-9.8)
        self.in_hand = []
        self.setup_robot()
        self.steps_taken = 0
        self.setup_workspace(rectangle_loc=rectangle_loc, circle_loc=circle_loc, board_loc=board_loc, obstacle_loc=obstacle_loc, hole_goal = hole_goal)


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
            self.grasp_joint = 10
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
    def setup_workspace(self, rectangle_loc = [[0.562, 0.151, 0.016], [1, 0.   ,  0.   ,  0   ]],
            circle_loc = [[0.425, 0.101, 0.01 ],[1, 0.  ,  0.  ,  0  ]],
            obstacle_loc = [[ 0.53172045, -0.03062703,  0.07507126], [1, -0.   ,  0.   ,  0   ]], board_loc = [[0.479, 0.0453, 0.015],[0.707, 0.707, 0.   , 0.   ]], hole_goal = [[0.55, 0.08, 0.0],[1,0,0,0]]):
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
        self.block_height = block_height
        self.board = ut.create_box(width, length, height, color = (1,0.7,0,1)) 
        ut.set_pose(self.board, board_loc)
        #make circle
        radius = 0.078/2
        self.circle = ut.create_cylinder(radius, block_height, color = (1,1,0,1))
        #make rectangle
        self.rectangle =  ut.create_box(0.092, 0.069, block_height, color=(0.5,0,0.1,1))
        self.obstacle = ut.create_box(0.08, 0.04, 0.08, color = (0.5,0.5,0.5,1))
        self.hole = ut.create_box(0.092, 0.069, 0.005, color = (0.1,0,0,1))
        rectangle_loc[0][-1] -= 0.5*block_height
        circle_loc[0][-1] -= 0.5*block_height
        obstacle_loc[0][-1] -= 0.5*0.08
        ut.set_pose(self.rectangle, rectangle_loc)
        ut.set_pose(self.circle, circle_loc)
        ut.set_pose(self.obstacle, obstacle_loc)
        ut.set_pose(self.hole, hole_goal)
        self.shape_name_to_shape = {}
        self.shape_name_to_shape[Circle] = self.circle
        self.shape_name_to_shape[Rectangle] = self.rectangle
        self.shape_name_to_shape[Obstacle] = self.obstacle
        input("workspace okay?")

    def get_closest_ee_goals(self, shape_goal, shape_class=Rectangle, grasp_offset = 0.055):
        #symmetry that minimizes the distance between shape_goal and the current ee pose. 
        curr_quat = Quaternion(p.getLinkState(self.robot, self.grasp_joint)[1])
        goal_quat = Quaternion(shape_goal[1])
        syms = shape_class.grasp_symmetries()
        transformation_lib_quats = [quaternion_about_axis(sym, (0,0,1))for sym in syms]
        sym_quats = [Quaternion(np.hstack([lib_quat[-1],lib_quat[0:3]])) for lib_quat in transformation_lib_quats]
        best_sym_idx = np.argmin([Quaternion.absolute_distance(curr_quat,Quaternion(matrix=np.dot(sym_quat.rotation_matrix,goal_quat.rotation_matrix))) for sym_quat in sym_quats
])
        best_sym = shape_class.symmetries()[best_sym_idx]
        #best_quat = Quaternion(quaternion_about_axis(best_sym, (0,0,1))).rotate(goal_quat) 
        #assert(Quaternion.absolute_distance(curr_quat, best_quat) < 0.01)
        best_quat = curr_quat #hack until I can get the angle stuff working
         
        self.grasp_offset = grasp_offset+self.block_height*0.5
        grasp_translation =np.array([0,0,self.grasp_offset])
        ee_trans = grasp_translation+shape_goal[0]
        grasp_rot = np.dot(curr_quat.rotation_matrix, np.linalg.inv(best_quat.rotation_matrix))
        grasp_quat = np.array([.707,.707,0,0])# hack until I can get the angle stuff working Quaternion(matrix=grasp_rot).elements
        grasp = (grasp_translation, grasp_quat)
        ee_pose = (ee_trans, best_quat.elements)
        return ee_pose, grasp


        


    """
    Trajectory from current pose to goal considering attachments
    Avoids obstacles. This is our planner. At the moment it does not consider any transition model uncertainty. Will update after experimental results
    """
    def make_traj(self, goal_pose, shape_class=Rectangle):
        state = p.saveState()
        #quat =  p.getLinkState(self.robot, self.grasp_joint)[1]
        quat = [1,0,0,0]
        #all of this only really makes sense with a full robot
        joints_to_plan_for = ut.get_movable_joints(self.robot)[:-2] #all but the fingers

        rest =  np.mean(np.vstack([ut.get_min_limits(self.robot, joints_to_plan_for), ut.get_max_limits(self.robot, joints_to_plan_for)]), axis=0)
        rest = [ut.get_joint_positions(self.robot, ut.get_movable_joints(self.robot))]
        lower = ut.get_min_limits(self.robot, joints_to_plan_for)
        upper = ut.get_max_limits(self.robot, joints_to_plan_for)
        ranges = 10*np.ones(len(joints_to_plan_for))
        null_space = [lower, upper, ranges, [rest]]
        null_space = None
       
        end_conf = ut.inverse_kinematics(self.robot, self.grasp_joint, goal_pose, movable_joints=joints_to_plan_for, null_space=null_space) #end conf to be in the goal loc
        if end_conf is None:
            print("No IK solution found")
            p.restoreState(state) 
            return None
        end_conf = end_conf[:len(joints_to_plan_for)]
        p.restoreState(state)
        #for attachment in self.in_hand:
        #    attachment.assign()
        traj = ut.plan_joint_motion(self.robot, joints_to_plan_for, end_conf, obstacles=[self.obstacle, self.circle], attachments=self.in_hand,
                      self_collisions=True, disabled_collisions=set(self.in_hand),
                      weights=None, resolutions=None)
        
        p.restoreState(state)
        return traj
    def attach_shape(self, shape_name, grasp_pose):
        self.grasp = grasp_pose
        attachment = ut.Attachment(self.robot, self.grasp_joint, grasp_pose, self.shape_name_to_shape[shape_name])
        new_obj_pose = np.array(p.getLinkState(self.robot, self.grasp_joint)[0])+self.grasp[0]
        #ut.set_point(self.shape_name_to_shape[shape_name], new_obj_pose)
        import ipdb; ipdb.set_trace()
        attachment.assign()
        self.in_hand = [attachment]

    def detach_shape(self, shape_name):
        self.in_hand = []
    def set_joints(self, joint_vals):
        ut.set_joint_positions(self.robot, ut.get_movable_joints(self.robot)[:len(joint_vals)], joint_vals)
        for attachment in self.in_hand:
            attachment.assign()
    def visualize_traj(self, path):
        for pt in path:
            self.set_joints(pt)
            input("ready for next set?")
    def visualize_points(self, pt_set):
        for pt in pt_set:
            pb_pt = ut.create_sphere(0.005)
            ut.set_point(pb_pt, pt)
    """
    Collision-free trajectory  to place object in hole
    """
    def grasp_object(self, shape_class=Rectangle, visualize=False):
        shape_goal = p.getBasePositionAndOrientation(pw.rectangle)
        ee_goals = []
        grasps = []
        state = p.saveState()
        sample_joint = 6
        original = ut.get_joint_position(self.robot, sample_joint)
        grasp_symmetries =  shape_class.grasp_symmetries()
        for sym in grasp_symmetries:
            if original+sym < 3 and original+sym > -3:
                ut.set_joint_position(self.robot, sample_joint, original+sym)
                curr_pos  = ut.get_joint_position(self.robot, sample_joint)
                ee_goal, grasp = pw.get_closest_ee_goals(shape_goal, shape_class=Rectangle)
                ee_goals.append(ee_goal)
                grasps.append(grasp)
        p.restoreState(state)

        #grasp = grasp_from_ee_and_obj(ee_goal, shape_goal)
        working_grasp = None
        working_traj = None
        for ee_goal, grasp in zip(ee_goals, grasps):
            grasp_traj = pw.make_traj(ee_goal)
            if grasp_traj is not None:
                working_grasp = grasp 
                working_traj = grasp_traj
        if visualize:
            pw.visualize_traj(working_traj)
        import ipdb; ipdb.set_trace()
        pw.attach_shape(Rectangle, working_grasp)
        return grasp_traj
    """
    Collision-free trajectory  to place object in hole
    """
    def place_object(self, visualize=False, shape_class=Rectangle, hole_goal= [[0.55, 0.08, 0.0],[1,0,0,0]]):
        ee_goal, grasp = pw.get_closest_ee_goals(hole_goal, shape_class=shape_class, grasp_offset = 0.095)
        traj = pw.make_traj(ee_goal)
        if visualize:
            pw.visualize_traj(traj)
        return traj



if __name__ == "__main__":
    #pw = PegWorld( visualize=False, handonly=False)
    pw = PegWorld(visualize=True, handonly = False)
    pw.grasp_object(shape_class=Rectangle, visualize=True)
    #pw.visualize_points(np.load("../real_robot/data/bad_model_states.npy", allow_pickle=True)[:,0:3])
    pw.place_object(shape_class=Rectangle, visualize=True)
