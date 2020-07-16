import pybullet as p
from autolab_core import RigidTransform
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
    def __init__(self, visualize=False, bullet=None, handonly=False,load_previous=False,  rectangle_loc = [[0.55, -0.113, 0.016], [1, 0.   ,  0.   ,  0  ]],circle_loc = [[0.425, 0.101, 0.01 ],[1, -0.  ,  0.  ,  0  ]],
            obstacle_loc = [[ 0.58172045, -0.04062703,  0.07507126], [1, -0.   ,  0.   ,  0   ]], board_loc = [[0.379, -0.0453, 0.013],[0.707, 0.707, 0.   , 0.   ]], hole_goal =  [[0.55,0.08, 0], [1,0,0,0]]):
        if visualize:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        
        #for testing
        self.handonly = handonly
        p.setGravity(0,0,-9.8)
        self.in_hand = []
        self.setup_robot()
        self.steps_taken = 0
        self.franka_tool_to_pb_link = 0.055 #measured empirically
        self.setup_workspace(rectangle_loc=rectangle_loc, circle_loc=circle_loc, board_loc=board_loc, obstacle_loc=obstacle_loc, hole_goal = hole_goal, load_previous=load_previous)

    """
    spawns a franka arm, eventually a FrankaArm object
    """
    def setup_robot(self):
        if not self.handonly:
            #self.robot = p.loadURDF(os.environ["HOME"]+"/ros/src/franka_ros/franka_description/robots/model.urdf") #fixme, point somewhere less fragile
            self.robot = p.loadURDF("../../models/robots/model.urdf")
            set_point(self.robot, (0,0,0.01))
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
    def setup_workspace(self, rectangle_loc = [[0.562, 0.003, 0.016], [1, 0.   ,  0.   ,  0   ]], load_previous=False, 
            circle_loc = [[0.425, 0.101, 0.01 ],[1, 0.  ,  0.  ,  0  ]],
            obstacle_loc = [[ 0.53172045, -0.03062703,  0.07507126], [1, -0.   ,  0.   ,  0   ]], board_loc = [[0.479, 0.0453, 0.013],[0.707, 0.707, 0.   , 0.   ]], hole_goal = [[0.55, 0.08, 0.0],[1,0,0,0]]):
        #RigidTransform(rotation=np.array([[-5.78152806e-02, -9.98327119e-01,  4.84639353e-07],
        #       [-9.98327425e-01,  5.78157598e-02,  3.97392158e-08],
        #              [ 4.07518811e-07, -6.59092487e-08, -9.99999635e-01]]), translation=np.array([ 0.53810962,  0.08998347, -0.00768057]), from_frame='peg_center', to_frame='world')
        #hole_pose = (np.array([ 0.53810962,  0.08998347, -0.00768057]), np.array([-2.89198577e-02, -9.19833769e-08,  1.37694750e-07,  9.99581685e-01]))
        #self.floor = p.loadURDF("../../models/short_floor.urdf")
        #make board
        width = 0.4
        length = 0.3
        fake_board_thickness = 0.05
        height = 0.01
        block_height = 0.01
        self.hole_depth = block_height
        self.block_height = block_height
        self.board = ut.create_box(width, length, height+fake_board_thickness, color = (1,0.7,0,1)) 
        board_loc[0][-1] -= 0.5*fake_board_thickness
        ut.set_pose(self.board, board_loc)
        #make circle
        radius = 0.078/2
        self.circle = ut.create_cylinder(radius, block_height, color = (1,1,0,1))
        #make rectangle
        self.rectangle =  ut.create_box(0.092, 0.069, block_height, color=(0.5,0,0.1,1))
        self.hole = ut.create_box(0.092, 0.069, 0.001, color = (0.1,0,0,1))
        board_z = 0.013+0.005
        #The perception z axis estimates are bad so let's use prior information to give it the right pose  
        rectangle_loc[0][-1] = board_z+0.5*block_height
        circle_loc[0][-1] = board_z+0.5*block_height
        hole_goal[0][-1] = board_z+0.5*0.001
        self.hole_goal = hole_goal
        if load_previous:
            rectangle_loc = np.load("saves/rectangle_loc.npy", allow_pickle=True)
            circle_loc = np.load("saves/circle_loc.npy", allow_pickle=True)
            obstacle_loc = np.load("saves/obstacle_loc.npy",allow_pickle=True)
            hole_goal = np.load("saves/hole_loc.npy", allow_pickle=True)
        else:
            np.save("saves/rectangle_loc.npy", rectangle_loc)
            np.save("saves/circle_loc.npy", circle_loc)
            np.save("saves/obstacle_loc.npy", obstacle_loc)
            np.save("saves/hole_loc.npy", hole_goal)
        
        if obstacle_loc is not None and obstacle_loc.any() is not None:
            self.obstacle = ut.create_box(0.08, 0.04, 0.1, color = (0.5,0.5,0.5,1))
            obstacle_loc[0][-1] = board_z+0.5*0.1
        else:
            self.obstacle = None
        ut.set_pose(self.rectangle, rectangle_loc)
        ut.set_pose(self.circle, circle_loc)
        self.shape_name_to_shape = {}
        if self.obstacle is not None:
            ut.set_pose(self.obstacle, obstacle_loc)
            self.shape_name_to_shape[Obstacle] = self.obstacle
        ut.set_pose(self.hole, hole_goal)
        self.shape_name_to_shape[Circle] = self.circle
        self.shape_name_to_shape[Rectangle] = self.rectangle
        input("workspace okay?")
        p.saveBullet("curr_state.bt")

    def get_closest_ee_goals(self, shape_goal, shape_class=Rectangle, grasp_offset = 0.055):
        #symmetry that minimizes the distance between shape_goal and the current ee pose. 
        curr_quat = Quaternion(p.getLinkState(self.robot, self.grasp_joint)[1])
        goal_quat = Quaternion(shape_goal[1])
        syms = shape_class.grasp_symmetries()
        transformation_lib_quats = [quaternion_about_axis(sym, (0,0,1))for sym in syms]
        sym_quats = [Quaternion(np.hstack([lib_quat[-1],lib_quat[0:3]])) for lib_quat in transformation_lib_quats]
        best_sym_idx = np.argmin([Quaternion.absolute_distance(curr_quat,Quaternion(matrix=np.dot(sym_quat.rotation_matrix,goal_quat.rotation_matrix))) for sym_quat in sym_quats
])
        best_sym = syms[best_sym_idx]
        #best_quat = Quaternion(quaternion_about_axis(best_sym, (0,0,1))).rotate(goal_quat) 
        #assert(Quaternion.absolute_distance(curr_quat, best_quat) < 0.01)
        best_quat = curr_quat #hack until I can get the angle stuff working
         
        #self.grasp_offset = grasp_offset+self.block_height*0.5
        grasp_translation =np.array([0,0,self.grasp_offset])
        ee_trans = grasp_translation+shape_goal[0]
        grasp_rot = np.dot(curr_quat.rotation_matrix, np.linalg.inv(best_quat.rotation_matrix))
        grasp_quat = np.array([1,0,0,0])# hack until I can get the angle stuff working Quaternion(matrix=grasp_rot).elements
        grasp = (grasp_translation, grasp_quat)
        ee_pose = (ee_trans, best_quat.elements)
        return ee_pose, grasp


        


    """
    Trajectory from current pose to goal considering attachments
    Avoids obstacles. This is our planner. At the moment it does not consider any transition model uncertainty. Will update after experimental results
    """
    def make_traj(self, goal_pose, shape_class=Rectangle, in_board = False):
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
        obstacles = [obs for obs in [self.obstacle, self.circle] if obs is not None]
        if not in_board:
            obstacles.append(self.board)
        disabled_collisions =  []
        if len(self.in_hand) == 1:
            disabled_collisions.append((self.in_hand[0].child, self.board))
            disabled_collisions.append((self.board, self.in_hand[0].child))
        traj = ut.plan_joint_motion(self.robot, joints_to_plan_for, end_conf, obstacles=obstacles, attachments=self.in_hand,
                      self_collisions=True, disabled_collisions=set(disabled_collisions),
                      weights=None, resolutions=None, smooth=100, restarts=5, iterations=100)
        
        p.restoreState(state)
        return traj
    def attach_shape(self, shape_name, grasp_pose):
        self.grasp = grasp_pose
        world_from_robot = ut.get_link_pose(self.robot, self.grasp_joint)
        world_from_obj = ut.get_link_pose(self.shape_name_to_shape[shape_name], -1)
        grasp_quat = ut.multiply(ut.invert(world_from_robot), world_from_obj)[1]
        self.grasp = (self.grasp[0], grasp_quat)
        attachment = ut.Attachment(self.robot, self.grasp_joint, self.grasp, self.shape_name_to_shape[shape_name])
        new_obj_pose = np.array(p.getLinkState(self.robot, self.grasp_joint)[0])+self.grasp[0]
        #ut.set_point(self.shape_name_to_shape[shape_name], new_obj_pose)
        attachment.assign()
        self.in_hand = [attachment]

    def fk_rigidtransform(self, joint_vals, return_rt=True):
        state = p.saveState()
        ut.set_joint_positions(self.robot, ut.get_movable_joints(self.robot)[:len(joint_vals)], joint_vals)
        link_trans, link_quat = ut.get_link_pose(self.robot, self.grasp_joint)
        p.restoreState(state)

        if return_rt:
            link_trans = list(link_trans)
            link_trans[-1] -= 0.055 #conversion to franka_tool
            link_quat = np.hstack([link_quat[-1],link_quat[0:3]])
            rt = RigidTransform(translation=link_trans, rotation=Quaternion(link_quat).rotation_matrix)
            return rt
        else:
            return (link_trans, link_quat)

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
        shape_goal = p.getBasePositionAndOrientation(self.rectangle)
        traj, grasp = self.sample_trajs(shape_goal, shape_class=shape_class)
        n_pts = 5
        mask = np.round(np.linspace(0,len(traj)-1, n_pts)).astype(np.int32)
        traj = np.array(traj)[mask] 
        if visualize:
            self.visualize_traj(traj)
        self.attach_shape(Rectangle, grasp)
        assert(traj is not None)
        return traj

    def sample_trajs(self, goal, shape_class=Rectangle, placing=False):
        state = p.saveState()
        sample_joint = 6
        original = ut.get_joint_position(self.robot, sample_joint)
        if placing:
            shape_frame_symmetries = shape_class.placement_symmetries()
            #shift the symmetrices by the grasp, since those are the actual angles we are sampling, not the robot joint angles. 
            symmetries = []
            for sampled_angle in shape_frame_symmetries:
                grasp_quat = self.grasp[1]
                grasp_yaw = ut.euler_from_quat(grasp_quat)[-1]
                symmetries.append(sampled_angle+grasp_yaw)
        else:
            symmetries =  shape_class.grasp_symmetries()
        self.grasp_offset = 0.010+self.franka_tool_to_pb_link
        #All of this is to make sure the grasps are within joint limits 
        ee_goals = []
        grasps = []
        joint_angles = []
        for sym in symmetries:
            if original+sym < 3 and original+sym > -3:
                ut.set_joint_position(self.robot, sample_joint, original+sym)
                curr_pos  = ut.get_joint_position(self.robot, sample_joint)
                ee_goal, grasp = self.get_closest_ee_goals(goal, shape_class=Rectangle, grasp_offset = self.grasp_offset)
                ee_goals.append(ee_goal)
                grasps.append(grasp)
                joint_angles.append(original+sym)
                
        p.restoreState(state)
        #grasp = grasp_from_ee_and_obj(ee_goal, shape_goal)
        working_grasp = []
        working_traj = []
        working_joint_angles = []
        for ee_goal, grasp, joint_angle in zip(ee_goals, grasps, joint_angles):
            grasp_traj = self.make_traj(ee_goal)
            if grasp_traj is not None:
                working_grasp.append(grasp) 
                working_traj.append(grasp_traj)
                working_joint_angles.append(joint_angle)
        assert(len(working_traj) > 0)
        min_joint_angle_idx = np.argmin(working_joint_angles)
        return working_traj[min_joint_angle_idx], working_grasp[min_joint_angle_idx]
    def close(self):
        p.disconnect()
    """
    Collision-free trajectory  to place object in hole
    """
    def place_object(self, visualize=False, shape_class=Rectangle, hole_goal=None, push_down=True):
        if hole_goal is None:
            hole_goal = self.hole_goal.copy()
            hole_goal[0][-1] = 0.03
        else:
            np.save("custom_hole_goal.npy", hole_goal)
        traj, grasp = self.sample_trajs(hole_goal, shape_class=Rectangle, placing=True) 
        if push_down:
            state = p.saveState()
            #Move in that last bit
            joint_vals = traj[-1]
            ut.set_joint_positions(self.robot, ut.get_movable_joints(self.robot)[:len(joint_vals)], joint_vals)
            goal_pose = self.fk_rigidtransform(traj[-1], return_rt=False)
            new_trans = (goal_pose[0][0], goal_pose[0][1], goal_pose[0][2]-0.7*self.hole_depth)
            end_traj = self.make_traj((new_trans, goal_pose[1]),in_board = True)
            print(len(end_traj), "length end traj")

            p.restoreState(state)
            traj += end_traj
        # end moving in that last bit
        if visualize:
            self.visualize_traj(traj)
        n_pts = min(10, len(traj))
        mask = np.round(np.linspace(0,len(traj)-1, n_pts)).astype(np.int32)
        traj = np.array(traj)[mask] 
        assert(traj is not None)
        return traj

if __name__ == "__main__":
    #pw = PegWorld( visualize=False, handonly=False)
    pw = PegWorld(visualize=True, handonly = False, load_previous=True)
    pw.grasp_object(shape_class=Rectangle, visualize=True)
    #pw.visualize_points(np.load("../real_robot/data/bad_odel_states.npy", allow_pickle=True)[:,0:3])
    pw.place_object(shape_class=Rectangle, visualize=True)
