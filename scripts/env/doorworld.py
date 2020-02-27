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
            obstacle_loc = [[ 0.62172045, -0.04062703,  0.07507126], [1, -0.   ,  0.   ,  0   ]], board_loc = [[0.479, 0.0453, 0.013],[0.707, 0.707, 0.   , 0.   ]], hole_goal =  [[0.55,0.08, 0], [1,0,0,0]]):
        if visualize:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
         
        self.handonly = handonly
        p.setGravity(0,0,-9.8)
        self.in_hand = []
        self.setup_robot()
        self.steps_taken = 0
        self.franka_tool_to_pb_link = 0.055 #measured empirically
        self.setup_workspace()

    """
    spawns a franka arm, eventually a FrankaArm object
    """
    def setup_robot(self):
        if not self.handonly:
            #self.robot = p.loadURDF(os.environ["HOME"]+"/ros/src/franka_ros/franka_description/robots/model.urdf") #fixme, point somewhere less fragile
            self.robot = p.loadURDF("../../models/robots/model.urdf")
            set_point(self.robot, (0,0,0.01))
            start_joints = [2.28650215, -1.36063288, -1.4431576, -1.93011263,0.23962597,  2.6992652,  0.82820212,0.03,0.03]
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
    def setup_workspace(self):
        self.door = p.loadURDF("../../models/door.urdf")
        door_pt = ([0.31, 0.46935375, 0])
        door_quat = (.707,0,0,.707)
        ut.set_pose(self.door,(door_pt, door_quat))
        import ipdb; ipdb.set_trace()

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
        best_quat = Quaternion(0.707, 0.707, 0, 0)#curr_quat #hack until I can get the angle stuff working
         
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
       
        end_conf = ut.inverse_kinematics(self.robot, self.grasp_joint, goal_pose, movable_joints=joints_to_plan_for, null_space=null_space) #end conf to be in the goal loc
        if end_conf is None:
           end_conf = ut.inverse_kinematics(self.robot, self.grasp_joint, goal_pose, movable_joints=joints_to_plan_for, null_space=None) #end conf to be in the goal loc
        import ipdb; ipdb.set_trace()
        if end_conf is None:
            print("No IK solution found")
            p.restoreState(state) 
            return None
        end_conf = end_conf[:len(joints_to_plan_for)]
        p.restoreState(state)
        #for attachment in self.in_hand:
        #    attachment.assign()
        obstacles = [self.obstacle, self.circle]
        if not in_board:
            obstacles.append(self.board)
        traj = ut.plan_joint_motion(self.robot, joints_to_plan_for, end_conf, obstacles=obstacles, attachments=self.in_hand,
                      self_collisions=True, disabled_collisions=set(self.in_hand),
                      weights=None, resolutions=None, smooth=100, restarts=5, iterations=100)
        
        p.restoreState(state)
        return traj
    def attach_shape(self, shape_name, grasp_pose):
        self.grasp = grasp_pose
        attachment = ut.Attachment(self.robot, self.grasp_joint, grasp_pose, self.shape_name_to_shape[shape_name])
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

    def sample_trajs(self, goal, shape_class=Rectangle):
        ee_goals = []
        grasps = []
        state = p.saveState()
        sample_joint = 6
        original = ut.get_joint_position(self.robot, sample_joint)
        grasp_symmetries =  [-np.pi/2, 0, np.pi/2] # fix eventually shape_class.grasp_symmetries()
        self.grasp_offset = 0.010+self.franka_tool_to_pb_link
        for sym in grasp_symmetries:
            if original+sym < 3 and original+sym > -3:
                ut.set_joint_position(self.robot, sample_joint, original+sym)
                curr_pos  = ut.get_joint_position(self.robot, sample_joint)
                ee_goal, grasp = self.get_closest_ee_goals(goal, shape_class=Rectangle, grasp_offset = self.grasp_offset)
                ee_goals.append(ee_goal)
                grasps.append(grasp)
        p.restoreState(state)
        #grasp = grasp_from_ee_and_obj(ee_goal, shape_goal)
        working_grasp = None
        working_traj = None
        for ee_goal, grasp in zip(ee_goals, grasps):
            grasp_traj = self.make_traj(ee_goal)
            if grasp_traj is not None:
                working_grasp = grasp 
                working_traj = grasp_traj
        assert(working_traj is not None)
        return working_traj, working_grasp
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
        traj, grasp = self.sample_trajs(hole_goal, shape_class=Rectangle) 
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
    pw = PegWorld(visualize=True, handonly = False, load_previous=False)
    pw.grasp_object(shape_class=Rectangle, visualize=True)
    #pw.visualize_points(np.load("../real_robot/data/bad_odel_states.npy", allow_pickle=True)[:,0:3])
    pw.place_object(shape_class=Rectangle, visualize=True)
