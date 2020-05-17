from env.pipeworld import PipeWorld
import time
from agent.history import History
from collections import deque
import pybullet as p
from gym.spaces import Box
import numpy as np
from env.pb_utils import plan_joint_motion, joint_controller, inverse_kinematics_helper, get_pose, joint_from_name, simulate_for_duration, get_movable_joints, set_joint_positions, control_joints, Attachment, create_attachment
import env.pb_utils as ut
class PegInsertEnv():
    def __init__(self, visualize=True, bullet=None, shift=0, start=None, force_feedback_type = None, goal=None):
        self.pw = PipeWorld(visualize=visualize, bullet=bullet, square = True)
        self.pipe_attach=None
        self.target_grip = 0.015
        self.default_width=0.010
        self.total_timeout = 100
        self.steps_taken = 0
        self.horizon = 20
        self.dt_pose = 0.1
        self.force_feedback_type = force_feedback_type

        if self.pw.handonly:
            self.ee_link=-1
            self.finger_joints =(0,1)
            p.enableJointForceTorqueSensor(self.pw.robot, 0)
            p.enableJointForceTorqueSensor(self.pw.robot, 1)
        else:            
            self.ee_link =8# 8#joint_from_name(self.pw.robot, "panda_hand_joint")
            self.joints=[1,2,3,4,5,6,7]
            self.finger_joints = (9,10)# [joint_from_name(self.pw.robot,"panda_finger_joint"+str(side)) for side in [1,2]]
            p.enableJointForceTorqueSensor(self.pw.robot, 9)
            p.enableJointForceTorqueSensor(self.pw.robot, 10)
            next(joint_controller(self.pw.robot, self.joints, [ 0.    ,  0.    , -1.5708,  0.    ,  1.8675,  0.    , 0   ]))
        self._shift = shift
        self.do_setup()
        sample_obs = self.observe_state()
        obs_size = sample_obs.shape
        max_x = 0.05
        max_y = 0.04
        min_z = 0.02
        max_z = 0.9
        self.observation_space = Box(low=np.array([-max_x, -max_y, min_z]), high=np.array([max_x, max_y, max_z]))
        largest_delta = 0.05
        z_delta = 0.1
        max_force = 300
        min_force = 30
        self.action_space =Box(low=np.array([-largest_delta, -largest_delta, -z_delta, min_force]),
                               high=np.array([largest_delta, largest_delta, z_delta, max_force]))

    def plot_path(self, path):
        pass
    def approach(self):
        grasp = np.array([0,0,0.2])
        target_point_1 = np.array(get_pose(self.pw.pipe))[0]+grasp
        grasp = np.array([0,0,0.13])
        target_point_2 = np.array(get_pose(self.pw.pipe))[0]+grasp
        target_quat = (1,0,0,0) #get whatever it is by default
        target_pose_1 = (target_point_1, target_quat)
        if isinstance(self.pw.hollow, list):
            obstacles = [self.pw.pipe]+self.pw.hollow
        else:
            obstacles = [self.pw.pipe, self.pw.hollow]
        self.go_to_pose(target_pose_1, obstacles=obstacles, maxForce=400)
        target_pose_2 = (target_point_2, target_quat)
        self.go_to_pose(target_pose_2, obstacles=obstacles)
        self.pipe_attach = create_attachment(self.pw.robot, self.ee_link, self.pw.pipe)
        self.squeeze(0,width = self.default_width)
    def goto_state(self, state):
        #sets the state, or at least what can be set, which is the first part
        pose = state[0:3]
        self.go_to_pose((pose, (1,0,0,0)), obstacles=[], attachments=[self.pipe_attach],
                        cart_traj=True, use_policy=False)

    def collision_fn(self, pt):
        return False #TODO make this actually check collision        
    def center_peg(self, use_policy=False):
        grasp = np.array([0,0,0.24])
        hollow_pt = self.get_hole_point()
        target_point =hollow_pt+grasp
        self.hollow_pose = hollow_pt
        target_quat = (1,0,0,0)
        target_pose = (target_point, target_quat)
        lift_point = np.array(p.getLinkState(self.pw.robot, self.finger_joints[0])[0])+grasp
        traj1 = self.go_to_pose((lift_point, target_quat), obstacles=[self.pw.hollow], attachments=[self.pipe_attach], cart_traj=True, use_policy=use_policy)
        traj2 = self.go_to_pose(target_pose, obstacles=[self.pw.hollow], attachments=[self.pipe_attach], cart_traj=True, use_policy=use_policy)
        self.save_state()
    def close(self):
        p.disconnect()

    def get_grasp(self):
        #vector from robot to pipe. If this translation is added, it gives you the pipe pose
        finger_poses = [p.getLinkState(self.pw.robot, finger) for finger in self.finger_joints]
        middle_pose = np.mean(np.vstack(finger_poses),axis=0)
        return self.get_pos()-middle_pose

    def insert(self, use_policy=False, target_force = 1):
        target_quat = (1,0,0,0) #get whatever it is by default
        grasp = np.array([0,0,0.18])
        target_point = np.array(self.hollow_pose)[0]+grasp
        target_pose = (target_point, target_quat)
        traj = self.go_to_pose(target_pose, obstacles=[], attachments=[self.pipe_attach], use_policy=use_policy, maxForce = target_force,cart_traj=True)
        return traj

    def change_grip(self,num, force=0):
        target = (num,num)
        if force == 0:
            next(joint_controller(self.pw.robot, self.finger_joints, target))
        else:
            self.squeeze(force)


    def go_to_conf(self, conf):
        control_joints(self.pw.robot, get_movable_joints(self.pw.robot)+list(self.finger_joints), tuple(conf)+(self.target_grip,self.target_grip))
        simulate_for_duration(0.3)
        self.steps_taken += 0.3

    def squeeze(self, force, width=None):
        self.squeeze_force = force
        diffs = deque(maxlen=5)
        k=0.00058
        kp = 0#0.00001# 0.00015
        n_tries = 400
        tol = 0.1
        for i in range(n_tries):
            if width is None:
                curr_force = self.get_gripper_force()
                diff = force-curr_force
                #print("diff", diff)
                diffs.append(curr_force)
                if len(diffs) == 2:
                    deriv_diff= diffs[1]-diffs[0]
                else:
                    deriv_diff=0
                curr_pos = p.getJointState(self.pw.robot,self.finger_joints[0])[0]
                target = curr_pos - (k*diff+kp*(deriv_diff))
            else:
                target =width
            control_joints(self.pw.robot, self.finger_joints, (target,target))
            self.target_grip = target
            if width is not None:
                simulate_for_duration(0.5)
                self.pw.steps_taken += 0.5 
                if self.pw.steps_taken >= self.total_timeout:
                    return
                break
            else:
                simulate_for_duration(0.1) #less sure of it
                self.pw.steps_taken += 0.1 
                if self.pw.steps_taken >= self.total_timeout:
                    return

            
            if len(diffs) > 3 and abs(diff) < tol and abs(np.mean(list(diffs)[-3:])-force) < tol:
                #print("Reached target force")
                break 
        if n_tries == i-1:
            print("Failure to reach", diff)

    def get_gripper_force(self):
        left = np.linalg.norm(p.getJointState(self.pw.robot,self.finger_joints[0])[2][3:])
        right = np.linalg.norm(p.getJointState(self.pw.robot,self.finger_joints[1])[2][3:])
        curr_force = (left+right)/2 
        return curr_force
    """
    action = (delta x, delta y, delta z, force)
    """
    def step(self, action):
        dx, dy, dz, max_force = action
        target_point = np.array(p.getBasePositionAndOrientation(self.pw.robot)[0])+np.array([dx, dy, dz])
        target_pose = (target_point, (1,0,0,0))
        self.go_to_pose(target_pose, obstacles=[], attachments=[], cart_traj=True, use_policy = False, maxForce = max_force)
        ob = self.observe_state()
        rew = int(self.is_pipe_in_hole())
        return ob, rew, False, {}
    
    def get_pos(self): 
        return np.array(p.getBasePositionAndOrientation(self.pw.pipe)[0])
    def get_qqdot(self):
        """

        :return: qqdot of the pipe
        """
        vel = p.getLinkState(self.pw.pipe, -1, computeLinkVelocity=1)[7]
        return np.hstack([self.get_pos(), vel]).flatten()

    """
    force_feedback_type is None, binary, discrete, cont
    discrete is presence/absence and magnitude
    """
    def observe_state(self):
        force_feedback = p.getJointState(self.pw.robot, self.finger_joints[0])[2] #should be symmetric across fingers, hence picking one
        binary_threshold = 1.5 #arbitrary really
        binary_force_feedback = (np.abs(force_feedback) > binary_threshold).astype(np.int32)
        if self.force_feedback_type == None:
            return self.get_pos()
        elif self.force_feedback_type == "cont":
            return np.hstack([self.get_pos(),force_feedback]).flatten()
        elif self.force_feedback_type == "binary":
            return np.hstack([self.get_pos(),binary_force_feedback]).flatten()
        elif self.force_feedback_type == "discrete":
            discrete_force_feedback =np.array(force_feedback).astype(np.int32)
            return np.hstack([self.get_pos(),discrete_force_feedback]).flatten()
        else:
            return self.get_pos()

    def reset(self):
        self.restore_state()
        #might want to add some randomness to the starting pose, but let's start with totally centered
        return self.observe_state(self.force_feedback_type)

    def go_to_pose(self,target_pose, obstacles=[], attachments=[], cart_traj=False, use_policy = False, maxForce = 100):
        total_traj = []
        if self.pw.handonly:
            p.changeConstraint(self.pw.cid, target_pose[0], target_pose[1], maxForce = maxForce)
            for i in range(80):
                simulate_for_duration(self.dt_pose)
                self.pw.steps_taken += self.dt_pose 
                if self.pw.steps_taken >= self.total_timeout:
                    return total_traj

                ee_loc = p.getBasePositionAndOrientation(self.pw.robot)[0]
                distance = np.linalg.norm(np.array(ee_loc)-target_pose[0])
                if distance < 1e-3:
                    break
                total_traj.append(ee_loc)
                if self.pipe_attach is not None:
                    self.squeeze(force=self.squeeze_force)


        else:
            for i in range(50):
                end_conf = inverse_kinematics_helper(self.pw.robot, self.ee_link, target_pose)
                if not use_policy:
                    motion_plan = plan_joint_motion(self.pw.robot, get_movable_joints(self.pw.robot), end_conf, obstacles = obstacles, attachments=attachments)
                    if motion_plan is not None:
                        for conf in motion_plan:
                            self.go_to_conf(conf)
                            ee_loc = p.getLinkState(self.pw.robot, 8)
                            if cart_traj:
                                total_traj.append(ee_loc[0]+ee_loc[1])
                            else:
                                total_traj.append(conf)
                            if self.pipe_attach is not None:
                                self.squeeze(force=self.squeeze_force)
                else:
                    ee_loc = p.getLinkState(self.pw.robot, 8)
                    next_loc = self.policy.predict(np.array(ee_loc[0]+ee_loc[1]).reshape(1,7))[0]
                    next_pos = next_loc[0:3]
                    next_quat = next_loc[3:]
                    next_conf = inverse_kinematics_helper(self.pw.robot, self.ee_link, (next_pos,next_quat)) 
                    if cart_traj:
                        total_traj.append(next_loc)
                    else:
                        total_traj.append(next_conf)
                    self.go_to_conf(next_conf)
                    if self.pipe_attach is not None:
                        self.squeeze(force=self.squeeze_force)
                    
                ee_loc = p.getLinkState(self.pw.robot, 8)[0]
                distance = np.linalg.norm(np.array(ee_loc)-target_pose[0])
                if distance < 1e-3:
                    break
        return total_traj
    def get_hole_point(self):
        if isinstance(self.pw.hollow,list):
            hollow_pt = np.mean([get_pose(obj)[0] for obj in self.pw.hollow],axis=0)
            return hollow_pt
        else:
            return np.array(p.getBasePositionAndOrientation(self.pw.hollow)[0])

    def goal_condition_met(self):
        return self.is_pipe_in_hole()
    def is_pipe_in_hole(self):
        pose_hollow = self.get_hole_point()
        pose_pipe, _ = p.getBasePositionAndOrientation(self.pw.pipe)
        xy_dist = np.linalg.norm(np.array(pose_pipe)[0:2]-np.array(pose_hollow)[0:2])
        z_dist = np.linalg.norm(pose_pipe[2]-pose_hollow[2])
        xy_threshold = 0.008
        z_threshold = 0.07
        if xy_dist < xy_threshold and z_dist < z_threshold:
            return True
        return False
    def restore_state(self):
        p.restoreState(self.bullet_id)

    def reset(self):
        self.steps_taken = 0
        self.restore_state()

    def save_state(self):
        self.bullet_id = p.saveState()

    def close(self):
        p.disconnect()
         
    def do_setup(self):
        self.change_grip(0.025, force=0)
        self.approach()
        self.change_grip(0.005, force=0.8) # force control this. 1 was ok
        simulate_for_duration(0.1)
        self.center_peg()
        simulate_for_duration(0.5)
        self.steps_taken += 0.5
        self.save_state()


if __name__ == "__main__":
    pga = PegInsertEnv(visualize=True, bullet="place.bullet")
    pga.reset()

    action = [0,0,-0.05,100]
    for i in range(2):
        pga.step(action)
    import ipdb; ipdb.set_trace()
    print("Success?", pga.is_pipe_in_hole())
    pga.reset()

