from pipeworld import PipeWorld
from simple_model import train_policy
from history import History
from collections import deque
import pybullet as p
import numpy as np
from pybullet_tools.utils import plan_joint_motion, joint_controller, inverse_kinematics_helper, get_pose, joint_from_name, simulate_for_duration, get_movable_joints, set_joint_positions, control_joints, Attachment, create_attachment
class PipeGraspAgent():
    def __init__(self, visualize=True, bullet=None):
        self.pw = PipeWorld(visualize=visualize, bullet=bullet)
            
        self.ee_link =8# 8#joint_from_name(self.pw.robot, "panda_hand_joint")
        self.joints=[1,2,3,4,5,6,7]
        self.target_grip = 0.015
        self.pipe_attach=None
        self.finger_joints = (9,10)# [joint_from_name(self.pw.robot,"panda_finger_joint"+str(side)) for side in [1,2]]
        p.enableJointForceTorqueSensor(self.pw.robot, 9)
        p.enableJointForceTorqueSensor(self.pw.robot, 10)
        next(joint_controller(self.pw.robot, self.joints, [ 0.    ,  0.    , -1.5708,  0.    ,  1.8675,  0.    , 0   ]))


    def approach(self):
        grasp = np.array([0,0,0.2])
        target_point_1 = np.array(get_pose(self.pw.pipe))[0]+grasp
        grasp = np.array([0,0,0.13])
        target_point_2 = np.array(get_pose(self.pw.pipe))[0]+grasp
        target_quat = (1,0.5,0,0) #get whatever it is by default
        target_pose = (target_point_1, target_quat)
        obstacles = [self.pw.pipe, self.pw.hollow]
        self.go_to_pose(target_pose, obstacles=obstacles)
        target_pose = (target_point_2, target_quat)
        self.go_to_pose(target_pose, obstacles=obstacles)
        self.pipe_attach = create_attachment(self.pw.robot, self.ee_link, self.pw.pipe)
        
    def place(self, use_policy=False):
        grasp = np.array([0,0,0.3])
        target_point = np.array(get_pose(self.pw.hollow))[0]+grasp
        target_quat = (1,0.5,0,0)
        target_pose = (target_point, target_quat)
        traj = self.go_to_pose(target_pose, obstacles=[self.pw.hollow], attachments=[self.pipe_attach], cart_traj=True, use_policy=use_policy)
        return traj
        
    def insert(self, use_policy=False):
        target_quat = (1,0.5,0,0) #get whatever it is by default
        grasp = np.array([0,0,0.18])
        target_point = np.array(get_pose(self.pw.hollow))[0]+grasp
        target_pose = (target_point, target_quat)
        traj = self.go_to_pose(target_pose, obstacles=[], attachments=[self.pipe_attach], use_policy=use_policy, cart_traj=True)
        #traj = self.go_to_pose(target_pose, obstacles=[], attachments=[self.pipe_attach], use_policy=use_policy, cart_traj=True)
        traj = self.go_to_pose_impedance(target_pose)
        return traj

    def change_grip(self,num, force=0):
        target = (num,num)
        if force == 0:
            next(joint_controller(self.pw.robot, self.finger_joints, target))
        else:
            self.squeeze(force)


    def go_to_conf(self, conf):
        control_joints(self.pw.robot, get_movable_joints(self.pw.robot)+list(self.finger_joints), tuple(conf)+(self.target_grip,self.target_grip))
        simulate_for_duration(0.2)

    def squeeze(self, force):
        self.squeeze_force = force
        diffs = deque(maxlen=5)
        k=0.00052
        kp = 0.00008# 0.00015
        n_tries = 400
        tol = 0.1
        for i in range(n_tries):
            curr_force = self.get_gripper_force()
            diff = force-curr_force
            diffs.append(curr_force)
            if len(diffs) == 2:
                deriv_diff= diffs[1]-diffs[0]
            else:
                deriv_diff=0
            curr_pos = p.getJointState(self.pw.robot,9)[0]
            target = curr_pos - (k*diff+kp*(deriv_diff))
            control_joints(self.pw.robot, self.finger_joints, (target,target))
            self.target_grip = target
            simulate_for_duration(0.1)
            
            if len(diffs) > 3 and abs(diff) < tol and abs(np.mean(list(diffs)[-3:])-force) < tol:
                break 
        if n_tries == i-1:
            print("Failure to reach", diff)

    def get_gripper_force(self):
        left = np.linalg.norm(p.getJointState(self.pw.robot,9)[2][3:])
        right = np.linalg.norm(p.getJointState(self.pw.robot,10)[2][3:])
        curr_force = (left+right)/2 
        return curr_force

    def go_to_pose_impedance(self, target_pose):
        #xdd = 
        K = np.eye(3)
        movable_joints = get_movable_joints(self.pw.robot)
        #q = get_joint_positions(self.pw.robot, movable_joints)
        #qdot = get_joint_velocities(self.pw.robot, movable_joints)
        #inertial_matrix = p.calculateMassMatrix(self.pw.robot, q) 
        joint_index = 8
        #jacobian = p.calculateJacobian(self.pw.robot, joint_index, [0,0,0], 
        #inertial_task_space_matrix = np.dot(jacobian, inertial_matrix)
        #pos_error = target_state - p.getLinkState(joint_index)[0]
        F = np.dot(K, pos_error)#+np.dot(inertial_task_space_matrix,xdd)
        #force in task space, inverse kinematics to get the torques
        #torques to do that
        torque = bullet.calculateInverseDynamics(id_robot, obj_pos, obj_vel, obj_acc)
        return p.setJointMotorControlArray(body, joints, p.TORQUE_CONTROL,
                                   forces = torque)
        
    def go_to_pose(self,target_pose, obstacles=[], attachments=[], cart_traj=False, use_policy = False):
        total_traj = []
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

    def is_pipe_in_hole(self):
        return False

    def collect_trajs(self):
        trajs = []
        successes = []
        self.do_setup()
        bullet_id = p.saveState()
        self.history = History()
        for i in range(2):
            p.restoreState(bullet_id)
            if i == 0:
                traj = self.insert()
            else:
                import ipdb; ipdb.set_trace()
                traj = self.insert(use_policy=True)
            successes.append(self.is_pipe_in_hole()) 
            trajs.append(traj)
            self.history.paths = trajs
            self.train_policy_with_trajs()

    def train_policy_with_trajs(self):
        self.policy = train_policy(self.history) 

        
         
    def do_setup(self):
        self.change_grip(0.025, force=0)
        self.approach()
        self.change_grip(0.005, force=0.7) # force control this. 1 was ok
        simulate_for_duration(0.1)
        self.place()



pga = PipeGraspAgent(visualize=True, bullet="place.bullet")
pga.collect_trajs()
pga.train_policy_with_trajs()
#pga.insert()
