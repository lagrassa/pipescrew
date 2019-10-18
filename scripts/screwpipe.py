from pipeworld import PipeWorld
from collections import deque
import pybullet as p
import numpy as np
from pybullet_tools.utils import plan_joint_motion, joint_controller, inverse_kinematics_helper, get_pose, joint_from_name, simulate_for_duration, get_movable_joints, set_joint_positions, control_joints, Attachment, create_attachment
class PipeGraspAgent():
    def __init__(self, visualize=True):
        self.pw = PipeWorld(visualize=visualize)
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
        
    def place(self):
        grasp = np.array([0,0,0.3])
        target_point = np.array(get_pose(self.pw.hollow))[0]+grasp
        target_quat = (1,0.5,0,0)
        target_pose = (target_point, target_quat)
        self.go_to_pose(target_pose, obstacles=[self.pw.hollow], attachments=[self.pipe_attach])

        
    def insert(self):
        target_quat = (1,0.5,0,0) #get whatever it is by default
        grasp = np.array([0,0,0.18])
        target_point = np.array(get_pose(self.pw.hollow))[0]+grasp
        target_pose = (target_point, target_quat)
        self.go_to_pose(target_pose, obstacles=[], attachments=[self.pipe_attach])

    def change_grip(self,num, force=0):
        target = (num,num)
        if force == 0:
            next(joint_controller(self.pw.robot, self.finger_joints, target))
        else:
            self.squeeze(force)


    def go_to_conf(self, conf):
        control_joints(self.pw.robot, get_movable_joints(self.pw.robot)+list(self.finger_joints), tuple(conf)+(self.target_grip,self.target_grip))
        simulate_for_duration(0.2)
        print("gripper force", self.get_gripper_force())

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
            print("curr force",curr_force)
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
                print("achieved at i= ", i)
                break 
        if n_tries == i-1:
            print("Failure to reach", diff)

    def get_gripper_force(self):
        left = np.linalg.norm(p.getJointState(self.pw.robot,9)[2][3:])
        right = np.linalg.norm(p.getJointState(self.pw.robot,10)[2][3:])
        curr_force = (left+right)/2 
        return curr_force

        
    def go_to_pose(self,target_pose, obstacles=[], attachments=[]):
        for i in range(10):
            end_conf = inverse_kinematics_helper(self.pw.robot, self.ee_link, target_pose)
            motion_plan = plan_joint_motion(self.pw.robot, get_movable_joints(self.pw.robot), end_conf, obstacles = obstacles, attachments=attachments)
            if motion_plan is not None:
                for conf in motion_plan:
                    self.go_to_conf(conf)
                    if self.pipe_attach is not None:
                        self.squeeze(force=self.squeeze_force)
            ee_loc = p.getLinkState(self.pw.robot, 8)[0]
            distance = np.linalg.norm(np.array(ee_loc)-target_pose[0])
            if distance < 1e-3:
                break



pga = PipeGraspAgent(visualize=True)
pga.change_grip(0.025, force=0)
pga.approach()
pga.change_grip(0.005, force=0.7) # force control this. 1 was ok
simulate_for_duration(0.5)
pga.place()
pga.insert()

