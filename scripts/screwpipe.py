from pipeworld import PipeWorld
import pybullet as p
import numpy as np
from pybullet_tools.utils import plan_joint_motion, joint_controller, inverse_kinematics_helper, get_pose, joint_from_name, simulate_for_duration, get_movable_joints, set_joint_positions, control_joints, Attachment, create_attachment
class PipeGraspAgent():
    def __init__(self, visualize=True):
        self.pw = PipeWorld(visualize=visualize)
        self.ee_link =8# 8#joint_from_name(self.pw.robot, "panda_hand_joint")
        self.joints=[1,2,3,4,5,6,7]
        self.pipe_attach=None
        self.finger_joints = (9,10)# [joint_from_name(self.pw.robot,"panda_finger_joint"+str(side)) for side in [1,2]]
        p.enableJointForceTorqueSensor(self.pw.robot, 9)
        p.enableJointForceTorqueSensor(self.pw.robot, 10)
        next(joint_controller(self.pw.robot, self.joints, [ 0.    ,  0.    , -1.5708,  0.    ,  1.8675,  0.    , 0   ]))


    def approach(self):
        grasp = np.array([0,0,0.15])
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
        grasp = np.array([0,0,0.28])
        target_point = np.array(get_pose(self.pw.hollow))[0]+grasp
        target_quat = (1,0.5,0,0)
        target_pose = (target_point, target_quat)
        self.go_to_pose(target_pose, obstacles=[self.pw.hollow], attachments=[self.pipe_attach])

        
    def insert(self):
        target_quat = (1,0.5,0,0) #get whatever it is by default
        grasp = np.array([0,0,0.22])
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
        control_joints(self.pw.robot, get_movable_joints(self.pw.robot), conf)
        simulate_for_duration(0.2)

    def squeeze(self, force):
        for i in range(220):
            left = np.linalg.norm(p.getJointState(self.pw.robot,9)[2][3:])
            right = np.linalg.norm(p.getJointState(self.pw.robot,10)[2][3:])
            curr_force = (left+right)/2 
            diff = force-curr_force
            curr_pos = p.getJointState(self.pw.robot,9)[0]
            k=0.0002
            target = curr_pos - k*diff
            control_joints(self.pw.robot, self.finger_joints, (target,target))
            simulate_for_duration(0.05)
            if abs(diff) < 0.1:
                break 

        
        
    def go_to_pose(self,target_pose, obstacles=[], attachments=[]):
        for i in range(10):
            end_conf = inverse_kinematics_helper(self.pw.robot, self.ee_link, target_pose)
            motion_plan = plan_joint_motion(self.pw.robot, get_movable_joints(self.pw.robot), end_conf, obstacles = obstacles, attachments=attachments)
            if motion_plan is not None:
                for conf in motion_plan:
                    self.go_to_conf(conf)
                    if self.pipe_attach is not None:
                        self.squeeze(force=1.8)
            ee_loc = p.getLinkState(self.pw.robot, 8)[0]
            distance = np.linalg.norm(np.array(ee_loc)-target_pose[0])
            print("distance", distance)
            if distance < 1e-3:
                break



pga = PipeGraspAgent(visualize=True)
pga.change_grip(0.025, force=0)
pga.approach()
pga.change_grip(0.005, force=1.8) # force control this. 1 was ok
pga.place()
pga.insert()
