from pipeworld import PipeWorld
import pybullet as p
import numpy as np
from pybullet_tools.utils import plan_joint_motion, joint_controller, inverse_kinematics_helper, get_pose, joint_from_name, simulate_for_duration, get_movable_joints, set_joint_positions
class PipeGraspAgent():
    def __init__(self, visualize=True):
        self.pw = PipeWorld(visualize=visualize)
        self.ee_link =8# 8#joint_from_name(self.pw.robot, "panda_hand_joint")
        self.joints=[1,2,3,4,5,6,7]
        self.finger_joints = (9,10)# [joint_from_name(self.pw.robot,"panda_finger_joint"+str(side)) for side in [1,2]]
        next(joint_controller(self.pw.robot, self.joints, [ 0.    ,  0.    , -1.5708,  0.    ,  1.8675,  0.    , 0   ]))


    def approach(self):
        grasp = np.array([0,0,0.15])
        target_point = np.array(get_pose(self.pw.pipe))[0]+grasp
        target_quat = (1,0.5,0,0) #get whatever it is by default
        target_pose = (target_point, target_quat)
        for i in range(10):
            end_conf = inverse_kinematics_helper(self.pw.robot, self.ee_link, target_pose)
            set_joint_positions(self.pw.robot, get_movable_joints(self.pw.robot), end_conf)
            ee_loc = p.getLinkState(self.pw.robot, 8)[0]
            distance = np.linalg.norm(np.array(ee_loc)-target_pose[0])
            if distance < 1e-3:
                break
        

        #motion_plan = plan_joint_motion(self.pw.robot, self.joints, end_conf[:len(self.joints)])
        #for conf in motion_plan:
        #    next(joint_controller(self.pw.robot, self.joints, conf))
        
    def insert(self):
        pass

    def change_grip(self,num):
        target = (num,num)
        next(joint_controller(self.pw.robot, self.finger_joints, target))



pga = PipeGraspAgent(visualize=True)
pga.change_grip(0.02)
pga.approach()
pga.change_grip(0)
pga.place()
pga.insert()
