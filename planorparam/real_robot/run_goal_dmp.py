import sys
import numpy as np
import argparse
import pickle
import csv
import matplotlib.pyplot as plt
import rospy

from cv_bridge import CvBridge, CvBridgeError
from autolab_core import RigidTransform, Point
from frankapy import FrankaArm

import realpeginsert
import shapes

if __name__ == '__main__':
    """
    Run the dmp for the peg picking task

    Inputs:
        -- pose_dmp_weights_path (str): the filepath of the dmp weights file
        -- time (float): duration to run dmp for in seconds
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_dmp_weights_path', '-w', type=str, default='')
    parser.add_argument('--time', '-t', type=float, default=10)
    args = parser.parse_args()

    gains = [2000, 2000, 2000, 1500, 1500, 1000, 1000] #joint impedances

    # load the dmp weights
    if not args.pose_dmp_weights_path:
        args.pose_dmp_weights_path = './data/dmp/IROS2020_dmp_data_goal_position_weights.pkl'
    pose_dmp_file = open(args.pose_dmp_weights_path,"rb")
    pose_dmp_info = pickle.load(pose_dmp_file)

    print('Starting robot')
    fa = FrankaArm()

    # Reset robot pose and joints
    print('Opening Grippers')
    fa.goto_gripper(0.08, grasp=False)
    print('Reset with pose')
    fa.reset_pose(duration=10)
    print('Reset with joints')
    fa.reset_joints()

    # instantiate object for AR tag detection
    frank = realpeginsert.Robot()

    # get transforms
    tag2worldTF = frank.detect_ar_world_pos(straighten=True,
                                             shape_class=shapes.Circle,
                                             goal=False)
    print('Object Transform: (x,y) = {}'.format(tag2worldTF))
    goal2worldTF = frank.detect_ar_world_pos(straighten=True,
                                             shape_class=shapes.Circle,
                                             goal=True)

    input("Press Enter to continue...")
    # grasp object
    frank.grasp_shape(tag2worldTF, realpeginsert.Circle,
                      monitor_execution=False, use_planner=True,
                      training=False)

    # fa.run_guide_mode(15)
    # fa.goto_gripper(0.025, grasp=True)

    goal = goal2worldTF.translation
    goal[2] += 0.03 # offset the height
    goal[0] += 0.01
    
    print('Goal Loc = {}'.format(goal))
    current_pose = fa.get_pose().translation
    print('Current Loc = {}'.format(current_pose))
    goal_pose = (goal - current_pose).tolist()
    print('Sensor Value = {}'.format(goal_pose))
    
    # scaling factor for z trajectory
    goal_pose.append(-0.5)
    print(goal_pose)


    input("Press Enter to continue...")
    print('Running the dmp')
    # fa.run_guide_mode(args.time)
    fa.execute_pose_dmp(pose_dmp_info,
                        duration=args.time,
                        use_goal_formulation=False,
                        initial_sensor_values=goal_pose,
                        position_only=True,
                        joint_impedances=gains
    )

    fa.goto_gripper(0.08, grasp=False)
