from frankapy import FrankaArm
import numpy as np
import rospy
import argparse
from cv_bridge import CvBridge, CvBridgeError
from autolab_core import RigidTransform, Point

# need to check that franka_arm.py is actually sending the gain values to ros in the skill.send_goal
#need to check c++ is receiving them too

def go2start(fa, joint_gains=None, cart_gains=None):
    """
    Resets joints to home and goes to starting position near door

    Inputs:
        fa: franka arm object
        joint_gains (list): list of 7 joint impedances to use for
            inital movements 
        cart_gains (list): list of 6 cartesian impedances to use for
            rest of movements to door, [x,y,z,roll,pitch,yaw]       

    Notes:
        Start position:
        start joints: [ 2.28650215 -1.36063288 -1.4431576  -1.93011263
                        0.23962597  2.6992652  0.82820212]
        start pose:
            Tra: [0.60516914 0.33243171 0.45531707]
            Rot: [[ 0.99917643 -0.02115786  0.03434495]
                  [-0.03350342  0.03890612  0.99868102]
                  [-0.02246619 -0.99900921  0.03816595]]
    """
    if cart_gains is None:
        cart_gains = [1000,1000,1000,50,50,50]
    if joint_gains is None:
        joint_gains = [500,500,500,500,500,500,500]

    reset_from_door(fa)

    #pose1
    joints1 = [1.25932568, -0.90343739, -1.24127123, -2.00878656,
               0.1644398, 1.77009426, 0.7167884]
    fa.goto_joints(joints1, duration=5, joint_impedances=joint_gains)

    #pose2
    joints2 = [1.77866878, -1.45379609, -1.45567126, -1.91807283,
               0.11080599, 2.21234057, 0.91823287]
    fa.goto_joints(joints2, duration=5, joint_impedances=joint_gains)

    #go to start pose
    print("Going to start position...")
    desired_tool_pose = RigidTransform(
        rotation=np.array([[0.99917643, -0.02115786, 0.03434495],
                           [-0.03350342, 0.03890612, 0.99868102],
                           [-0.02246619, -0.99900921, 0.03816595]]),
        translation=np.array([0.60516914, 0.33243171, 0.45531707]),
        from_frame='franka_tool', to_frame='world')
    fa.goto_pose_with_cartesian_control(desired_tool_pose,
                                        duration=10.0,
                                        cartesian_impedances=cart_gains)
    joints3 = [2.28650215, -1.36063288, -1.4431576, -1.93011263,
               0.23962597, 2.6992652, 0.82820212]
    fa.goto_joints(joints3, duration=5, joint_impedances=joint_gains)

def go2door(fa, gains=None, close_gripper=False):
    """
    Goes to door handle

    Inputs:
        fa: franka arm object
        gains: cartesian_impedances gains to use
    """
    if gains is None:
        gains = [1000,1000,1000,50,50,50]
    
    #go to grasp location
    desired_tool_pose = RigidTransform(
        rotation=np.array([[0.9988874, -0.0389031, -0.0262919],
                           [0.02844157, 0.05578267, 0.99803772],
                           [-0.03736013, -0.99767509, 0.05682816]]),
        translation=np.array([0.61257949, 0.46935375, 0.46155012]),
        from_frame='franka_tool', to_frame='world')
    fa.goto_pose_with_cartesian_control(desired_tool_pose,
                                        duration=5.0,
                                        cartesian_impedances=gains)

    if close_gripper == True:
        fa.close_gripper()

def reset_from_door(fa):
    """
    Longer reset with low gains
    """
    # Reset robot pose and joints
    print('Opening Grippers')
    fa.open_gripper()
    print('Reset with pose')
    fa.reset_pose(duration=10, cartesian_impedances=[500,500,500,10,10,10])
    print('Reset with joints')
    fa.reset_joints()

def go2start_from_door(fa):
    """
    Go directly to the start location, assuming you are already close to the pose
    Might behave badly if you try from a pose that isn't close
    """
    cart_gains = [1000,1000,1000,50,50,50]
    joint_gains = [500,500,500,500,500,500,500]

    input("Please make sure robot is near the desired start position\n Press enter to continue")

    print("Going to start position...")
    desired_tool_pose = RigidTransform(
        rotation=np.array([[0.99917643, -0.02115786, 0.03434495],
                           [-0.03350342, 0.03890612, 0.99868102],
                           [-0.02246619, -0.99900921, 0.03816595]]),
        translation=np.array([0.60516914, 0.33243171, 0.45531707]),
        from_frame='franka_tool', to_frame='world')
    fa.goto_pose_with_cartesian_control(desired_tool_pose,
                                        duration=10.0,
                                        cartesian_impedances=cart_gains)

    joints2 = [2.28650215, -1.36063288, -1.4431576, -1.93011263,
               0.23962597, 2.6992652, 0.82820212]
    fa.goto_joints(joints2, duration=5, joint_impedances=joint_gains)
    print("Done")

def turn_knob(fa, steps=5, step_duration=5, gains=None):
    """
    Turn door handle

    Inputs:
        fa: franka arm object
        steps: number of delta movements to make
        step_duration: time to run each step for in seconds
        gains: cartesian_impedances gains to use
    """
    if gains is None:
        gains = [1000,1000,1000,1,1,1]

    desired_tool_pose = RigidTransform(
        rotation=np.eye(3),
        translation=np.array([0.01, 0, -0.01]),
        from_frame='franka_tool', to_frame='franka_tool')
    
    for _ in range(steps): 
        fa.goto_pose_delta_with_cartesian_control(desired_tool_pose,
                                                duration=step_duration,
                                                cartesian_impedances=gains)
    
def pull_door(fa, duration=5.0, gains=None, open_gripper=True):
    """
    Pull on door
    
    Inputs:
        fa: the franka arm object
        duration: duration of movement
        gains: the cartesian impedance gains to use 
    """
    if gains is None:
        gains = [400, 800, 1000, 5, 5, 5]

    desired_tool_pose = RigidTransform(
        rotation=np.array([[0.43484823, 0.8440584, 0.31377554,],
                            [-0.062703, -0.31921447, 0.94560479,],
                            [0.89830735, -0.43086923, -0.08588651]]),
        translation=np.array([0.52567046, 0.28332678, 0.37527456]),
        from_frame='franka_tool', to_frame='world')
    fa.goto_pose_with_cartesian_control(desired_tool_pose,
                                        duration=duration,
                                        cartesian_impedances=gains)

    if open_gripper:
        fa.open_gripper()

if __name__ == '__main__':
    # Examples and for collecting robot poses
    # Everything is hard coded :/
    
    print('Starting robot')
    fa = FrankaArm()
    go2start(fa)

    print(fa.get_pose())
    print(fa.get_joints())

    input("Press enter to continue")
    go2door(fa, close_gripper=True)
    fa.close_gripper()

    turn_knob(fa)

    input("Press enter to continue")
    pull_door(fa)

    go2start_from_door(fa)

    reset_from_door(fa)

# # These are really small and don't work well with the low gains
# #1 degree rotation
#     desired_tool_pose = RigidTransform(
#         rotation=np.array([[0.9998477, 0.0000000, 0.0174524],
#                            [0.0000000, 1.0000000, 0.0000000],
#                            [-0.0174524, 0.0000000, 0.9998477]]),
#         translation=np.array([0.001253871, 0, 0.001253871]),
#         from_frame='franka_tool', to_frame='franka_tool')
#     fa.goto_pose_delta_with_cartesian_control(desired_tool_pose,
#                                               duration=5.0,
#                                               cartesian_impedances=gains)

# #2.5degrees
#     desired_tool_pose = RigidTransform(
#         rotation=np.array([[0.9990482, 0.0000000, 0.0436194],
#                            [0.0000000, 1.0000000, 0.0000000],
#                            [-0.0436194, 0.0000000, 0.9990482]]),
#         translation=np.array([0.00313436, 0, 0.00313436]),
#         from_frame='franka_tool', to_frame='franka_tool')
#     fa.goto_pose_delta_with_cartesian_control(desired_tool_pose,
#                                               duration=5.0,
#                                               cartesian_impedances=gains)

# #5degrees
#     desired_tool_pose = RigidTransform(
#         rotation=np.array([[ 0.9961947, 0.0000000, 0.0871557],
#                            [ 0.0000000, 1.0000000, 0.0000000],
#                            [-0.0871557, 0.0000000, 0.9961947]]),
#         translation=np.array([0.00626745, 0, 0.00626745]),
#         from_frame='franka_tool', to_frame='franka_tool')
#     fa.goto_pose_delta_with_cartesian_control(desired_tool_pose,
#                                               duration=5.0,
#                                               cartesian_impedances=gains)
