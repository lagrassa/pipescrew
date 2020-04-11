from frankapy import FrankaArm
import GPy
import numpy as np
from pyquaternion import Quaternion
import rospy
import argparse
from env.doorworld import  import DoorWorld
from cv_bridge import CvBridge, CvBridgeError
from autolab_core import RigidTransform, Point
from franka_action_lib.msg import RobotState

# need to check that franka_arm.py is actually sending the gain values to ros in the skill.send_goal
#need to check c++ is receiving them too
try:
    bad_model_states = np.load("data/door_bad_model_states.npy")
    good_model_states = np.load("data/door_bad_model_states.npy")
except FileNotFoundError:
    bad_model_states = []
    good_model_states = []
CONTACT_THRESHOLD=-1.7
gp = None
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

def feelforce():
    ros_data = rospy.wait_for_message("/robot_state_publisher_node_1/robot_state", RobotState)
    force = ros_data.O_F_ext_hat_K
    return force[2]


def in_region_with_modelfailure(pt, retrain=True):
    if retrain:
        bad = np.load("data/door_bad_model_states.npy", allow_pickle=True)
        good = np.load("data/door_good_model_states.npy", allow_pickle=True)
        if len(bad) == 0 and len(good) == 0:
            return False
        X = np.vstack([good, bad])
        Y = np.vstack([np.ones((len(good),1)), np.zeros((len(bad),1))])
        if gp is None:
            gp = GPy.models.GPClassification(X,Y)
            for i in range(6):
                gp.optimize()
    pred = gp.predict(np.hstack([pt.translation, pt.quaternion]).reshape(1,-1))[0]
    return not bool(np.round(pred.item()))

def follow_traj(fa,path, cart_gain = 2000, z_cart_gain = 2000, rot_cart_gain = 300,
                expect_contact=False,
                pb_world=None, monitor_execution=True, traj_type = "joint", dt = 8):
    cart_poses = []
    if monitor_execution:
        for pt in path:
            if traj_type != "cart":
                cart_pt = pb_world.fk_rigidtransform(pt)
            else:
                cart_pt = pt
            if in_region_with_modelfailure(cart_pt):
                #import ipdb; ipdb.set_trace()
                print("expected model failure")
                #return "expected_model_failure"
    for pt, i in zip(path, range(len(path))):
        if traj_type == "cart":
            new_pos = pt
        else:
            new_pos = pb_world.fk_rigidtransform(pt)
        cart_poses.append(new_pos)
        if traj_type == "cart":
            fa.goto_pose_with_cartesian_control(new_pos, cartesian_impedances=[cart_gain, cart_gain, z_cart_gain,rot_cart_gain, rot_cart_gain, rot_cart_gain]) #one of these but with impedance control? compliance comes from the matrix so I think that's good enough
        else:
            #print("Curr joints", np.round(fa.get_joints(),2))
            #print("proposed joints", np.round(pt,2))
            #input("Are these joints ok?")
            joints_list = np.array(pt).tolist()
            if not fa.is_joints_reachable(joints_list):
                print("Joints not reachable")
                import ipdb; ipdb.set_trace()
            else:
                fa.goto_joints(np.array(pt).tolist(), duration=dt)
        force = feelforce()
        model_deviation = False
        cart_to_sigma = lambda cart: np.exp(-0.00026255*cart-4.14340759)
        sigma_cart =  cart_to_sigma(np.array([cart_gain, cart_gain, z_cart_gain]))
        sigma_cart = 0.008
        rot_sigma = cart_to_sigma(rot_cart_gain)
        if monitor_execution:
            try:
                if (np.abs(fa.get_pose().translation-new_pos.translation) >1.96*sigma_cart).any():
                    model_deviation = True
                    print("Farther than expected. Expected "+str(np.round(new_pos.translation,2))+" but got "+
                          str(np.round(fa.get_pose().translation,2)))
                if (Quaternion.absolute_distance(Quaternion(fa.get_pose().quaternion),Quaternion(new_pos.quaternion)) > 1.96*rot_sigma):
                    model_deviation = True
                    print("Farther than expected. Expected "+str(np.round(new_pos.quaternion,2))+" but got "+
                          str(np.round(fa.get_pose().quaternion,2)))
            except ValueError:
                input("fa.get_pose() failed. Continue?")

            if force < CONTACT_THRESHOLD and not expect_contact:
                print("unexpected contact")
                model_deviation = True
            elif force > CONTACT_THRESHOLD and expect_contact:
                print("expected contact")
                model_deviation = True
            if model_deviation and len(cart_poses) >= 2:
                new_bad_state = np.hstack([cart_poses[-2].translation,cart_poses[-2].quaternion])
                if len(bad_model_states) ==0:
                    bad_model_states = new_bad_state
                bad_model_states = np.vstack([bad_model_states, new_bad_state])
            elif not model_deviation and len(cart_poses) >= 2:
                if len(good_model_states) == 0:
                    new_good_state = (np.hstack([cart_poses[-2].translation,cart_poses[-2].quaternion]))
                good_model_states = np.vstack([good_model_states,new_good_state])
            np.save("data/door_bad_model_states.npy", bad_model_states)
            np.save("data/door_good_model_states.npy", good_model_states)
            if model_deviation:
                print("Ending MB policy due to model deviation")
                return("model_failure")
    if monitor_execution:
        return "expected_good"

def go2door(fa, gains=None, close_gripper=False, use_planner = False, world=None):
    """
    Goes to door handle

    Inputs:
        fa: franka arm object
        gains: cartesian_impedances gains to use
    """
    if use_planner:
        traj = world.grasp_object()
        result = follow_traj(traj, monitor_execution=True)
        return result
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

def turn_knob(fa, steps=5, step_duration=5, gains=None, use_planner = False, world=None):
    """
    Turn door handle

    Inputs:
        fa: franka arm object
        steps: number of delta movements to make
        step_duration: time to run each step for in seconds
        gains: cartesian_impedances gains to use
    """
    if use_planner:
        traj = world.grasp_object()
        result = follow_traj(traj, monitor_execution=True)
        return result


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
    dw = DoorWorld()
    print(fa.get_pose())
    print(fa.get_joints())

    input("Press enter to continue")
    go2door(fa, close_gripper=True, use_planner = True, world=dw)
    fa.close_gripper()

    res = turn_knob(fa, use_planner = True, world = dw)
    if res == "expected_model_failure" or res == "model_deviation":
        gains1 = [1000,1000,1000,1,1,1]
        dmp_info2 = "/home/stevenl3/misc_backups/robot_data_door_turn2_position_weights.pkl"
        fa.execute_position_dmp(dmp_info2, duration=10,
            skill_desc='position_dmp', cartesian_impedance=gains1)
    else:
        print("Worked as expected!")


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
