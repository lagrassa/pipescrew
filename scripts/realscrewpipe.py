import time
import numpy as np
from frankapy import FrankaArm
from autolab_core import RigidTransform
fa = FrankaArm()
trans =  np.load("data/above_pose_trans.npy")
def a_to_rot(alpha):
    rot = np.array([np.cos(alpha), -np.sin(alpha), 0, -np.sin(alpha), -np.cos(alpha), 0, 0, 0, -1]).reshape((3,3)) 
    return rot

def rotate_hand(fa):
    rt = fa.get_pose()
    trans = rt.translation
    joints = fa.get_joints().tolist()
    for alpha in np.linspace(-2.7,2.7,5):
        joints[-1] = alpha
        rot = a_to_rot(alpha)
        fa.goto_joints(joints)
        print("alpha", alpha)
        time.sleep(0.0002)

def move_joints_to_val(fa, val=-2.5): 
    joints = fa.get_joints().tolist()
    joint_min=val
    joints[-1] = joint_min
    fa.goto_joints(joints)
   

def insert_pipe(z_imped):
    trans =  np.load("data/above_pose_trans.npy")
    rot =  np.load("data/above_pose_rot.npy")
    position = RigidTransform(rotation=rot, translation=trans,from_frame='franka_tool', to_frame='world')
    box_adjustment = 0.02
    position.translation[-1] += box_adjustment
    print(fa.goto_pose_with_cartesian_control(position, cartesian_impedances=[3000, 3000, 3000, 300, 300, 300]))
    #fa.open_gripper()
    #input("Add gripper")
    #fa.close_gripper()
    #trans =  np.load("data/insert_pose_trans.npy")
    move_joints_to_val(fa)
    standard_diff = 0.05
    trans[-1] -= standard_diff
    #rot =  np.load("data/insert_pose_rot.npy")
    #0.089 or below is where it's at
    #
    insert_position= RigidTransform(rotation=fa.get_pose().rotation, translation=trans,from_frame='franka_tool', to_frame='world')
    print(fa.goto_pose_with_cartesian_control(insert_position, cartesian_impedances=[3000, 3000, z_imped, 300, 300, 300]))
    rotate_hand(fa)
    print("actual z", fa.get_pose().translation[-1])
    print("intended z", insert_position.translation[-1])
    input("hold pipe in place to keep it from going on the floor")
    fa.open_gripper()
    move_joints_to_val(fa, val=0)

def grasp_pipe():
    fa.open_gripper()
    above_pipe_adjustment = 0.09
    above_trans= [ 0.38148621, -0.20044326,  0.06727412+above_pipe_adjustment]
    rot =  np.load("data/above_pose_rot.npy")
    position = RigidTransform(rotation=rot, translation=above_trans,from_frame='franka_tool', to_frame='world')
    fa.goto_pose_with_cartesian_control(position, cartesian_impedances=[3000, 3000, 3000, 300, 300, 300])
    position.translation[-1] -= above_pipe_adjustment
    fa.goto_pose_with_cartesian_control(position, cartesian_impedances=[3000, 3000, 3000, 300, 300, 300])
    fa.close_gripper()
    above_trans[-1] += 0.08
    position.translation = above_trans
    fa.goto_pose_with_cartesian_control(position, cartesian_impedances=[3000, 3000, 3000, 300, 300, 300])

if __name__ == "__main__":
    grasp_pipe()
    insert_pipe(70)


