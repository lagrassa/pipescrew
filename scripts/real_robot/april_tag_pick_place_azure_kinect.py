import os
import logging
import argparse
from time import sleep

import numpy as np
import sys
print(sys.path)

from autolab_core import RigidTransform, YamlConfig
from visualization import Visualizer3D as vis3d

from perception_utils.apriltags import AprilTagDetector
from perception_utils.realsense import get_first_realsense_sensor

from perception import Kinect2SensorFactory, KinectSensorBridged
from perception.camera_intrinsics import CameraIntrinsics

from frankapy import FrankaArm

def subsample(data, rate=0.1):
    idx = np.random.choice(np.arange(len(data)), size=int(rate * len(data)))
    return data[idx]


def make_det_one(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ np.eye(len(R)) @ Vt


def get_closest_grasp_pose(T_tag_world, T_ee_world):
    tag_axes = [
        T_tag_world.rotation[:,0], -T_tag_world.rotation[:,0],
        T_tag_world.rotation[:,1], -T_tag_world.rotation[:,1]
    ]
    x_axis_ee = T_ee_world.rotation[:,0]
    dots = [axis @ x_axis_ee for axis in tag_axes]
    grasp_x_axis = tag_axes[np.argmax(dots)]
    grasp_z_axis = np.array([0, 0, -1])
    grasp_y_axis = np.cross(grasp_z_axis, grasp_x_axis)
    grasp_R = make_det_one(np.c_[grasp_x_axis, grasp_y_axis, grasp_z_axis])
    #TEST#####
    #grasp_R=np.array([[9.99e-01,-1.5234e-04,3.277999e-04],[-1.52390514e-04, -9.99990352e-01,  1.41480959e-04],[ 3.27775229e-04, -1.41529542e-04, -9.99999936e-01]])
    #########
    grasp_translation = T_tag_world.translation + np.array([0, 0, -cfg['cube_size'] / 2])
    return RigidTransform(
        rotation=grasp_R,
        translation=grasp_translation,
        from_frame=T_ee_world.from_frame, to_frame=T_ee_world.to_frame
    )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='/home/stevenl3/Documents/planorparam/scripts/real_robot/april_tag_pick_place_azure_kinect_cfg.yaml')
    parser.add_argument('--no_grasp', '-ng', action='store_true')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)
    T_camera_world = RigidTransform.load(cfg['T_k4a_franka_path'])

    logging.info('Starting robot')
    fa = FrankaArm()    
#    fa.reset_joints()
#    fa.reset_pose()
#    fa.open_gripper()
    
    T_ready_world = fa.get_pose()
    T_ready_world.translation[0] += 0.25
    T_ready_world.translation[2] = 0.4    

    #ret = fa.goto_pose(T_ready_world)

    logging.info('Init camera')
    
    #sensor = get_first_realsense_sensor(cfg['rs']) #original
    sensor=Kinect2SensorFactory.sensor('bridged',cfg) #Kinect sensor object
    
    sensor.start()
    logging.info('Detecting April Tags')
    april = AprilTagDetector(cfg['april_tag'])
    
    #intr = sensor.color_intrinsics #original
    intr=CameraIntrinsics('k4a',972.31787109375,971.8189086914062,
                          1022.3043212890625,777.7421875,height=1536,width=2048)

    T_tag_camera = april.detect(sensor, intr, vis=cfg['vis_detect'])[0]
    T_tag_world = T_camera_world * T_tag_camera
    logging.info('Tag has translation {}'.format(T_tag_world.translation))
    import ipdb; ipdb.set_trace()

    T_tag_tool = RigidTransform(rotation = np.eye(3),translation = [0,0,0.04], from_frame=T_tag_world.from_frame, to_frame="franka_tool") 
    T_tool_world =  T_tag_world *T_tag_tool.inverse()
    ret = fa.goto_pose(T_tool_world)

    logging.info('Finding closest orthogonal grasp')
    T_grasp_world = get_closest_grasp_pose(T_tag_world, T_ready_world)
    T_lift = RigidTransform(translation=[0, 0, 0.2], from_frame=T_ready_world.to_frame, to_frame=T_ready_world.to_frame)
    T_lift_world = T_lift * T_grasp_world

    logging.info('Visualizing poses')
    _, depth_im, _ = sensor.frames()
    points_world = T_camera_world * intr.deproject(depth_im)

    if cfg['vis_detect']:
        vis3d.figure()
        vis3d.pose(RigidTransform())
        vis3d.points(subsample(points_world.data.T, 0.01), color=(0,1,0), scale=0.002)
        vis3d.pose(T_ready_world, length=0.05)
        vis3d.pose(T_camera_world, length=0.1)
        vis3d.pose(T_tag_world)
        vis3d.pose(T_grasp_world)
        vis3d.pose(T_lift_world)
        vis3d.show()

    #const_rotation=np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    #test = RigidTransform(rotation=const_rotation,translation=T_tag_world.translation, from_frame='franka_tool', to_frame='world')
    #import pdb; pdb.set_trace()
    
    rotation=T_tag_world.rotation
    rotation[0:2,:]=-1*rotation[0:2,:]  

   
#    test = RigidTransform(rotation=rotation,translation=T_tag_world.translation, from_frame='franka_tool', to_frame='world')
#    fa.goto_pose_with_cartesian_control(test)
#    fa.close_gripper()

    # if not args.no_grasp:
    #     logging.info('Commanding robot')
    #     fa.goto_pose_with_cartesian_control(T_lift_world)
    #     import ipdb; pdb.set_trace()

    #     fa.goto_pose_with_cartesian_control(T_grasp_world)
    #     fa.close_gripper()
    #     fa.goto_pose_with_cartesian_control(T_lift_world)
    #     sleep(3)
    #     fa.goto_pose_with_cartesian_control(T_grasp_world)
    #     fa.open_gripper()
    #     fa.goto_pose_with_cartesian_control(T_lift_world)
    #     fa.goto_pose_with_cartesian_control(T_ready_world)

    #import IPython; IPython.embed(); exit(0)
