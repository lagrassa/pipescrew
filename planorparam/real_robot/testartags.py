from frankapy import FrankaArm
import GPy
import scipy.stats
mvn = scipy.stats.multivariate_normal
import os
from shapes import Circle, Rectangle, Obstacle
import rospy
from pyquaternion import Quaternion
import cv_bridge
import time
#rospy.init_node("planorparam")
import numpy as np
from franka_interface_msgs.srv import GetCurrentRobotStateCmd
from franka_interface_msgs.msg import RobotState
from autolab_core import RigidTransform
import GPy as gpy
from autolab_core import RigidTransform, YamlConfig
from visualization import Visualizer3D as vis3d

from perception_utils.apriltags import AprilTagDetector
from perception_utils.realsense import get_first_realsense_sensor
from modelfree import processimgs, vae
from modelfree.ILPolicy import ILPolicy, process_action_data
#from test_module.test_behaviour_cloning import test_behaviour_cloning as get_il_policy
from perception import Kinect2SensorFactory, KinectSensorBridged
from sensor_msgs.msg import Image
from perception.camera_intrinsics import CameraIntrinsics
robot = True
if robot:
    from frankapy import FrankaArm
    fa = FrankaArm()
    import ipdb; ipdb.set_trace()
else:
    rospy.init_node("fo")

class Robot():
    def __init__(self, visualize=False, setup_pb=True):
        self.contact_threshold = -3.5 # less than is in contact
        self.autoencoder = None
        self.il_policy=None
        self.bridge = cv_bridge.CvBridge()
        self.shape_to_goal_loc = {}
        self.shape_type_to_ids = {Rectangle:(0,1,2,3)}
        self.shape_goal_type_to_ids = {Rectangle:1}
        self.grasp_offset = 0.024
        self.gp = None
        self.visualize=visualize
        self.setup_perception()
        if setup_pb:
            self.setup_pbworld(visualize=visualize)
    def setup_perception(self):
        self.cfg = YamlConfig("april_tag_pick_place_azure_kinect_cfg.yaml")
        self.T_camera_world = RigidTransform.load(self.cfg['T_k4a_franka_path'])
        self.sensor = Kinect2SensorFactory.sensor('bridged', self.cfg)  # Kinect sensor object
        prefix = ""
        self.sensor.topic_image_color  = prefix+self.sensor.topic_image_color
        self.sensor.topic_image_depth  = prefix+self.sensor.topic_image_depth
        self.sensor.topic_info_camera  = prefix+self.sensor.topic_info_camera
        self.sensor.start()
        self.april = AprilTagDetector(self.cfg['april_tag'])
        # intr = sensor.color_intrinsics #original
        overhead_intr = CameraIntrinsics('k4a', fx=970.4990844726562,cx=1025.4967041015625, fy=970.1990966796875, cy=777.769775390625, height=1536, width=2048) #fx fy cx cy overhead
        frontdown_intr = CameraIntrinsics('k4a',fx=611.9021606445312,cx=637.0317993164062,fy=611.779968261718,cy=369.051239013671, height=1536, width=2048) #fx fy cx cy frontdown
        self.intr = overhead_intr

    """
    @param bool straighten - whether the roll and pitch should be
                forced to 0 as prior information that the object is flat
    @param shape_class Shape \in {Circle, Square, Rectangle, etc} - type of shape the detector should look for
    @param goal whether the detector should look for the object or goal hole
    """
    def detect_ar_world_pos(self,straighten=True, shape_class = Rectangle, goal=False):
        #O, 1, 2, 3 left hand corner. average [0,2] then [1,3]
        T_tag_cameras = []
        detections = self.april.detect(self.sensor, self.intr, vis=1)#self.cfg['vis_detect'])
        import ipdb; ipdb.set_trace()
        detected_ids = []
        for new_detection in detections:
            detected_ids.append(int(new_detection.from_frame.split("/")[1])) #won't work for non-int values
            T_tag_cameras.append(new_detection)
        T_tag_camera = shape_class.tforms_to_pose(detected_ids, T_tag_cameras, goal=goal) #as if there were a tag in the center
        T_tag_camera.to_frame="kinect2_overhead"
        T_tag_world = self.T_camera_world * T_tag_camera
        #T_tag_world.translation[1] -= 0.06
        import ipdb; ipdb.set_trace()
        if straighten:
            T_tag_world  = straighten_transform(T_tag_world)
        print("detected pose", np.round(T_tag_world.translation,3))
        return T_tag_world

    def get_shape_location(self, shape_type):
        return self.detect_ar_world_pos(shape_class=shape_type)

def straighten_transform(rt):
    angles = rt.euler_angles
    roll_off = angles[0]
    pitch_off = angles[1]
    roll_fix = RigidTransform(rotation = RigidTransform.x_axis_rotation(np.pi-roll_off),  from_frame = rt.from_frame, to_frame=rt.from_frame)
    pitch_fix = RigidTransform(rotation = RigidTransform.y_axis_rotation(pitch_off), from_frame = rt.from_frame, to_frame=rt.from_frame)
    new_rt = rt*roll_fix*pitch_fix
    return new_rt



if __name__ == "__main__":
    n_exps = 3
    #rospy.init_node("test", anonymous=True)
    robot = Robot(visualize=True, setup_pb=False)
    print("shape loc", np.round(robot.get_shape_location(Rectangle).translation,2))
    import ipdb; ipdb.set_trace()
    print("goal loc", np.round(robot.get_shape_goal_location(Rectangle).translation,2))
    for i in [18,19,20,21,22]: #18 is where we do DAGGER
        run_insert_exp(robot, i, training=True, use_planner=True)
    
