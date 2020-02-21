from frankapy import FrankaArm
import rospy
import cv_bridge
import time
#rospy.init_node("planorparam")
import numpy as np
from franka_action_lib.srv import GetCurrentRobotStateCmd
from franka_action_lib.msg import RobotState
from autolab_core import RigidTransform
import GPy as gpy
from autolab_core import RigidTransform, YamlConfig
from visualization import Visualizer3D as vis3d

from perception_utils.apriltags import AprilTagDetector
from perception_utils.realsense import get_first_realsense_sensor
from modelfree import processimgs, vae
from modelfree.ILPolicy import ILPolicy, process_action_data
from perception import Kinect2SensorFactory, KinectSensorBridged
from sensor_msgs.msg import Image
from perception.camera_intrinsics import CameraIntrinsics

from frankapy import FrankaArm
fa = FrankaArm()

class Circle:
    pass

class Square:
    pass
class Rectangle:
    def __init__(self):
        pass
    @staticmethod
    def symmetries():
        return [-np.pi, 0, np.pi]
    """
    returns the intervals of rotation that are identical, at least enough to be useful. 
    The triangle can't be rotated but the rectangle can be inserted at any interval of 3.14
    """
    @staticmethod
    def tforms_to_pose(ids, tforms,goal=False):
        if goal:
            rectangles_ids = [4,5,6,7]
        else:
            rectangles_ids = [0,1,2,3]
        relevant_ids = [ar_id for ar_id in ids if ar_id in rectangles_ids]
        relevant_tforms = []
        for id_num in relevant_ids:
            tform_idx = ids.index(id_num)
            relevant_tforms.append(tforms[tform_idx])
        
        avg_rotation = relevant_tforms[0]
        for i in range(1, len(relevant_tforms)):
            avg_rotation = avg_rotation.interpolate_with(relevant_tforms[i], 0.5)

        if len(relevant_ids) == 4:
            translation = np.mean(np.vstack([T.translation for T in relevant_tforms]), axis=0)
        elif 0 in relevant_ids and 2 in relevant_ids:
            translation = np.mean(np.vstack([T.translation for T in relevant_tforms]), axis=0)
        elif 1 in relevant_ids and 3 in relevant_ids:
            translation = np.mean(np.vstack([T.translation for T in relevant_tforms]), axis=0)
        else:
            print(relevant_ids)
            print("Not enough detections to make accurate pose estimate")

        return RigidTransform(rotation = avg_rotation.rotation, translation = translation, to_frame = tforms[0].to_frame, from_frame = "peg_center")

class Robot():
    def __init__(self):
        robot_state_server_name = '/get_current_robot_state_server_node_1/get_current_robot_state_server'
        rospy.wait_for_service(robot_state_server_name)
        self._get_current_robot_state = rospy.ServiceProxy(robot_state_server_name, GetCurrentRobotStateCmd)
        self.contact_threshold = -3.5 # less than is in contact
        self.bad_model_states = []
        self.good_model_states = []
        self.autoencoder = None
        self.il_policy=None
        self.bridge = cv_bridge.CvBridge()
        self.shape_to_goal_loc = {}
        self.shape_type_to_ids = {Rectangle:(0,1,2,3)}
        self.shape_goal_type_to_ids = {Rectangle:1}
        self.setup_perception()

    def setup_perception(self):
        self.cfg = YamlConfig("april_tag_pick_place_azure_kinect_cfg.yaml")
        self.T_camera_world = RigidTransform.load(self.cfg['T_k4a_franka_path'])
        self.sensor = Kinect2SensorFactory.sensor('bridged', self.cfg)  # Kinect sensor object
        prefix = "/overhead"
        self.sensor.topic_image_color  = prefix+self.sensor.topic_image_color
        self.sensor.topic_image_depth  = prefix+self.sensor.topic_image_depth
        self.sensor.topic_info_camera  = prefix+self.sensor.topic_info_camera
        self.sensor.start()
        self.april = AprilTagDetector(self.cfg['april_tag'])
        # intr = sensor.color_intrinsics #original
        self.intr = CameraIntrinsics('k4a', 970.4990844726562,970.1990966796875, 1025.4967041015625, 777.769775390625, height=1536, width=2048)
        
    """
    executes the model-free policy
    """
    def modelfree(self, cart_gain =500, z_cart_gain = 500, rot_cart_gain = 100):
        for i in range(50): #TODO put in a real termination condition
            data = rospy.wait_for_message("/frontdown/rgb/image_raw", Image)
            img = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            camera_data = processimgs.process_raw_camera_data(img.reshape((1,)+img.shape))
            image = camera_data[0,:,:,:]
            camera_data = camera_data.reshape((camera_data.shape[0], camera_data.shape[1]*camera_data.shape[2] *camera_data.shape[3]))
            if self.autoencoder is None:
                my_vae, encoder, decoder, inputs, outputs, output_tensors = vae.make_dsae(image.shape[0], image.shape[1], n_channels = image.shape[2])
                my_vae.load_weights("test_weights.h5y")
                self.autoencoder = encoder

            encoded_camera_data = self.autoencoder.predict(camera_data)[0]
            ee_data = np.array(rospy.wait_for_message("/robot_state_publisher_node_1/robot_state", RobotState).O_T_EE)
            ee_data = ee_data.reshape((1,)+ee_data.shape)
            ee_data = process_action_data(ee_data)
            if self.il_policy is None:
                self.il_policy = ILPolicy(np.hstack([encoded_camera_data, ee_data]), ee_data, load_fn = "models/ilpolicy.h5y")
            delta_ee_pos = self.il_policy(np.hstack([encoded_camera_data, ee_data]))
            import ipdb; ipdb.set_trace()
            new_rot = RigidTransform.rotation_from_quaternion(delta_ee_pos[0,3:])
            delta_ee_rt = RigidTransform(translation=delta_ee_pos[0,0:3], rotation=new_rot)
            curr_rt = fa.get_pose()
            #print("Delta pose", next_pos.translation - curr_rt.translation)
            #print("Delta angle", np.array(next_pos.euler_angles) - np.array(curr_rt.euler_angles))
            #input("OK to go to pose difference from MF?")
            delta_ee_rt.from_frame = "franka_tool"
            delta_ee_rt.to_frame = "franka_tool"
            fa.goto_pose_delta(delta_ee_rt, cartesian_impedances=[cart_gain, cart_gain, z_cart_gain,rot_cart_gain, rot_cart_gain, rot_cart_gain]) #one of these but with impedance control? compliance comes from the matrix so I think that's good enough

    def detect_ar_world_pos(self,straighten=True, shape_class = Rectangle, goal=False):
        #O, 1, 2, 3 left hand corner. average [0,2] then [1,3]
        T_tag_cameras = []
        detections = self.april.detect(self.sensor, self.intr, vis=self.cfg['vis_detect'])

        detected_ids = []
        for new_detection in detections:
            detected_ids.append(int(new_detection.from_frame.split("/")[1])) #won't work for non-int values
            T_tag_cameras.append(new_detection)
        T_tag_camera = shape_class.tforms_to_pose(detected_ids, T_tag_cameras, goal=goal) #as if there were a tag in the center
        T_tag_world = self.T_camera_world * T_tag_camera
        if straighten:
            T_tag_world  = straighten_transform(T_tag_world)
        print("detected pose", np.round(T_tag_world.translation,2))

        return T_tag_world
    """
    Goes to shape center and then grasps it 
    """
    def grasp_shape(self,T_tag_world, shape_type):
       self.holding_type = shape_type
       x_offset = 0
       self.grasp_offset = 0.025
       start = fa.get_pose()
       T_tag_tool = RigidTransform(rotation=np.eye(3), translation=[x_offset, 0, self.grasp_offset], from_frame="peg_center",
                                   to_frame="franka_tool")
       T_tool_world = T_tag_world * T_tag_tool.inverse()
       self.original_object_rotation = T_tool_world.copy()  
       #find the closeset symmetry.  Probably should wrap in a function
       start_yaw = start.euler_angles[-1] #yaw we're at right now
       planned_yaw = T_tool_world.euler_angles[-1] #yaw we need to go to 
       grasp_symmetries = [-np.pi,-np.pi/2, 0, np.pi/2, np.pi]
       best_symmetry_idx = np.argmin([np.linalg.norm((planned_yaw+sym)-start_yaw ) for sym in grasp_symmetries]) #wraparound :c
       best_correct_yaw = grasp_symmetries[best_symmetry_idx]
       good_rotation = RigidTransform.z_axis_rotation((planned_yaw+best_correct_yaw) -start_yaw)
       T_tool_world.rotation = np.dot(good_rotation, start.rotation)
       #T_tool_world.rotation = start.rotation
       
       fa.open_gripper()
       path = self.linear_interp_planner(start, T_tool_world)
       self.follow_traj(path)
       fa.close_gripper()

    """
    assumes shape is already held 
    """
    def insert_shape(self):
        #find corresponding hole and orientation
        T_tag_world = self.get_shape_goal_location( self.holding_type)
        T_tag_tool = RigidTransform(rotation=np.eye(3), translation=[0, 0, 0.0],
                                    from_frame=T_tag_world.from_frame,
                                    to_frame="franka_tool")
        T_tool_world = T_tag_world * T_tag_tool.inverse()
        start = fa.get_pose()
        #set rotation to closest rotation in symmetry to pose
        start_yaw = self.original_object_rotation.euler_angles[-1] #yaw we're at right now
        #start_yaw = start.euler_angles[-1] #yaw we're at right now
        planned_yaw = T_tool_world.euler_angles[-1] #yaw we need to go to 
        symmetries = self.holding_type.symmetries()
        best_symmetry_idx = np.argmin([np.linalg.norm((planned_yaw+sym)-start_yaw ) for sym in symmetries]) #wraparound :c
        best_correct_yaw = symmetries[best_symmetry_idx]
        good_rotation = RigidTransform.z_axis_rotation((planned_yaw+best_correct_yaw) -start_yaw)
        above_surface = 0.01
        #go up first to avoid friction
        curr_pos = fa.get_pose()
        curr_pos.translation[-1] += above_surface
        up_pose = curr_pos 
        up_path = [up_pose]
        self.follow_traj(up_path, cart_gain = 2200, z_cart_gain = 2200, rot_cart_gain=250)
        T_tool_world.rotation = np.dot(start.rotation,good_rotation)
        above_T_tool_world = T_tool_world.copy()
        above_T_tool_world.translation[-1] = fa.get_pose().translation[-1] #keep it at the same z
        #now over where it needs to be and at the right place
        path = self.linear_interp_planner(fa.get_pose(), above_T_tool_world)
        self.follow_traj(path, cart_gain = 2200, z_cart_gain = 2200, rot_cart_gain=250)
        T_tool_world.translation[-1] = self.grasp_offset
        self.follow_traj([T_tool_world], cart_gain = 600, z_cart_gain = 600, rot_cart_gain=250)
        

    def follow_traj(self, path, cart_gain = 2500, z_cart_gain = 2500, rot_cart_gain = 300):
        for pt, i in zip(path, range(len(path))):
            new_pos = pt
            expect_contact = False
            if new_pos.translation[2] < 0.01:
                expect_contact = True

            fa.goto_pose_with_cartesian_control(new_pos, cartesian_impedances=[cart_gain, cart_gain, z_cart_gain,rot_cart_gain, rot_cart_gain, rot_cart_gain]) #one of these but with impedance control? compliance comes from the matrix so I think that's good enough
            #fa.goto_pose(new_pos) #one of these but with impedance control? compliance comes from the matrix so I think that's good enough
            #consider breaking this one up to make it smoother
            force = self.feelforce()
            model_deviation = False
            pose_thresh = 0.02
            if np.linalg.norm(fa.get_pose().translation-new_pos.translation) > pose_thresh:
                model_deviation = True
                print("Farther than expected. Expected "+str(np.round(new_pos.translation,2))+" but got "+
                      str(np.round(fa.get_pose().translation,2)))
            if force < self.contact_threshold and not expect_contact:
                print("unexpected contact")
                model_deviation = True
            elif force > self.contact_threshold and expect_contact:
                print("expected contact")
                model_deviation = True
            if model_deviation:
                self.bad_model_states.append(path[i-1])
            else:
                self.good_model_states.append(path[i-1])
            np.save("data/bad_model_states.npy", self.bad_model_states)
            np.save("data/good_model_states.npy", self.good_model_states)
            if model_deviation:
                print("Ending MB policy due to model deviation")
                return
    def train_model(self):
        self.high_state = [0.4,0.4,0.05]
        self.low_state = [-0.4, -0.4, 0.00]
        lengthscale = (self.high_state-self.low_state) * 0.05
        k = gpy.kern.Matern52(self.high_state, ARD=True, lengthscale=lengthscale)
        states = np.vstack([self.bad_model_states, self.good_model_states])
        labels = np.vstack([-np.ones(len(self.bad_model_states)), np.ones(len(self.good_model_states))])
        self.model = gpy.models.GPRegression(states, labels, k)
        self.model['.*variance'].constrain_bounded(1e-1,2., warning=False)
        self.model['Gaussian_noise.variance'].constrain_bounded(1e-4,0.01, warning=False)
        # These GP hyper parameters need to be calibrated for good uncertainty predictions.
        self.model.optimize(messages=False)


    def get_shape_goal_location(self, shape_type):
        if shape_type not in self.shape_to_goal_loc.keys():
            goal_loc = self.detect_ar_world_pos(shape_class = Rectangle, goal=True, straighten=True)
            self.shape_to_goal_loc[shape_type] = goal_loc
        return self.shape_to_goal_loc[shape_type]

    def get_shape_location(self, shape_type):
        return self.detect_ar_world_pos(shape_class=shape_type)
    """
    moves arm back to pose where it's not in the way of the camera
    """
    def reset_arm(self):
        fa.reset_joints()

    def feelforce(self):
        ros_data = self._get_current_robot_state().robot_state
        force = ros_data.O_F_ext_hat_K
        return force[2]
    def kinesthetic_teaching(self, prefix="test"):
        time = 4
        do_intel = False
        do_kinect = True
        self.ee_infos = []
        def callback_ee(data):
            ee_state = data.O_T_EE
            self.ee_infos.append(ee_state)
        if do_intel:
            intel_data = None
            self.intel_camera_images = []
            def callback_intel(data):
                image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
                self.intel_camera_images.append(image)
            self.intel_subscriber = rospy.Subscriber("/camera/color/image_raw", Image, callback_intel)
        if do_kinect:
            kinect_data = None
            self.kinect_camera_images = []
            def callback_kinect(data):
                image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
                self.kinect_camera_images.append(image)
                ee_data= rospy.wait_for_message("/robot_state_publisher_node_1/robot_state", RobotState)
                callback_ee(ee_data) #helps in syncing
            self.kinect_subscriber = rospy.Subscriber("/frontdown/rgb/image_raw", Image, callback_kinect)

        input("Ready for kinesthetic teaching?")
        fa.apply_effector_forces_torques(time, 0, 0, 0)
        np.save("data/"+str(prefix)+"ee_data.npy", self.ee_infos)
        if do_intel:
            np.save("data/"+str(prefix)+"intel_data.npy", self.intel_camera_images)

            self.intel_subscriber.unregister()
        if do_kinect:
            np.save("data/"+str(prefix)+"kinect_data.npy", self.kinect_camera_images)
            self.kinect_subscriber.unregister()

    """
    Linear interpolation of poses, including quaternion
    """
    def linear_interp_planner(self, start, goal, n_pts = 2):
        return start.linear_trajectory_to(goal, n_pts)
    def keyboard_teleop(self):
        print("WASD teleop space for up c for down q and e for spin. k to increase delta, j to decrease ")
        delta = 0.01
        angle_delta = 0.03
        delta_rts = []
        imgs = []
        ee_rts = []
        while True:
            rt = RigidTransform(from_frame="franka_tool", to_frame="franka_tool")
            val = input()
            data = rospy.wait_for_message("/frontdown/rgb/image_raw", Image)
            img = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            if val == "a":
                rt.translation[0] -= delta
            elif val == "d":
                rt.translation[0] += delta
            elif val == "s":
                rt.translation[1] -= delta
            elif val == "w":
                rt.translation[1] += delta
            elif val == " ":
                rt.translation[-1] += delta
            elif val == "c":
                rt.translation[-1] -= delta
            elif val == "q":
                rt_rot = RigidTransform.z_axis_rotation(angle_delta)
                rt.rotation = np.dot(rt.rotation, rt_rot)
            elif val == "e":
                rt_rot = RigidTransform.z_axis_rotation(-angle_delta)
                rt.rotation = np.dot(rt.rotation, rt_rot)
            elif val == 'k':
                delta *= 2
                angle_delta *= 2
                continue
            elif val == "j":
                delta /= 2
                angle_delta /=2
                continue
            elif val == "quit":
                break
            imgs.append(img) 
            ee = fa.get_pose()
            ee_rts.append(ee) 
            fa.goto_pose_delta(rt, cartesian_impedances=[500, 500, 500, 100, 100, 100]) #one of these but with impedance control? compliance comes from the matrix so I think that's good enough
            delta_rts.append(rt)
            transes = []
            ee_transes = []
            quats = []
            ee_quats = []
            for rt, ee_rt in zip(delta_rts, ee_rts):
                transes.append(rt.translation)
                ee_transes.append(ee_rt.translation)
                quats.append(rt.quaternion)
                ee_quats.append(ee_rt.quaternion)
            
            #transform to the right way
            #fa.goto_pose(rt)
        img_result = np.zeros((len(imgs),)+imgs[0].shape)
        for i in range(len(imgs)):
            img_result[i,:,:] = imgs[i]
        return np.hstack([np.vstack(transes), np.vstack(quats)]), img_result, np.hstack([np.vstack(ee_transes), np.vstack(ee_quats)])

def straighten_transform(rt):
    angles = rt.euler_angles
    roll_off = angles[0]
    pitch_off = angles[1]
    roll_fix = RigidTransform(rotation = RigidTransform.x_axis_rotation(np.pi-roll_off),  from_frame = rt.from_frame, to_frame=rt.from_frame)
    pitch_fix = RigidTransform(rotation = RigidTransform.y_axis_rotation(pitch_off), from_frame = rt.from_frame, to_frame=rt.from_frame)
    new_rt = rt*roll_fix*pitch_fix
    return new_rt


def run_insert_exp(robot, prefix, training=True):
    fa.open_gripper()
    fa.reset_joints()
    input("reset scene. Ready?")
    shape_center = robot.get_shape_location(Rectangle)
    robot.grasp_shape(shape_center, Rectangle)
    robot.insert_shape()
    if training: 
        #robot.kinesthetic_teaching(prefix)
        actions, images, ees = robot.keyboard_teleop()
        np.save("data/"+str(prefix)+"actions.npy", actions)
        np.save("data/"+str(prefix)+"kinect_data.npy", images)
        np.save("data/"+str(prefix)+"ee_data.npy", ees)
    else:
        robot.modelfree()


if __name__ == "__main__":
    n_exps = 3
    robot = Robot()
    #res = robot.keyboard_teleop()
    robot.modelfree()
    import ipdb; ipdb.set_trace()
    fa.open_gripper()
    fa.reset_joints()

    input("reset scene. Ready?")
    print("goal loc", np.round(robot.get_shape_goal_location(Rectangle).translation,2))
    for i in [0,1]:
        run_insert_exp(robot, i, training=True)
    
