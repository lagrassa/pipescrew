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
from test_module.test_behaviour_cloning import test_behaviour_cloning as get_il_policy
from perception import Kinect2SensorFactory, KinectSensorBridged
from sensor_msgs.msg import Image
from perception.camera_intrinsics import CameraIntrinsics
from env.pegworld import PegWorld
from utils.conversion_scripts import rigid_transform_to_pb_pose
from frankapy import FrankaArm
fa = FrankaArm()

class Robot():
    def __init__(self, visualize=False, setup_pb=True):
        robot_state_server_name = '/get_current_robot_state_server_node_1/get_current_robot_state_server'
        rospy.wait_for_service(robot_state_server_name)
        self._get_current_robot_state = rospy.ServiceProxy(robot_state_server_name, GetCurrentRobotStateCmd)
        self.contact_threshold = -3.5 # less than is in contact
        self.bad_model_states = np.load("data/bad_model_states.npy")
        self.good_model_states = np.load("data/good_model_states.npy")
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
    def setup_pbworld(self, visualize):
        board_loc = [[0.479,0.0453,0.013],[0.707,0.707,0,0]]
        circle_loc = rigid_transform_to_pb_pose(self.detect_ar_world_pos(straighten=True, shape_class = Circle, goal=False))
        obstacle_loc = rigid_transform_to_pb_pose(self.detect_ar_world_pos(straighten=True, shape_class = Obstacle, goal=False))
        rectangle_loc = rigid_transform_to_pb_pose(self.get_shape_location(Rectangle))
        hole_goal = rigid_transform_to_pb_pose(self.get_shape_goal_location(Rectangle))
        self.pb_world = PegWorld(rectangle_loc=rectangle_loc, hole_goal = hole_goal, circle_loc=circle_loc, board_loc=board_loc, obstacle_loc=obstacle_loc, visualize=visualize)
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
    def modelfree(self, cart_gain =1200, z_cart_gain = 1200, rot_cart_gain = 60, execute=True, length=10):
        for i in range(length): #TODO put in a real termination condition
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
                #self.il_policy = ILPolicy(np.hstack([encoded_camera_data, ee_data]), ee_data, load_fn = "models/ilpolicy.h5y")
                self.il_policy = get_il_policy()#self.il_policy = ILPolicy(np.hstack([encoded_camera_data, ee_data]), ee_data, model_type ="forest", load_fn = "models/rfweights.npy")
            delta_ee_pos = self.il_policy(np.hstack([encoded_camera_data, ee_data]))
            print(np.round(delta_ee_pos, 3), "next pos")
            new_rot = RigidTransform.rotation_from_quaternion(delta_ee_pos[0,3:])
            delta_ee_rt = RigidTransform(translation=delta_ee_pos[0,0:3], rotation=new_rot)
            if not execute:
                return delta_ee_rt
            if execute:
                curr_rt = fa.get_pose()
                scalar = 1
                delta_ee_rt.translation *= scalar
                #print("Delta pose", next_pos.translation - curr_rt.translation)
                #print("Delta angle", np.array(next_pos.euler_angles) - np.array(curr_rt.euler_angles))
                #input("OK to go to pose difference from MF?")
                delta_ee_rt.from_frame = "franka_tool"
                delta_ee_rt.to_frame = "franka_tool"

                fa.goto_pose_delta(delta_ee_rt,cartesian_impedances=[cart_gain, cart_gain, z_cart_gain,rot_cart_gain, rot_cart_gain, rot_cart_gain]) #one of these but with impedance control? compliance comes from the matrix so I think that's good enough
                #res = input("Complete?")
                z_pos = fa.get_pose().translation[-1]
                print("current z_pos", np.round(z_pos,4))
                in_hole = 0.026
                if z_pos <= in_hole:
                    print("Detected successfully in hole")
                    return 
                #if "y" in res:
                #    return

    
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
        print("detected pose", np.round(T_tag_world.translation,3))
        return T_tag_world

    def model_based_grasp_shape(self, T_tag_world, shape_type, monitor_execution=False, grasp_offset=None):
       if grasp_offset is None:
           grasp_offset = self.grasp_offset
       fa.open_gripper()
       path = self.pb_world.grasp_object(visualize=True)
       res =  self.follow_traj(path, monitor_execution=monitor_execution, traj_type="joint", dt=3.5)
       fa.close_gripper()
       return res

        
    """
    Goes to shape center and then grasps it 
    """
    def grasp_shape(self,T_tag_world, shape_type, grasp_offset=0.024, 
                    monitor_execution=True, use_planner=False):
       self.holding_type = shape_type
       if use_planner:
           return self.model_based_grasp_shape(T_tag_world, shape_type, grasp_offset=grasp_offset, monitor_execution=monitor_execution)

       x_offset = 0
       self.grasp_offset = grasp_offset
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
    #    best_correct_yaw = grasp_symmetries[best_symmetry_idx]
    #    good_rotation = RigidTransform.z_axis_rotation((planned_yaw+best_correct_yaw) -start_yaw)
    #    T_tool_world.rotation = start.rotation
       T_tool_world_fixed_rot = RigidTransform(rotation=fa.get_pose().rotation,
            translation=T_tool_world.translation, from_frame="franka_tool",
            to_frame="world")

       fa.open_gripper()
    #    path = self.linear_interp_planner(start, T_tool_world)
       path = self.linear_interp_planner(fa.get_pose(), T_tool_world_fixed_rot)
       self.follow_traj(path, monitor_execution=monitor_execution)
       fa.close_gripper()

    def sample_modelfree_precond_pt(self):
        ee_data = None
        for fn in os.listdir("data/"):
            if "ee_data" in fn:
                new_ee_data = np.load("data/"+fn)[0,:]
                if ee_data is None:
                    ee_data = new_ee_data
                else:
                    ee_data = np.vstack([ee_data, new_ee_data])
        #self.gp_precond = GPy.models.GPClassification(ee_data,np.ones((len(ee_data),1)))
        #self.gp_precond.optimize() #just once
        #sample point from modelfree 1recond
        #return ee_data[np.random.randint(len(ee_data))]#TODO do this using a larger set of samples using the GP, but this is faster
        my_mvn = mvn(mean=np.mean(ee_data,axis=0)+np.array([-0.00,-0.00,0,0,0,0,0]), cov = np.cov(ee_data.T))
        return my_mvn.rvs()


    def goto_modelfree_precond(self, use_planner=False):
        sample = self.sample_modelfree_precond_pt() 
        monitor_execution=True
        #go up slightly to avoid the expected model failure region
        if use_planner:
            sample[2] += 0.02 #keep it out of the board
            sample[2] -= self.grasp_offset #keep it out of the board
            pb_sample = (sample[0:3], sample[3:])
            place_traj = self.pb_world.place_object(hole_goal=pb_sample, shape_class=Rectangle, visualize=True, push_down = False)
            res =  self.follow_traj(place_traj, cart_gain = 2000, z_cart_gain = 2000, rot_cart_gain=250, monitor_execution=monitor_execution, traj_type="joint", dt = 3)

        else:
            up_amount = 0.008
            rotation = RigidTransform.rotation_from_quaternion(sample[3:])
            sample = RigidTransform(translation=sample[0:3], rotation = rotation, from_frame="franka_tool", to_frame="world")
            above_precond_pose = sample.copy()
            above_precond_pose.translation[-1] += up_amount
            up_pose = fa.get_pose().copy()
            up_pose.translation[-1] += up_amount
            self.follow_traj([up_pose], monitor_execution=True)
            path = self.linear_interp_planner(fa.get_pose(), above_precond_pose)
            self.follow_traj(path, monitor_execution=True)
            self.follow_traj([sample], monitor_execution=True)


    def model_based_insert_shape(self, T_tool_world, monitor_execution=True):
        hole_goal = rigid_transform_to_pb_pose(T_tool_world)
        place_traj = self.pb_world.place_object(shape_class=Rectangle, visualize=True)
        res =  self.follow_traj(place_traj, cart_gain = 2000, z_cart_gain = 2000, rot_cart_gain=250, monitor_execution=monitor_execution, traj_type="joint", dt = 3)
        if res == "expected_good":
            down_pose = fa.get_pose()
            down_amount = 0.02
            down_pose.translation[-1] -= down_amount
            return self.follow_traj([down_pose], cart_gain = 500, z_cart_gain = 500, rot_cart_gain=250, monitor_execution=monitor_execution, traj_type="cart", dt = 2)
            
        else:
            return res
    """
    assumes shape is already held 
    """
    def insert_shape(self, use_planner=True):
        T_tag_world = self.get_shape_goal_location( self.holding_type)
        T_tag_tool = RigidTransform(rotation=np.eye(3), translation=[0, 0, 0.0],
                                    from_frame=T_tag_world.from_frame,
                                    to_frame="franka_tool")
        try:
            T_tool_world = T_tag_world * T_tag_tool.inverse()
        except:
            print("Transform didn't work with RT library. Falling back")
            T_tool_world = T_tag_world.copy()
            T_tool_world.from_frame = "franka_tool"
            T_tool_world.to_frame = "world"
        if use_planner:
            return self.model_based_insert_shape(T_tool_world)
        #find corresponding hole and orientation
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
        
        res = self.follow_traj(path, cart_gain = 2200, z_cart_gain = 2200, rot_cart_gain=250)
        if res != "expected_good":
            return res
        T_tool_world.translation[-1] = self.grasp_offset
        modelfree_precond_mean = [ 0.53684655,  0.10351364,  0.02622978, -0.00137794,  0.9999593 ,
                       -0.00405089, -0.00428286]#for debugging
        T_tool_world.translation = modelfree_precond_mean[0:3]
        T_tool_world.rotation = RigidTransform.rotation_from_quaternion(modelfree_precond_mean[3:])
        T_tool_world.translation[0] -= 0.015
        T_tool_world.translation[1] -= 0.015
        res = self.follow_traj([T_tool_world], cart_gain = 1000, z_cart_gain = 1000, rot_cart_gain=300)
        if res != "expected_good":
            return res
        T_tool_world.translation[-1] -= 0.02
        res = self.follow_traj([T_tool_world], cart_gain = 150, z_cart_gain = 700, rot_cart_gain=150)
        return res

    def in_region_with_modelfailure(self,pt, retrain=True):
        if retrain:
            bad = np.load("data/bad_model_states.npy", allow_pickle=True)
            good = np.load("data/good_model_states.npy", allow_pickle=True)
            if len(bad) == 0 and len(good) == 0:
                return False
            X = np.vstack([good, bad])
            Y = np.vstack([np.ones((len(good),1)), np.zeros((len(bad),1))])
            if self.gp is None:
                self.gp = GPy.models.GPClassification(X,Y)
                for i in range(6):
                    self.gp.optimize()
        pred = self.gp.predict(np.hstack([pt.translation, pt.quaternion]).reshape(1,-1))[0]
        return not bool(np.round(pred.item()))

    def follow_traj(self, path, cart_gain = 2000, z_cart_gain = 2000, rot_cart_gain = 300, monitor_execution=True, traj_type = "cart", dt = 2):
        cart_poses = []
        if monitor_execution:
            for pt in path:
                if traj_type != "cart":
                    cart_pt = self.pb_world.fk_rigidtransform(pt)
                else:
                    cart_pt = pt
                if self.in_region_with_modelfailure(cart_pt):
                    #import ipdb; ipdb.set_trace()
                    print("expected model failure")
                    #return "expected_model_failure"
        for pt, i in zip(path, range(len(path))):
            if traj_type == "cart":
                new_pos = pt
            else:
                new_pos = self.pb_world.fk_rigidtransform(pt)
            cart_poses.append(new_pos)
            expect_contact = False
            if new_pos.translation[2]-self.grasp_offset < 0.005:
                expect_contact = True
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
            force = self.feelforce()
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

                if force < self.contact_threshold and not expect_contact:
                    print("unexpected contact")
                    model_deviation = True
                elif force > self.contact_threshold and expect_contact:
                    print("expected contact")
                    model_deviation = True
                if model_deviation and len(cart_poses) >= 2:
                    self.bad_model_states = np.vstack([self.bad_model_states, (np.hstack([cart_poses[-2].translation,cart_poses[-2].quaternion]))])
                elif not model_deviation and len(cart_poses) >= 2:
                    self.good_model_states = np.vstack([self.good_model_states, (np.hstack([cart_poses[-2].translation,cart_poses[-2].quaternion]))])
                np.save("data/bad_model_states.npy", self.bad_model_states)
                np.save("data/good_model_states.npy", self.good_model_states)
                if model_deviation:
                    print("Ending MB policy due to model deviation")
                    return("model_failure")
        if monitor_execution:
            return "expected_good"

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
        print("WASD teleop space for up c for down q and e for spin. k to increase delta, j to decrease o is OK ")
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
            if val == "quit":
                break
            if "a" in val:
                rt.translation[0] -= delta
            if "d" in val:
                rt.translation[0] += delta
            if "s" in val:
                rt.translation[1] -= delta
            if "w" in val:
                rt.translation[1] += delta
            if " " in val:
                rt.translation[-1] += delta
            if "c" in val:
                rt.translation[-1] -= delta
            if val == "o":
                suggested_rt = self.modelfree(execute=False)
                res = input("input OK?")
                if "y" in res:
                    print("adding to set")
                    rt = suggested_rt
                    rt.from_frame = "franka_tool"
                    rt.to_frame = "franka_tool"
                else:
                    continue
            if "q" in val:
                rt_rot = RigidTransform.z_axis_rotation(angle_delta)
                rt.rotation = np.dot(rt.rotation, rt_rot)
            if "e" in val:
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
            imgs.append(img) 
            ee = fa.get_pose()
            ee_rts.append(ee) 
            #rt.translation += np.random.uniform(low = -0.001, high = 0.001, size=rt.translation.shape)
            fa.goto_pose_delta(rt, cartesian_impedances=[1200, 1200, 1200, 60, 60, 60]) #one of these but with impedance control? compliance comes from the matrix so I think that's good enough
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


def run_insert_exp(robot, prefix, training=True, use_planner=False):
    fa.open_gripper()
    fa.reset_joints()
    input("reset scene. Ready?")
    shape_center = robot.get_shape_location(Rectangle)
    robot.setup_pbworld(visualize=False)
    robot.grasp_shape(shape_center, Rectangle, use_planner=use_planner)
    start_time = time.time()
    #result = robot.insert_shape(use_planner=use_planner)
    robot.insert_shape(use_planner=True)
    print("time elapsed", time.time()-start_time)
    return
    result = "expected_model_failure" #does happen, just a workaround for a bug in the code
    if result == "model_failure" or result == "expected_model_failure":
        if training: 
            #robot.kinesthetic_teaching(prefix)
            robot.goto_modelfree_precond(use_planner=True)
            actions, images, ees = robot.keyboard_teleop()
            np.save("data/"+str(prefix)+"actions.npy", actions)
            np.save("data/"+str(prefix)+"kinect_data.npy", images)
            np.save("data/"+str(prefix)+"ee_data.npy", ees)
        else:
            robot.goto_modelfree_precond(use_planner=True)
            robot.modelfree() #avoiding the region where the model is bad
            print("time elapsed", time.time()-start_time)
        
    else:
        print("Success on first try! No learning needed")

    robot.pb_world.close()


if __name__ == "__main__":
    n_exps = 3
    
    fa.open_gripper()
    fa.reset_joints()
    robot = Robot(visualize=True, setup_pb=False)
    #robot.modelfree()
    #import ipdb; ipdb.set_trace()
    #actions, images, ees = robot.keyboard_teleop()
    #np.save("data/"+str(prefix)+"actions.npy", actions)
    #np.save("data/"+str(prefix)+"kinect_data.npy", images)
    #np.save("data/"+str(prefix)+"ee_data.npy", ees)
  

    #input("reset scene. Ready?")
    print("goal loc", np.round(robot.get_shape_goal_location(Rectangle).translation,2))
    for i in [18,19,20,21,22]: #18 is where we do DAGGER
        run_insert_exp(robot, i, training=False, use_planner=True)
    
