import numpy as np
import modelfree.processimgs as processimgs
from PIL import Image
import os
from skimage.color import rgb2hsv
from modelfree.ILPolicy import ILPolicy, process_action_data
from utils.transformations import *
from modelfree import vae

def get_deltas(later_ees, first_ees):
    trans = later_ees[:,0:3]-first_ees[:,0:3]
    #quaterions are what the rotation would need to be to go from first_ees to later_ees
    quats = []
    for i in range(later_ees.shape[0]):
        start_matrix = quaternion_matrix(first_ees[i,3:])
        end_matrix = quaternion_matrix(later_ees[i,3:])
        matrix_between = np.dot(end_matrix, np.linalg.inv(start_matrix))
        quat_between = quaternion_from_matrix(matrix_between)
        quats.append(quat_between)
    return np.hstack([trans, np.vstack(quats)])


def test_behaviour_cloning(data_folder,shape_class, regularize=True):
    camera_data_list = []
    ee_data_list = [] 
    action_data_list = []
    #camera_data_raw = np.load("data/0kinect_data.npy")
    #ee_data = np.load("data/0ee_data.npy")
    #action_data = np.load("data/0actions.npy")
    #n_waypoints = 10
    for fn in os.listdir("data/"+data_folder+"/"):
        if "kinect_data" in fn:
            new_raw_camera_data = np.load("data/"+data_folder+"/"+fn)
            #mask = np.round(np.linspace(0,new_raw_camera_data.shape[0]-1,n_waypoints)).astype(np.int32)
            #new_raw_camera_data = new_raw_camera_data[mask]
            new_camera_data_processed = processimgs.process_raw_camera_data(new_raw_camera_data, shape_class)
            camera_data_list.append(new_camera_data_processed)
        elif "ee_data" in fn:
            new_ee_data = np.load("data/"+data_folder+"/"+fn)
            ee_data_list.append(new_ee_data)
        elif "actions" in fn:
            new_action_data = np.load("data/"+data_folder+"/"+fn)
            action_data_list.append(new_action_data)
    camera_data = np.concatenate(camera_data_list,axis=0)
    ee_data = np.concatenate(ee_data_list, axis=0)
    action_data = np.concatenate(action_data_list, axis=0)
    if regularize:
        camera_data += np.random.uniform(low=-10, high=10, size=camera_data.shape)
        ee_data += np.random.uniform(low=-0.005, high=0.005, size=ee_data.shape)
        action_data += np.random.uniform(low=-0.005, high=0.005, size=action_data.shape)
    image = camera_data[0,:,:]
    camera_data = camera_data.reshape((camera_data.shape[0], camera_data.shape[1]*camera_data.shape[2]))
    my_vae, encoder, decoder, inputs, outputs, output_tensors = vae.make_dsae(image.shape[0], image.shape[1], n_channels = 1)
    my_vae.load_weights("models/"+data_folder+"/test_weights.h5y")
    encoded_camera_data = encoder.predict(camera_data)[0]
    #ee_data = process_action_data(ee_data)
    ee_deltas = action_data #get_deltas(ee_data[1:],ee_data[:-1]) #delta from last
    input_data = np.hstack([encoded_camera_data, ee_data])
    il_policy = ILPolicy(input_data, ee_deltas, model_type="forest")
    il_policy.train_model(input_data, ee_deltas, n_epochs = 1000, validation_split=0.1 )
    thresh = 0.02
    errors = []
    angle_errors = []
    for i in range(len(ee_data)-1):
        input_data = np.hstack([encoded_camera_data[i,:], ee_data[i,:]])
        proposed_delta = il_policy(input_data)
        actual_delta = ee_deltas[i,:]
        error = np.linalg.norm(proposed_delta[0,0:3]-actual_delta[0:3])
        angle_error = np.linalg.norm(proposed_delta[0,3:]- actual_delta[3:]) 
        angle_errors.append(angle_error)
        errors.append(error)
    print("Mean error", np.mean(errors))
    print("std error", np.std(errors))
    print("max error", np.max(errors))

    print("Mean error", np.mean(angle_errors))
    print("std error", np.std(angle_errors))
    print("max error", np.max(angle_errors))
    assert(np.max(errors) < 0.1)
    assert(np.mean(errors) < 0.015)
    il_policy.save_model("models/ilpolicy.h5y")
    return il_policy

if __name__ == "__main__":
    shape_class="Rectangle"
    data_folder = "rectangle20"
    test_behaviour_cloning(data_folder, shape_class)
