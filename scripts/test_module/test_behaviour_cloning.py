import numpy as np
import modelfree.processimgs as processimgs
from PIL import Image

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


def test_behaviour_cloning():
    camera_data_raw = np.load("data/0kinect_data.npy")
    ee_data = np.load("data/0ee_data.npy")
    action_data = np.load("data/0actions.npy")
    #n_waypoints = 10
    for i in range(1,18):
        try:
            new_raw_camera_data = np.load("data/"+str(i)+"kinect_data.npy")
        except FileNotFoundError:
            continue
        #mask = np.round(np.linspace(0,new_raw_camera_data.shape[0]-1,n_waypoints)).astype(np.int32)
        #new_raw_camera_data = new_raw_camera_data[mask]
        camera_data_raw = np.vstack([camera_data_raw, new_raw_camera_data])
        new_ee_data = np.load("data/"+str(i)+"ee_data.npy")
        new_action_data = np.load("data/"+str(i)+"actions.npy")
        #new_ee_data = new_ee_data[mask]
        ee_data = np.vstack([ee_data, new_ee_data])
        action_data = np.vstack([action_data, new_action_data])

    camera_data = processimgs.process_raw_camera_data(camera_data_raw)
    image = camera_data[0,:,:,:]
    camera_data = camera_data.reshape((camera_data.shape[0], camera_data.shape[1]*camera_data.shape[2] *camera_data.shape[3]))
    my_vae, encoder, decoder, inputs, outputs, output_tensors = vae.make_dsae(image.shape[0], image.shape[1], n_channels = image.shape[2])
    my_vae.load_weights("test_weights.h5y")
    encoded_camera_data = encoder.predict(camera_data)[0]
    #ee_data = process_action_data(ee_data)
    ee_deltas = action_data #get_deltas(ee_data[1:],ee_data[:-1]) #delta from last
    input_data = np.hstack([encoded_camera_data, ee_data])
    il_policy = ILPolicy(input_data, ee_deltas, model_type="forest")
    il_policy.train_model(input_data, ee_deltas, n_epochs = 6000, validation_split=0.05, params_file = "models/rfparams.npy")
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
    test_behaviour_cloning()
