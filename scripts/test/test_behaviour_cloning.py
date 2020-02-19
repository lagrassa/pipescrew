import numpy as np
import modelfree.processimgs as processimgs
from PIL import Image

from skimage.color import rgb2hsv
from modelfree.ILPolicy import ILPolicy, process_action_data
from modelfree import vae

def test_behaviour_cloning():
    camera_data_raw = np.load("data/0kinect_data.npy")
    ee_data = np.load("data/0ee_data.npy")
    for i in range(1,4):
        camera_data_raw = np.vstack([camera_data_raw, np.load("data/"+str(i)+"kinect_data.npy")])
        ee_data = np.vstack([ee_data, np.load("data/"+str(i)+"ee_data.npy")])
    camera_data = processimgs.process_raw_camera_data(camera_data_raw)
    image = camera_data[0,:,:,:]
    camera_data = camera_data.reshape((camera_data.shape[0], camera_data.shape[1]*camera_data.shape[2] *camera_data.shape[3]))
    my_vae, encoder, decoder, inputs, outputs, output_tensors = vae.make_dsae(image.shape[0], image.shape[1], n_channels = image.shape[2])
    my_vae.load_weights("test_weights.h5y")
    encoded_camera_data = encoder.predict(camera_data)[0]
    il_policy = ILPolicy(encoded_camera_data, ee_data)
    il_policy.train_model(encoded_camera_data, ee_data, n_epochs = 6000, validation_split=0.05 )
    thresh = 0.02
    errors = []
    ee_data = process_action_data(ee_data)
    for i in range(len(ee_data)-1):
        proposed_joints = il_policy(encoded_camera_data[i-1,:])
        actual_joints = ee_data[i]
        error = np.linalg.norm(proposed_joints[0,0:3]-actual_joints[0:3])
        errors.append(error)
    print("Mean error", np.mean(errors))
    print("std error", np.std(errors))
    print("max error", np.max(errors))
    assert(np.max(errors) < 0.1)
    assert(np.mean(errors) < 0.01)
    il_policy.save_model("models/ilpolicy.h5y")

test_behaviour_cloning()
