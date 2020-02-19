import numpy as np
from PIL import Image

from skimage.color import rgb2hsv
from modelfree.ILPolicy import ILPolicy
from modelfree import vae

def test_behaviour_cloning():
    camera_data_raw = np.load("data/0kinect_data.npy")
    joint_data = np.load("data/0ee_data.npy")
    width = 80
    height = 50
    camera_data = np.zeros((camera_data_raw.shape[0], width, height,1))
    for i in range(camera_data_raw.shape[0]):
        new_im = Image.fromarray(camera_data_raw[i,:,:,0:3]).resize((height, width))
        new_im = rgb2hsv(new_im)
        new_im = new_im[:,:,0]
        new_im = new_im.reshape(new_im.shape+(1,))
        camera_data[i,:,:,:] = new_im
        image = camera_data[0,:,:,:]
    camera_data = camera_data.reshape((camera_data.shape[0], camera_data.shape[1]*camera_data.shape[2] *camera_data.shape[3]))
    my_vae, encoder, decoder, inputs, outputs, output_tensors = vae.make_dsae(image.shape[0], image.shape[1], n_channels = image.shape[2])
    my_vae.load_weights("test_weights.h5y")
    encoded_camera_data = encoder.predict(camera_data)[0]
    il_policy = ILPolicy(encoded_camera_data, joint_data)
    il_policy.train_model(encoded_camera_data, joint_data, n_epochs = 200 )
    thresh = 0.05
    for i in range(len(joint_data)-1):
        proposed_joints = il_policy(encoded_camera_data[i-1,:])
        actual_joints = joint_data[i]
        if (np.linalg.norm(proposed_joints-actual_joints)< thresh):
            print("Found big error of", np.linalg.norm(proposed_joints-actual_joints))
            assert(False)

test_behaviour_cloning()