import numpy as np
import modelfree.processimgs as processimgs
from PIL import Image

from skimage.color import rgb2hsv
from modelfree.ILPolicy import ILPolicy
from modelfree import vae

def test_behaviour_cloning():
    camera_data_raw = np.load("data/0kinect_data.npy")
    joint_data = np.load("data/0ee_data.npy")
    for i in range(1,4):
        camera_data_raw = np.vstack([camera_data_raw, np.load("data/"+str(i)+"kinect_data.npy")])
        joint_data = np.vstack([joint_data, np.load("data/"+str(i)+"ee_data.npy")])
    camera_data = processimgs.process_raw_camera_data(camera_data_raw)
    image = camera_data[0,:,:,:]
    camera_data = camera_data.reshape((camera_data.shape[0], camera_data.shape[1]*camera_data.shape[2] *camera_data.shape[3]))
    my_vae, encoder, decoder, inputs, outputs, output_tensors = vae.make_dsae(image.shape[0], image.shape[1], n_channels = image.shape[2])
    my_vae.load_weights("test_weights.h5y")
    encoded_camera_data = encoder.predict(camera_data)[0]
    il_policy = ILPolicy(encoded_camera_data, joint_data)
    il_policy.train_model(encoded_camera_data, joint_data, n_epochs = 5000, validation_split=0.1 )
    thresh = 0.02
    errors = []
    for i in range(len(joint_data)):
        import ipdb; ipdb.set_trace()
        proposed_joints = il_policy(encoded_camera_data[i,:])
        actual_joints = joint_data[i]
        error = np.linalg.norm(proposed_joints-actual_joints)
        errors.append(error)
    print("Mean error", np.mean(errors))
    print("std error", np.std(errors))
    print("max error", np.max(errors))
    assert(np.max(errors) < 0.02)

    il_policy.save_model("models/ilpolicy.h5y")

test_behaviour_cloning()
