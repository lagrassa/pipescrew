from modelfree import vae, processimgs
import os
import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from skimage import transform
from skimage.transform import rotate, AffineTransform
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy import ndimage


def test_one_image():
    import imageio
    image =imageio.imread("encoder_test_img_Color.png")
    image = rgb2hsv(image)
    image = image[:,:,0]
    image = image.reshape(image.shape+(1,)) #compatible with 3 channels
    n_images = 18
    data = np.zeros((n_images,)+image.shape)
    for i in range(n_images):
        data[i,:,:,:] = image
    data = data.reshape((data.shape[0], data.shape[1]*data.shape[2] *data.shape[3]))
    #my_vae, encoder, decoder, inputs, outputs, output_tensors = vae.make_vae(original_dim=image.shape[0] * image.shape[1])
    my_vae, encoder, decoder, inputs, outputs, output_tensors = vae.make_dsae(image.shape[0], image.shape[1], n_channels = image.shape[-1])
    n_train = int(0.75*data.shape[0])
    vae.train_vae(my_vae, data, n_train, inputs, outputs, output_tensors)
    encoded = encoder(data)

def test_kinect_ims(data_folder, shape_class, augment =True):
    data_list = []
    for fn in os.listdir("data/"+data_folder+"/"):
        if "kinect_data" in fn:
            new_data = np.load("data/"+data_folder+"/"+fn)
            new_data_processed = processimgs.process_raw_camera_data(new_data, shape_class)      
            data_list.append(new_data_processed)
    camera_data = np.concatenate(data_list, axis=0)
    if augment:
        num_rotations = 10
        num_shear = 10
        num_color = 10
        num_random_noise = 10
        new_camera_data = camera_data.copy()
        for im in camera_data:
            for _ in range(num_rotations):
                new_im = rotate(im, angle=np.random.uniform(low=-9,high=9))
                new_camera_data = np.vstack([new_camera_data, new_im.reshape((1,)+new_im.shape)])
            for _ in range(num_shear):
                tf = AffineTransform(shear=np.random.uniform(low=-0.5, high=0.5))
                new_im = transform.warp(im, tf, order=1, preserve_range=True, mode='wrap')
                new_camera_data = np.vstack([new_camera_data, new_im.reshape((1,)+new_im.shape)])
            for _ in range(num_random_noise):
                new_im  = 255*random_noise(im/255., var=0.01**2)
                new_camera_data = np.vstack([new_camera_data, new_im.reshape((1,)+new_im.shape)])
            for _ in range(num_color):
                new_im = im*np.random.uniform(low=0.9, high=1.1)
                new_camera_data = np.vstack([new_camera_data, new_im.reshape((1,)+new_im.shape)])
                new_im = im + np.random.uniform(low=-0.1*255, high=0.1*255)
                new_camera_data = np.vstack([new_camera_data, new_im.reshape((1,)+new_im.shape)])
        camera_data = new_camera_data
     
    image = camera_data[0,:,:]
    camera_data = camera_data.reshape((camera_data.shape[0], camera_data.shape[1]*camera_data.shape[2]))
    idxs = list(range(camera_data.shape[0]))
    np.random.shuffle(idxs)
    camera_data = camera_data[idxs]
    my_vae, encoder, decoder, inputs, outputs, output_tensors = vae.make_dsae(image.shape[0], image.shape[1], n_channels = 1)
    n_train = 0.1
    vae.train_vae(my_vae, camera_data, n_train, inputs, outputs, output_tensors, n_epochs = 20)

    my_vae.save_weights("models/"+data_folder+"/test_weights.h5y")
shape_class = 'Rectangle'
data_folder='rectangle20'
test_kinect_ims(data_folder, shape_class)


