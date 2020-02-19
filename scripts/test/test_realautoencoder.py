from modelfree import vae, processimgs
import numpy as np
from PIL import Image
from skimage.color import rgb2hsv

def test_one_image():
    import imageio
    image =imageio.imread("encoder_test_img_Color.png")
    image = rgb2hsv(image)
    image = image[:,:,0]
    image = image.reshape(image.shape+(1,)) #compatible with 3 channels
    n_images = 8
    data = np.zeros((n_images,)+image.shape)
    for i in range(n_images):
        data[i,:,:,:] = image
    data = data.reshape((data.shape[0], data.shape[1]*data.shape[2] *data.shape[3]))
    #my_vae, encoder, decoder, inputs, outputs, output_tensors = vae.make_vae(original_dim=image.shape[0] * image.shape[1])
    my_vae, encoder, decoder, inputs, outputs, output_tensors = vae.make_dsae(image.shape[0], image.shape[1], n_channels = image.shape[-1])
    n_train = int(0.75*data.shape[0])
    vae.train_vae(my_vae, data, n_train, inputs, outputs, output_tensors)
    encoded = encoder(data)

def test_kinect_ims():
    camera_data_raw = np.load("data/0kinect_data.npy")
    for i in range(1,5):
        camera_data_raw = np.vstack([camera_data_raw, np.load("data/"+str(i)+"kinect_data.npy")])

    #width = 40
    #height = 50
    #camera_data = np.zeros((camera_data_raw.shape[0], height,width, 1))
    #for i in range(camera_data_raw.shape[0]):
    #    new_im = Image.fromarray(camera_data_raw[i,275:440,520:700,0]).resize((width, height))
    #    #import ipdb; ipdb.set_trace()
    #    #new_im = rgb2hsv(new_im[:,:,0:3])
    #    #new_im = rgb2hsv(new_im)
    #    #new_im = new_im[:,:,0]
    #    new_im = np.array(new_im)
    #    new_im = new_im.reshape(new_im.shape+(1,))
    #    camera_data[i,:,:,:] = new_im
    camera_data = processimgs.process_raw_camera_data(camera_data_raw)
    image = camera_data[0,:,:,:]
    camera_data = camera_data.reshape((camera_data.shape[0], camera_data.shape[1]*camera_data.shape[2] *camera_data.shape[3]))
    my_vae, encoder, decoder, inputs, outputs, output_tensors = vae.make_dsae(image.shape[0], image.shape[1], n_channels = image.shape[-1])
    n_train = 0.05
    vae.train_vae(my_vae, camera_data, n_train, inputs, outputs, output_tensors, n_epochs = 300)

    my_vae.save_weights("test_weights.h5y")
test_kinect_ims()


