from PIL import Image
import  numpy as np

def process_raw_camera_data(camera_data_raw):
    width = 50
    height = 60
    camera_data = np.zeros((camera_data_raw.shape[0], height,width, 1))
    for i in range(camera_data_raw.shape[0]):
        new_im = Image.fromarray(camera_data_raw[i,275:440,520:700,0]).resize((width, height))
        #import ipdb; ipdb.set_trace()
        #new_im = rgb2hsv(new_im[:,:,0:3])
        #new_im = rgb2hsv(new_im)
        #new_im = new_im[:,:,0]
        new_im = np.array(new_im)
        new_im = new_im.reshape(new_im.shape+(1,))
        camera_data[i,:,:,:] = new_im
    return camera_data
