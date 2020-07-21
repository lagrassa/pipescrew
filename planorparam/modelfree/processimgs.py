from PIL import Image
import  numpy as np
shape_to_bbox={}
shape_to_bbox["Rectangle"] = [(311,497),(486,694)]
shape_to_bbox["Square"] = [(159,269), (161,357)]
shape_to_bbox["Circle"] = [(172,286), (510,677)]

def process_raw_camera_data(camera_data_raw, shape_class="Rectangle"):
    width = 50
    height = 60
    n_images = camera_data_raw.shape[0]
    camera_data = np.zeros((n_images, height,width))
    for i in range(n_images):
        bbox = shape_to_bbox[shape_class]
        new_im = Image.fromarray(camera_data_raw[i,bbox[0][0]:bbox[0][1],bbox[1][0]:bbox[1][1],0]).resize((width, height))
        new_im = np.array(new_im)
        new_im = new_im.reshape(new_im.shape)
        camera_data[i,:,:] = new_im
        return camera_data
