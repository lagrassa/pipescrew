from modelfree import vae
import numpy as np
from skimage.color import rgb2hsv
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
my_vae, encoder, decoder, inputs, outputs, output_tensors = vae.make_vae(original_dim=image.shape[0] * image.shape[1])
#my_vae, encoder, decoder, inputs, outputs, output_tensors = vae.make_dsae(image.shape[0], image.shape[1], n_channels = image.shape[-1])
n_train = 10
vae.train_vae(my_vae, data, n_train, inputs, outputs, output_tensors)
encoded = encoder(data)
