from modelfree import vae
import numpy as np
import imageio
image =imageio.imread("encoder_test_img_Color.png")/255.
n_images = 5
data = np.zeros((n_images,)+image.shape)
for i in range(n_images):
    data[i,:,:,:] = image
vae, encoder, decoder, inputs, outputs, output_tensors = vae.make_dsae(image.shape[0], image.shape[1])
n_train = 10
vae.train_vae(vae, data, n_train, inputs, outputs, output_tensors)
encoded = encoder(data)
