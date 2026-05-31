import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import log10
from skimage.metrics import structural_similarity as find_ssim

IMAGE_SIZE = 32
IMAGE_ADDRESS = "..\\datasets\\test_images.npy"
MODEL_ADDRESS = "models\\cs_net_1024"
BATCH_SIZE = 64
STRIDE = 32

# Plots an image
def plot_image(img):
    plt.imshow(img)
    plt.show()
# Calculates and returns PSNR value between two images.
def find_psnr(im1, im2):
    return 10*log10(255**2/np.square(im1 - im2).mean())

def concatenation_block(pseudoblocks):
    pseudoimage = pseudoblocks[:,0,0]
    for column in range(pseudoblocks.shape[2]-1):
        pseudoimage = tf.keras.layers.Concatenate(axis=2)([pseudoimage, pseudoblocks[:,0,column+1]])
    for row in range(pseudoblocks.shape[1]-1):
        next_row = pseudoblocks[:,row+1,0]
        for column in range(pseudoblocks.shape[2]-1):
            next_row = tf.keras.layers.Concatenate(axis=2)([next_row, pseudoblocks[:,row+1,column+1]])
        pseudoimage = tf.keras.layers.Concatenate(axis=1)([pseudoimage, next_row])
    return tf.expand_dims(pseudoimage, axis=3)

# Fetching and preprocessing images for test.
images = np.load(IMAGE_ADDRESS)
images = np.expand_dims(images, axis=3)

# Controlling the integrity of the data
print("Nummber of the test images: ", images.shape[0])
print("Size of the test images: ", images[0].shape)

# Loading the model and creating the 
model = tf.keras.models.load_model(MODEL_ADDRESS)

# Standalone measurement module
input_layer = tf.keras.Input(shape=(IMAGE_SIZE,IMAGE_SIZE,1))
output_layer = model.get_layer(model.layers[1].name)
x = output_layer(input_layer)

measurement_model = tf.keras.Model(inputs = input_layer, outputs = x)

# Standalone reconstruction module
input_layer = tf.keras.Input(shape=(1, 1, 1024))
x = model.get_layer(model.layers[2].name)(input_layer)
x = model.get_layer(model.layers[3].name)(x)
concatenate = concatenation_block(x)
x1 = model.get_layer(model.layers[6].name)(concatenate)
x2 = model.get_layer(model.layers[7].name)(x1)
x1 = model.get_layer(model.layers[8].name)([x1,x2])
x2 = model.get_layer(model.layers[9].name)(x1)
x1 = model.get_layer(model.layers[10].name)([x1,x2])
x2 = model.get_layer(model.layers[11].name)(x1)
x1 = model.get_layer(model.layers[12].name)([x1,x2])
x2 = model.get_layer(model.layers[13].name)(x1)
x1 = model.get_layer(model.layers[14].name)([x1,x2])
x2 = model.get_layer(model.layers[15].name)(x1)
x1 = model.get_layer(model.layers[16].name)([x1,x2])
output_layer = model.get_layer(model.layers[17].name)(x1)

recon_model = tf.keras.Model(inputs = input_layer, outputs = output_layer)

# Converting the images into measurements
measurements = measurement_model.predict(images)
mes = list(range(2,33))
mes = [x * 32 for x in mes]

for i in mes:
    mes_crop = np.copy(measurements)
    mes_crop[:,:,:,i:] = np.zeros((10000,1,1,1024-i))
    print("Measurement size: ", mes_crop[:,:,:,:i].shape)
    reconstructions = recon_model.predict(mes_crop)
    image_psnr = []
    image_ssim = []
    for i in range(images.shape[0]):
        image_psnr.append(find_psnr(reconstructions[i], images[i]))
        image_ssim.append(find_ssim(np.squeeze(reconstructions[i], axis=2), np.squeeze(images[i], axis=2), 
                        data_range=images.max() - images.min()))
    avg_psnr = sum(image_psnr)/images.shape[0]
    avg_ssim = sum(image_ssim)/images.shape[0]
    print("Average PSNR for measurement size = %d: %.4f" % (i, avg_psnr))
    print("Average SSIM for measurement size = %d: %.4f" % (i, avg_ssim))
