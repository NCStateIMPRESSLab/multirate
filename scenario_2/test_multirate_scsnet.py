# Importing necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import log10
from skimage.metrics import structural_similarity as find_ssim

IMAGE_SIZE = 32
IMAGE_ADDRESS = "..\\datasets\\test_images.npy"
MODEL_ADDRESS = "models\\multirate_csnet_32"
BATCH_SIZE = 32
BLOCK_SIZE = 32

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

model = tf.keras.models.load_model(MODEL_ADDRESS)
layer_names = model.layers
for index, layer in enumerate(layer_names):
    layer_names[index] = layer.name
    
# Standalone measurement module
input_layer = tf.keras.Input(shape=(IMAGE_SIZE,IMAGE_SIZE,1))
output_layer = model.get_layer(layer_names[1])
x = output_layer(input_layer)
measurement_model = tf.keras.Model(inputs = input_layer, outputs = x)

# Standalone reconstruction module
input_layer = tf.keras.Input(tensor=output_layer.output)
recon_model = tf.keras.Model(inputs = input_layer, outputs = model.outputs)

# Converting the images into measurements
measurements = measurement_model.predict(images)
mes = list(range(1,int(IMAGE_SIZE**2/BLOCK_SIZE)+1))
mes = [x * BLOCK_SIZE for x in mes]

avg_psnr = []
avg_ssim = []
for i in mes: 
    mes_crop = np.copy(measurements)
    mes_crop[:,:,:,i:] = np.zeros((10000,1,1,1024-i))
    print("Measurement size: ", mes_crop[:,:,:,:i].shape)
    reconstructions = recon_model.predict(mes_crop)
    reconstructions_high = reconstructions[31]

    image_psnr = []
    image_ssim = []
    for j in range(images.shape[0]):
        image_psnr.append(find_psnr(reconstructions_high[j], images[j]))
        image_ssim.append(find_ssim(np.squeeze(reconstructions_high[j], axis=2), np.squeeze(images[j], axis=2), 
                        data_range=images.max() - images.min()))
    avg_psnr.append(sum(image_psnr)/images.shape[0])
    avg_ssim.append(sum(image_ssim)/images.shape[0])
    #print("Average PSNR for measurement size = %d: %.4f" % (i, avg_psnr[int(i/BLOCK_SIZE)-1]))
    #print("Average SSIM for measurement size = %d: %.4f" % (i, avg_ssim[int(i/BLOCK_SIZE)-1]))

# Test code for evaluating the reconstruction and visualization.
"""
reconstructions = recon_model.predict(measurements)
reconstructions_high = reconstructions[0]

image_psnr = []
image_ssim = []
for i in range(images.shape[0]):
    image_psnr.append(find_psnr(reconstructions_high[i], images[i]))
    image_ssim.append(find_ssim(np.squeeze(reconstructions_high[i], axis=2), np.squeeze(images[i], axis=2), 
                    data_range=images.max() - images.min()))
avg_psnr = sum(image_psnr)/images.shape[0]
avg_ssim = sum(image_ssim)/images.shape[0]
print("Average PSNR: ", avg_psnr)
print("Average SSIM: ", avg_ssim)

index = 0

plt.figure(figsize=(4, 4), dpi=80)
plt.axis('off')
plt.imshow(images[index], cmap='gray')
plt.savefig("imreal.png", format="png", bbox_inches="tight")
plt.show()
print("Original image")

plt.figure(figsize=(4, 4), dpi=80)
plt.axis('off')
plt.imshow(reconstructions_high[index], cmap='gray')
plt.savefig("impred.png", format="png", bbox_inches="tight")
plt.show()
print("Image %i PSNR: %.4f" % (index+1, image_psnr[index]))
print("Image %i SSIM: %.4f" % (index+1, image_ssim[index]))
"""