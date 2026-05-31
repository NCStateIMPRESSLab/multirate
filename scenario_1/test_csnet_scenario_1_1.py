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

# Fetching and preprocessing images for test.
images = np.load(IMAGE_ADDRESS)
images = np.expand_dims(images, axis=3)

# Controlling the integrity of the data
print("Nummber of the test images: ", images.shape[0])
print("Size of the test images: ", images[0].shape)

model = tf.keras.models.load_model(MODEL_ADDRESS)
predictions = model.predict(images)

image_psnr = []
image_ssim = []
for i in range(images.shape[0]):
    image_psnr.append(find_psnr(predictions[i], images[i]))
    image_ssim.append(find_ssim(np.squeeze(predictions[i], axis=2), np.squeeze(images[i], axis=2), 
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
plt.imshow(predictions[index], cmap='gray')
plt.savefig("impred.png", format="png", bbox_inches="tight")
plt.show()
print("Image %i PSNR: %.4f" % (index+1, image_psnr[index]))
print("Image %i SSIM: %.4f" % (index+1, image_ssim[index]))