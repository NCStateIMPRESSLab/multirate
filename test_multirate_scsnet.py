# %% LPIPS metric !!!RUN THIS ONE BEFORE IMPORTING OTHER LIBRARIES ON SPYDER!!!
import torch; 
import lpips;
#loss_fn_alex = lpips.LPIPS(net='alex');    # best forward scores
loss_fn_vgg = lpips.LPIPS(net="vgg");       # closer to "traditional" perceptual loss, when used for optimization

# %% Importing necessary libraries
print("Importing necessary libraries");
import os;
import cv2;
import numpy as np; 
import tensorflow as tf; 
from math import log10, ceil;
from skimage.metrics import structural_similarity as find_ssim; 

# %% Defining functions
print("Defining functions"); 
# Calculates and returns PSNR value between two images.
def find_psnr(im1, im2):
    return 10*log10(255**2/np.square(im1 - im2).mean()); 


# Import a test image as blocks
def import_image(img: np.array, block_size: int, noise=False, snr=0):
    blocks = [];
    height, width = img.shape[:2];
    row = ((height - block_size) // block_size) + 1;
    col = ((width - block_size) // block_size) + 1;
    
    for i in range(row):
        for j in range(col):
            blocks.append(img[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]);
            if noise:
                sig_pow = np.sum(np.abs(blocks[-1]), axis=None) / (3*block_size**2);
                noise = 10**(snr/10)*sig_pow * np.random.rand(block_size, block_size, 3);
                blocks[-1] = blocks[-1] + noise;
    
    return np.asarray(blocks);


# Merges image blocks to create a full image.
def merge_blocks(blocks, height, width):
    construction = np.array([]);
    for i in range(height):
        const_width = blocks[width*i];
        for j in range(width-1):
            const_width = np.concatenate((const_width, blocks[width*i+j+1]), 1);
        if i == 0:
            construction = const_width;
        else:
            construction = np.concatenate((construction, const_width), 0);
    return construction;

# %% Defining constants
print("Defining constants");
NUM_LAYER = 16;
IMAGE_SIZE = 32;
NUM_CHANNEL = 3;
SAMPLE_RATE = 1;
nB = IMAGE_SIZE**2*SAMPLE_RATE;
BLOCK_SIZE = ceil(nB / NUM_LAYER);

IS_NOISE = False;
if IS_NOISE:
    SNR = 0;
    noise_pow = 1 / 10**(SNR/10);

MEASUREMENT_RATE = 0.8;
MEASUREMENT_NUM = int(NUM_CHANNEL*IMAGE_SIZE**2 * MEASUREMENT_RATE);

TEST_DATA = "bsds"; 

# Addresses
TEST_IMAGE_ADDRESS = ""; 
if TEST_DATA == "bsds": TEST_IMAGE_ADDRESS += "../dataset/test_images_bsds500"; 
elif TEST_DATA == "kodak": TEST_IMAGE_ADDRESS += "../dataset/test_images_kodak"; 
elif TEST_DATA == "mcmx": TEST_IMAGE_ADDRESS += "../dataset/DataSet/McM18"; 
MODEL_ADDRESS = "./models/scsnet_"+str(BLOCK_SIZE)+"/model.keras"; 

# %% Importing the trained model
print("Importing the trained model"); 
model = tf.keras.models.load_model(MODEL_ADDRESS); 

layer_names = model.layers; 
for index, layer in enumerate(layer_names):
    layer_names[index] = layer.name; 
    
# Standalone measurement module
input_layer = tf.keras.Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3)); 
output_layer = model.get_layer(layer_names[1]); 
x = output_layer(input_layer); 
measurement_model = tf.keras.Model(inputs = input_layer, outputs = x); 

# Standalone reconstruction module
input_layer = tf.keras.Input(tensor=output_layer.output); 
reconstruction_model = tf.keras.Model(inputs = input_layer, outputs = model.outputs); 

# %% Running the model for test
print("Running the model for test"); 
image_blocks = []; 
original_images = []; predicted_images = []; 
image_psnr = []; image_ssim = []; image_lpips = []; 

image_num = 0;
for root, dirs, files in os.walk(TEST_IMAGE_ADDRESS, topdown=True):
    for name in files:
       addr = os.path.join(TEST_IMAGE_ADDRESS, name);
       new_image = cv2.imread(os.path.join(TEST_IMAGE_ADDRESS, name));
       
       height, width = new_image.shape[:2];
       height = ((height - IMAGE_SIZE) // IMAGE_SIZE) + 1;
       width = ((width - IMAGE_SIZE) // IMAGE_SIZE) + 1;
       image_blocks.append(import_image(new_image, IMAGE_SIZE));
       
       print("Sampling the image.");
       measurements = measurement_model.predict(image_blocks[-1], verbose=1);
       if IS_NOISE:
           signal_pow = np.sum(np.abs(measurements)) / measurements.size;
           measurements = measurements + noise_pow * signal_pow * np.random.normal(size=measurements.shape);
       measurements = np.concatenate([measurements[:,:,:,:MEASUREMENT_NUM], np.zeros(shape=(measurements.shape[0],1,1,NUM_CHANNEL*nB - MEASUREMENT_NUM))], axis=3);
       
       print("Reconstructing the image from the samples.");
       #predictions = model.predict(image_blocks[-1], verbose=1)[1];
       predictions = reconstruction_model.predict(measurements, verbose=1)[-1];

       predictions = np.where(predictions > 255, 255, predictions);
       predictions = np.where(predictions < 0, 0, predictions);
       
       original_images.append(merge_blocks(image_blocks[-1], height, width));
       predicted_images.append(merge_blocks(predictions.astype(int), height, width));
       
       o_lpips = torch.Tensor(np.transpose(original_images[-1],(2,0,1)));
       p_lpips = torch.Tensor(np.transpose(predicted_images[-1],(2,0,1)));
       
       image_lpips.append(loss_fn_vgg(o_lpips, p_lpips)[0][0]);
       image_ssim.append(find_ssim(original_images[-1], predicted_images[-1], channel_axis=-1, data_range=255, multichannel=True));
       image_psnr.append(find_psnr(original_images[-1], predicted_images[-1]));
       image_num = image_num + 1;

# %% Calculating the average metrics
print("Calculating the average metrics"); 
avg_psnr = sum(image_psnr)/image_num;
avg_ssim = sum(image_ssim)/image_num;
avg_lpips = sum(image_lpips)/image_num;

print("\nAverage PSNR value of "+TEST_DATA+" dataset: ", avg_psnr); 
print("Average SSIM value of "+TEST_DATA+" dataset: ", avg_ssim); 
print("Average LPIPS value of "+TEST_DATA+" dataset: \n", avg_lpips); 

# %% Visualizing the original and the predicted images
print("Visualizing the original and the predicted images\n");
# Visualizing the original and the predicted images
index = 9;

# Visualizing the original image
org_img = original_images[index].astype("uint8");
cv2_original_image = cv2.resize(org_img, (2*org_img.shape[1],2*org_img.shape[0]), interpolation=cv2.INTER_AREA);
cv2.imshow("Original Image", cv2_original_image);
cv2.waitKey(0);
#cv2.imwrite(TEST_DATA+"_original_image_"+str(index)+".png", original_images[index])

# Visualizing the predicted image
pred_img = predicted_images[index].astype("uint8");
cv2_predicted_image = cv2.resize(pred_img, (2*pred_img.shape[1],2*pred_img.shape[0]), interpolation=cv2.INTER_AREA);
cv2.imshow("Predicted Image (SCS-NET)", cv2_predicted_image);
cv2.waitKey(0);
cv2.imwrite(TEST_DATA+"_predicted_image_"+str(index)+"_scsnet_"+str(int(100*MEASUREMENT_RATE))+".png", predicted_images[index]); 

print("PSNR: ", find_psnr(original_images[index], predicted_images[index]));
print("SSIM: ", find_ssim(original_images[index], predicted_images[index], channel_axis=-1, data_range=255, multichannel=True));
print("LPIPS: ", image_lpips[index]);

# %% Visualizing a patch from the predicted image
print("Visualizing a patch from the predicted image\n"); 
from bm3d import bm3d; 

height, width = pred_img.shape[:2]; 
patch_x = 240; patch_y = 180; 
patch_h = 128; patch_w = 128; 
patch_s = 6;

#pred_img_n = bm3d(pred_img, 14); 
patch = pred_img[patch_y:patch_y + patch_h, patch_x:patch_x+patch_w, :]; 

cv2_patch = cv2.resize(patch, (patch_s*patch.shape[1],patch_s*patch.shape[0]), interpolation=cv2.INTER_AREA); 
cv2.imshow("Patch from the Predicted Image", cv2_patch); 
cv2.waitKey(0); 
cv2.imwrite(TEST_DATA+"_predicted_image_patch_"+str(index)+"_scsnet_"+str(int(100*MEASUREMENT_RATE))+".png", patch); 