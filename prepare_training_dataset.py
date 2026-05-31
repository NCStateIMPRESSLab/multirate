""" Importing libraries """
import cv2
import numpy as np
import os
import random 
import argparse

parser = argparse.ArgumentParser();
parser.add_argument("--block_size", type=int, default=32, help="Image Block size");
parser.add_argument("--padding", type=int, default=0, help="Block padding");
parser.add_argument("--panchroma", type=bool, default=False, help="Whether includes panchromatic channel or not");
parser.add_argument("--noise", type=bool, default=False, help="whether includes noise or not");
parser.add_argument("--snr", type=int, default=0, help="SNR per block, if present");
parser.add_argument("--input_directory", type=str, help="Input directory that contains training images");
parser.add_argument("--output_directory", type=str, help="Output directory for training image blocks");
parser.add_argument("--output_name", type=str, help="Name of the output file");
args = parser.parse_args();

TRAINING_IMAGE_ADDRESS = args.input_directory;
TRAINING_DATA_SAVE_ADDRESS = args.output_directory;
TRAINING_DATA_NAME = args.output_name;
"""
TRAINING_IMAGE_ADDRESS = "../dataset/BSDS500-master/BSDS500/data/images/train";
TRAINING_DATA_SAVE_ADDRESS = "./dataset";
TRAINING_DATA_NAME = "bdsd500_train";
"""
PANCHROMA = args.panchroma;
BLOCK_SIZE = args.block_size;                               # The size of the repeating filter block, represented as P in the paper
PADDING = BLOCK_SIZE + args.padding;                        # The size of the image block used in training.
NOISE = args.noise;
SNR = args.snr;

x = 10**(SNR/10);

blocks = []
for root, dirs, files in os.walk(TRAINING_IMAGE_ADDRESS, topdown=True):
    for name in files:
       print(os.path.join(TRAINING_IMAGE_ADDRESS, name));
       # OpenCV reads images in BGR format
       img = cv2.imread(os.path.join(TRAINING_IMAGE_ADDRESS, name));
       # Creating the panchromatic channel as the fourth channel to be used in the training
       if PANCHROMA: 
           grayscale = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=2);
           img = np.concatenate((grayscale, img), axis=2);
       
       height, width = img.shape[:2];
       row = int(height/BLOCK_SIZE)-2;
       col = int(width/BLOCK_SIZE)-2;
       for i in range(row):
           for j in range(col):
               blocks.append(img[i*BLOCK_SIZE:i*BLOCK_SIZE+PADDING,j*BLOCK_SIZE:j*BLOCK_SIZE+PADDING]);
               
               if NOISE:
                   sig_pow = np.sum(np.abs(blocks[-1]), axis=None) / (3*BLOCK_SIZE**2);
                   noise = 10**(SNR/10)*sig_pow * np.random.rand(BLOCK_SIZE, BLOCK_SIZE, 3);
                   
                   blocks[-1] = blocks[-1] + noise;

blocks = np.asarray(blocks);

# Shuffling the blocks for a randomized training dataset
shuffle_list = list(range(len(blocks)));
random.shuffle(shuffle_list);
blocks = blocks[shuffle_list];

# Saving the model.
if not os.path.exists(TRAINING_DATA_SAVE_ADDRESS):
    os.makedirs(TRAINING_DATA_SAVE_ADDRESS);

if NOISE: TRAINING_DATA_NAME + "_noise_"+str(SNR)+"dB";

np.save(os.path.join(TRAINING_DATA_SAVE_ADDRESS, TRAINING_DATA_NAME), blocks);

"""
# Visualizing the dataset
index = 5

cv2_img = cv2.resize(blocks[index,:,:,1:].astype("uint8"), (480,480), interpolation=cv2.INTER_AREA)
cv2.imshow("Image Block", cv2_img)
cv2.waitKey(0)

cv2_pan = cv2.resize(blocks[index,:,:,0].astype("uint8"), (480,480), interpolation=cv2.INTER_AREA)
cv2.imshow("Panchromatic Block", cv2_pan)
cv2.waitKey(0)
"""
