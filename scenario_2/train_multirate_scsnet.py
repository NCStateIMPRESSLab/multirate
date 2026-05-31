# Importing necessary libraries
import os
import numpy as np
import tensorflow as tf
from math import ceil
from multirate_model_csnet import measurement_csnet, multirate_base_layer, multirate_enhancement_layer, WeightedOutputLayer

# Initializing constants
IMAGE_ADDRESS = "..\\datasets\\train_images.npy"
IMAGE_SIZE = 32
BATCH_SIZE = 64
EPOCH_NUM = 500
INITIAL_LR = 0.0001

# Model parameters
SAMPLE_RATE = 1
nB = IMAGE_SIZE**2*SAMPLE_RATE
BLOCK_SIZE = 256

# Loading images from their folder
training_images = np.load(IMAGE_ADDRESS)
# Normalization
training_images = training_images / 255
training_images = np.expand_dims(training_images, axis=3)

# Controlling data integrity
print("Shape of the data: ", training_images.shape)
print("Number of measurements per block: ", nB)
training_sample_num = training_images.shape[0]
training_step_size = training_sample_num//BATCH_SIZE
# Creating a TensorFlow Dataset object
training_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_images)).batch(BATCH_SIZE).shuffle(training_sample_num)
# Defining the model

# Defining the model
input_layer = tf.keras.Input(shape=(IMAGE_SIZE,IMAGE_SIZE,1))
measurements = measurement_csnet(input_layer, IMAGE_SIZE, nB)
en0 = multirate_base_layer(measurements[:,:,:,:BLOCK_SIZE], IMAGE_SIZE)
en1 = multirate_enhancement_layer(en0, measurements[:,:,:,BLOCK_SIZE:2*BLOCK_SIZE], IMAGE_SIZE)
en2 = multirate_enhancement_layer(en1, measurements[:,:,:,2*BLOCK_SIZE:3*BLOCK_SIZE], IMAGE_SIZE)
en3 = multirate_enhancement_layer(en2, measurements[:,:,:,3*BLOCK_SIZE:4*BLOCK_SIZE], IMAGE_SIZE)
"""
en4 = multirate_enhancement_layer(en3, measurements[:,:,:,4*BLOCK_SIZE:5*BLOCK_SIZE], IMAGE_SIZE)
en5 = multirate_enhancement_layer(en4, measurements[:,:,:,5*BLOCK_SIZE:6*BLOCK_SIZE], IMAGE_SIZE)
en6 = multirate_enhancement_layer(en5, measurements[:,:,:,6*BLOCK_SIZE:7*BLOCK_SIZE], IMAGE_SIZE)
en7 = multirate_enhancement_layer(en6, measurements[:,:,:,7*BLOCK_SIZE:8*BLOCK_SIZE], IMAGE_SIZE)
en8 = multirate_enhancement_layer(en7, measurements[:,:,:,8*BLOCK_SIZE:9*BLOCK_SIZE], IMAGE_SIZE)
en9 = multirate_enhancement_layer(en8, measurements[:,:,:,9*BLOCK_SIZE:10*BLOCK_SIZE], IMAGE_SIZE)
en10 = multirate_enhancement_layer(en9, measurements[:,:,:,10*BLOCK_SIZE:11*BLOCK_SIZE], IMAGE_SIZE)
en11 = multirate_enhancement_layer(en10, measurements[:,:,:,11*BLOCK_SIZE:12*BLOCK_SIZE], IMAGE_SIZE)
en12 = multirate_enhancement_layer(en11, measurements[:,:,:,12*BLOCK_SIZE:13*BLOCK_SIZE], IMAGE_SIZE)
en13 = multirate_enhancement_layer(en12, measurements[:,:,:,13*BLOCK_SIZE:14*BLOCK_SIZE], IMAGE_SIZE)
en14 = multirate_enhancement_layer(en13, measurements[:,:,:,14*BLOCK_SIZE:15*BLOCK_SIZE], IMAGE_SIZE)
en15 = multirate_enhancement_layer(en14, measurements[:,:,:,15*BLOCK_SIZE:16*BLOCK_SIZE], IMAGE_SIZE)
en16 = multirate_enhancement_layer(en15, measurements[:,:,:,16*BLOCK_SIZE:17*BLOCK_SIZE], IMAGE_SIZE)
en17 = multirate_enhancement_layer(en16, measurements[:,:,:,17*BLOCK_SIZE:18*BLOCK_SIZE], IMAGE_SIZE)
en18 = multirate_enhancement_layer(en17, measurements[:,:,:,18*BLOCK_SIZE:19*BLOCK_SIZE], IMAGE_SIZE)
en19 = multirate_enhancement_layer(en18, measurements[:,:,:,19*BLOCK_SIZE:20*BLOCK_SIZE], IMAGE_SIZE)
en20 = multirate_enhancement_layer(en19, measurements[:,:,:,20*BLOCK_SIZE:21*BLOCK_SIZE], IMAGE_SIZE)
en21 = multirate_enhancement_layer(en20, measurements[:,:,:,21*BLOCK_SIZE:22*BLOCK_SIZE], IMAGE_SIZE)
en22 = multirate_enhancement_layer(en21, measurements[:,:,:,22*BLOCK_SIZE:23*BLOCK_SIZE], IMAGE_SIZE)
en23 = multirate_enhancement_layer(en22, measurements[:,:,:,23*BLOCK_SIZE:24*BLOCK_SIZE], IMAGE_SIZE)
en24 = multirate_enhancement_layer(en23, measurements[:,:,:,24*BLOCK_SIZE:25*BLOCK_SIZE], IMAGE_SIZE)
en25 = multirate_enhancement_layer(en24, measurements[:,:,:,25*BLOCK_SIZE:26*BLOCK_SIZE], IMAGE_SIZE)
en26 = multirate_enhancement_layer(en25, measurements[:,:,:,26*BLOCK_SIZE:27*BLOCK_SIZE], IMAGE_SIZE)
en27 = multirate_enhancement_layer(en26, measurements[:,:,:,27*BLOCK_SIZE:28*BLOCK_SIZE], IMAGE_SIZE)
en28 = multirate_enhancement_layer(en27, measurements[:,:,:,28*BLOCK_SIZE:29*BLOCK_SIZE], IMAGE_SIZE)
en29 = multirate_enhancement_layer(en28, measurements[:,:,:,29*BLOCK_SIZE:30*BLOCK_SIZE], IMAGE_SIZE)
en30 = multirate_enhancement_layer(en29, measurements[:,:,:,30*BLOCK_SIZE:31*BLOCK_SIZE], IMAGE_SIZE)
en31 = multirate_enhancement_layer(en30, measurements[:,:,:,31*BLOCK_SIZE:32*BLOCK_SIZE], IMAGE_SIZE)"""
multirate_model = tf.keras.Model(inputs=input_layer, outputs=[en0,en1,en2,en3])
""",en4,en5,en6,en7,
                                                              en8,en9,en10,en11,en12,en13,en14,en15,
                                                              en16,en17,en18,en19,en20,en21,en22,en23,
                                                              en24,en25,en26,en27,en28,en29,en30,en31"""
# Defining a dynamic learning rate with a scheduler function.
"""
def scheduler(epoch, lr):
    if epoch % 20 == 0:
        return lr * (1/2)
    else:
        return lr
"""
def scheduler(epoch, lr):
    lr_new = exp(epoch * (log(0.0001) - log(0.1)) / 499 + log(0.1))
    return lr_new
callback = [tf.keras.callbacks.LearningRateScheduler(scheduler)]

# Setting hyperparameters and the loss
multirate_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=["mae"], loss_weights=(0.25,0.25,0.25,0.25))

history = multirate_model.fit(training_dataset,
                    epochs=EPOCH_NUM,
                    steps_per_epoch=ceil(training_sample_num/BATCH_SIZE))

multirate_model.save("multirate_csnet_"+str(BLOCK_SIZE)+"_fixed")
print(history.history)
