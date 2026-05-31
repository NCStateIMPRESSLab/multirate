# Importing necessary libraries
import numpy as np
import tensorflow as tf
from math import ceil

# Initializing hyperparameter constants
IMAGE_SIZE = 32
SAMPLE_RATE = 0.375
BATCH_SIZE = 64
EPOCH_NUM = 100
IMAGE_ADDRESS = "..\\datasets\\train_images.npy"
# Initializing model constants
B = 32 # Sampling filter size
l = 1 # Image channel number
nB = int(SAMPLE_RATE*l*B**2) # Filter number
F = 3
D = 64

# Loading images from their folder
training_images = np.load(IMAGE_ADDRESS)
training_images = np.expand_dims(training_images, axis=3)
# Controlling data integrity
print("Shape of the data: ", training_images.shape)
print("Number of measurements per block: ", nB)
training_sample_num = training_images.shape[0]
# Creating a TensorFlow Dataset object
training_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_images)).batch(BATCH_SIZE).shuffle(training_sample_num)

# Measurement block consists of a 2D convolutional layer defined in the paper as "sampling layer" and a flattening layer.
# The convolution operation here does not have a bias value and an activation function.
def sampling_block(input_layer, filter_num, filter_size):
    sampling = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(filter_size, filter_size), 
                                      strides=filter_size, padding="valid",
                                      activation=None, use_bias=False)(input_layer) # Sampling layer.
    return sampling
# The initial reconstruction block maps the measurement inputs to an initial solution.
def initial_reconstruction_block(measurement, channel, filter_size):
    init_recon = tf.keras.layers.Conv2D(filters=channel*filter_size**2, kernel_size=(1,1), 
                                        strides=1, padding="valid",
                                        activation=None, use_bias=False)(measurement)
    rescale = tf.keras.layers.Reshape((init_recon.shape[1],init_recon.shape[2],
                                       filter_size,filter_size))(init_recon)
    concatenate = concatenation_block(rescale)
    return concatenate
# Subblock which creates a concatenated pseudoimage from the reconstructed pseudoblocks.
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
def reconstruction_block(input_layer, filter_num, filter_size):
    conv = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(filter_size,filter_size), 
                                  strides=1, padding="same",
                                  activation="relu", use_bias=True)(input_layer)
    res = tf.keras.layers.Add()([input_layer, conv])
    return res

input_layer = tf.keras.Input(shape=(IMAGE_SIZE,IMAGE_SIZE,l)) # Input layer
measurement = sampling_block(input_layer, nB, B) # Sampling block
pseudoimage = initial_reconstruction_block(measurement, l, B) # Initial Reconstruction Block
# The deep reconstruction block is a simple fully connected convolutional layer stack.
ftr_ext = tf.keras.layers.Conv2D(filters=D, kernel_size=(F,F), strides=1, padding="same",
                                activation="relu", use_bias=True)(pseudoimage)
# Consecutive reconstruction blocks
recon1 = reconstruction_block(ftr_ext, D, F)
recon2 = reconstruction_block(recon1, D, F)
recon3 = reconstruction_block(recon2, D, F)
recon4 = reconstruction_block(recon3, D, F)
recon5 = reconstruction_block(recon4, D, F)
output_layer = tf.keras.layers.Conv2D(filters=l, kernel_size=(F,F), strides=1, padding="same",
                                activation="relu", use_bias=True)(recon5)

csnet = tf.keras.Model(inputs=input_layer,outputs=output_layer)
print(csnet.summary())

# Defining a dynamic learning rate with a scheduler function.
def scheduler(epoch, lr):
    if epoch % 20 == 0:
        return lr * (1/2)
    else:
        return lr
callback = [tf.keras.callbacks.LearningRateScheduler(scheduler)]
csnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, 
                                                 beta_1=0.9, beta_2=0.999, epsilon=1e-07),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=["mae"])
history = csnet.fit(training_dataset,
                    epochs=EPOCH_NUM,
                    steps_per_epoch=ceil(training_sample_num/BATCH_SIZE),
                    callbacks=callback)

csnet.save("cs_net_"+str(nB))

print(history.history)