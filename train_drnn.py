# %% Importing necessary libraries
print("Importing necessary libraries"); 
import numpy as np; 
import tensorflow as tf; 
from model_drnn import multirate_model_drnn, create_measurement_matrix; 
from math import ceil; 

# %% Initializing constants
print("Initializing constants"); 
IMAGE_ADDRESS = "./datasets/train_images.npy"; 
IMAGE_SIZE = 32; 
BATCH_SIZE = 128; 
EPOCH_NUM = 100; 
INITIAL_LR = 0.001; 
LOAD = True; 

# Model parameters
SAMPLE_RATE = 1; 
nB = IMAGE_SIZE**2*SAMPLE_RATE; 
BLOCK_SIZE = 64; 

# Loading images from their folder
training_images = np.load(IMAGE_ADDRESS); 
# Normalization
training_images = np.expand_dims(training_images, axis=3); 

if not LOAD: 
    mesmax = create_measurement_matrix(IMAGE_SIZE, nB); 
    np.save("measurement_matrix.npy", mesmax); 
mesmax = np.load("measurement_matrix.npy"); 

# Controlling data integrity
print("Shape of the data: ", training_images.shape); 
print("Number of measurements per block: ", nB); 
training_sample_num = training_images.shape[0]; 

training_input = []; 
for image in training_images:
    sampled_image = np.matmul(mesmax, image.flatten()); 
    training_input.append(sampled_image); 
training_input = np.asarray(training_input); 

rep = int(nB/BLOCK_SIZE); 
training_input = np.tile(training_input, (rep,1)); 
training_images = np.tile(training_images, (rep,1,1,1)); 

for index in range(rep-1):
    training_input[index*training_sample_num:(index+1)*training_sample_num,(index+1)*BLOCK_SIZE:] \
        = np.zeros(shape=(training_sample_num,(rep-index-1)*BLOCK_SIZE)); 

training_sample_num = training_images.shape[0]; 
training_step_size = training_sample_num//BATCH_SIZE; 

# Creating a TensorFlow Dataset object
training_dataset = tf.data.Dataset.from_tensor_slices((training_input, training_images)).batch(BATCH_SIZE).shuffle(training_sample_num); 

model = multirate_model_drnn(IMAGE_SIZE, nB);
#model.summary()

# Defining a dynamic learning rate with a scheduler function.
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[training_step_size * 10, training_step_size * 10, training_step_size * 10, training_step_size * 10],
        values=[INITIAL_LR, INITIAL_LR * 0.5, INITIAL_LR * 0.25, INITIAL_LR * 0.1, INITIAL_LR * 0.1]); 

# Setting hyperparameters and the loss
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=["mae"]); 

# Training the model
history = model.fit(training_dataset,
                    epochs=EPOCH_NUM,
                    batch_size=BATCH_SIZE,
                    steps_per_epoch=ceil(training_sample_num/BATCH_SIZE)); 

model.save("multirate_drnn_"+str(BLOCK_SIZE)); 
print(history.history); 