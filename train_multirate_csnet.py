# Importing necessary libraries
import numpy as np;
import tensorflow as tf;
from math import ceil;
from model_multirate_csnet import multirate_model_csnet, weighted_mse_loss;

# Initializing constants
IMAGE_ADDRESS = ".\\datasets\\train_images.npy";
IMAGE_SIZE = 32;
BATCH_SIZE = 64;
EPOCH_NUM = 100;
INITIAL_LR = 0.001;

# Model parameters
SAMPLE_RATE = 1;
NUM_LAYER = 16;
NUM_CHANNEL = 1;
nB = IMAGE_SIZE**2*SAMPLE_RATE;
BLOCK_SIZE = 256;

# Loading images from their folder
training_images = np.load(IMAGE_ADDRESS);
training_images = training_images / 255;    # Normalization
training_images = np.expand_dims(training_images, axis=3);

# Controlling data integrity
print("Shape of the data: ", training_images.shape);
print("Number of measurements per block: ", nB);
training_sample_num = training_images.shape[0];
training_step_size = training_sample_num//BATCH_SIZE;
# Creating a TensorFlow Dataset object
training_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_images)).batch(BATCH_SIZE).shuffle(training_sample_num);

loss_weights = tf.Variable(tf.random_normal_initializer(shape=[NUM_LAYER], dtype=tf.float32), 
                           trainable=True);

multirate_model = multirate_model_csnet(IMAGE_SIZE, BLOCK_SIZE, nB);    # Calling the model
multirate_model.summary();

# Defining a dynamic learning rate with a scheduler function.
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[training_step_size * 10, training_step_size * 10, training_step_size * 10],
        values=[INITIAL_LR, INITIAL_LR * 0.5, INITIAL_LR * 0.25, INITIAL_LR * 0.1]);

# Setting hyperparameters and the loss
multirate_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=weighted_mse_loss(loss_weights),
              metrics=["mae"]);

# Training the model
history = multirate_model.fit(training_dataset,
                    epochs=EPOCH_NUM,
                    batch_size=BATCH_SIZE,
                    steps_per_epoch=ceil(training_sample_num/BATCH_SIZE));

multirate_model.save("multirate_csnet_"+str(BLOCK_SIZE))
print(history.history)