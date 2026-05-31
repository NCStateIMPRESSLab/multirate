# Importing necessary libraries
import os
import numpy as np
import tensorflow as tf
from math import ceil
from model_multirate_csnet_drnn import multirate_model_csnet_drnn, create_measurement_matrix

#strategy = tf.distribute.MirroredStrategy();
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync));

# Initializing constants
IMAGE_ADDRESS = ["./dataset/bsds500_train.npy",
                 "./dataset/bsds500_test.npy",
                 "./dataset/bsds500_val.npy",
                 #"./dataset/div2k_train.npy"];
                 #"./dataset/div2k_val.npy",
                 #"./dataset/urban100_1.npy",
                 "./dataset/urban100_2.npy"];
DEVICE = "/gpu:0";

IMAGE_SIZE = 32;
BATCH_SIZE = 64;
EPOCH_NUM = 200;
INITIAL_LR = 0.001;
LOAD = False;

# Model parameters
SAMPLE_RATE = 1;
NUM_LAYER = 16;
NUM_CHANNEL = 3;
nB = IMAGE_SIZE**2*SAMPLE_RATE*NUM_CHANNEL;
BLOCK_SIZE = ceil(nB / NUM_LAYER);

# Loading images from their folder
#with strategy.scope():
# Loading images from their folder
with tf.device(DEVICE):
    training_images = np.load(IMAGE_ADDRESS[0]);
    for idx in range(1,len(IMAGE_ADDRESS)):
        training_images = np.concat([training_images, np.load(IMAGE_ADDRESS[idx])], axis=0);
        #training_images = np.expand_dims(training_images, axis=3);
        #training_images = training_images / 255;        # Normalization

    if not LOAD: 
        mesmax = create_measurement_matrix(NUM_CHANNEL*(IMAGE_SIZE)**2, nB);
        np.save("measurement_matrix"+str(nB)+".npy", mesmax);
    mesmax = np.load("measurement_matrix"+str(nB)+".npy");

    # Controlling data integrity
    print("Shape of the data: ", training_images.shape);
    print("Number of measurements per block: ", nB);
    training_sample_num = training_images.shape[0];
    training_step_size = training_sample_num//BATCH_SIZE;

    training_input = []
    for image in training_images:
        sampled_image = np.matmul(mesmax, image.flatten());
        training_input.append(sampled_image);
    training_input = np.asarray(training_input)[:,np.newaxis,np.newaxis,:];

    rep = int(nB/BLOCK_SIZE);
    training_input = np.tile(training_input, (rep,1,1,1));
    training_images = np.tile(training_images, (rep,1,1,1));

    for index in range(rep-1):
        training_input[index*training_sample_num:(index+1)*training_sample_num,:,:,(index+1)*BLOCK_SIZE:] \
            = np.zeros(shape=(training_sample_num,1,1,(rep-index-1)*BLOCK_SIZE));

    training_sample_num = training_images.shape[0];
    training_step_size = training_sample_num//BATCH_SIZE;

    # Creating a TensorFlow Dataset object
    #training_dataset = tf.data.Dataset.from_tensor_slices((training_input, training_images)).batch(BATCH_SIZE).shuffle(training_sample_num);
    
    loss = [tf.keras.losses.MeanSquaredError()]*NUM_LAYER;
    metrics = ["mae"]*NUM_LAYER;
    loss_weights = [1/NUM_LAYER]*NUM_LAYER;

    drnn_model = multirate_model_csnet_drnn(IMAGE_SIZE, BLOCK_SIZE, nB);   # Calling the model
    drnn_model.summary();

    # Defining a dynamic learning rate with a scheduler function.
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[training_step_size * 10, training_step_size * 10, training_step_size * 10],
            values=[INITIAL_LR, INITIAL_LR * 0.5, INITIAL_LR * 0.25, INITIAL_LR * 0.1]);

    # Setting hyperparameters and the loss
    drnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=["mae"]);

    # Training the model
    history = drnn_model.fit(x=training_input,
                             y=training_images,
                             epochs=EPOCH_NUM,
                             batch_size=BATCH_SIZE,
                             steps_per_epoch=ceil(training_sample_num/BATCH_SIZE));

OUTPUT_ADDRESS = "./models/multirate_drnn_"+str(BLOCK_SIZE);
if not os.path.exists(OUTPUT_ADDRESS):
    os.makedirs(OUTPUT_ADDRESS);

drnn_model.save(OUTPUT_ADDRESS+"/model.keras");

with open(OUTPUT_ADDRESS+"/history.txt", "w") as f:
    print(history.history, file=f);
