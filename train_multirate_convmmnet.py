# %% Importing necessary libraries
print("Importing necessary libraries"); 
import os;
import numpy as np;
import tensorflow as tf;
from math import ceil, exp, log;
from model_multirate_convmmnet import multirate_model_convmmnet;
#from model_multirate_convmmnet import weighted_mse_loss;

# %% Defining functions
print("Defining functions"); 
"""
loss_weights = tf.Variable(tf.random.uniform(shape=[NUM_LAYER], minval=0, maxval=1, dtype=tf.dtypes.float32), 
                           trainable=True);

def _custom_loss(y_true, y_pred):
    loss_mse = [];
    for idx in range(loss_weights.shape[0]):
        diff = y_true[idx] - y_pred[idx];
        sqr = tf.math.multiply(diff, diff);
        loss_mse.append(tf.math.reduce_mean(sqr));
    loss_mse = tf.Variable(loss_mse);
    loss = tf.tensordot(loss_weights, loss_mse, axes=1);
    return loss;
"""

# Defining a dynamic learning rate with a scheduler function.
def scheduler(epoch, lr):
    lr_new = exp(epoch * (log(0.0001) - log(0.1)) / 499 + log(0.1));
    return lr_new;

#strategy = tf.distribute.MirroredStrategy();
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync));

# %% Initializing constants
print("Initializing constants"); 
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

# Model parameters
SAMPLE_RATE = 1;
NUM_LAYER = 16;
NUM_CHANNEL = 3;
nB = IMAGE_SIZE**2*SAMPLE_RATE;
BLOCK_SIZE = ceil(nB / NUM_LAYER);

# %% Training the model 
print("Training the model"); 
#with strategy.scope():
with tf.device(DEVICE):
    # Loading images from their folder
    training_images = np.load(IMAGE_ADDRESS[0]);
    for idx in range(1,len(IMAGE_ADDRESS)):
        training_images = np.concat([training_images, np.load(IMAGE_ADDRESS[idx])], axis=0);
        #training_images = training_images / 255;        # Normalization
        #training_images = np.expand_dims(training_images, axis=3);

    # Controlling data integrity
    print("Shape of the data: ", training_images.shape);
    print("Number of measurements per block: ", nB);
    training_sample_num = training_images.shape[0];
    training_step_size = training_sample_num//BATCH_SIZE;

    # Creating a TensorFlow Dataset object
    #training_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_images)).batch(BATCH_SIZE).shuffle(training_sample_num);
    
    #loss_weights = tf.Variable(tf.random.uniform(shape=[NUM_LAYER], minval=0, maxval=1, dtype=tf.dtypes.float32), trainable=True);
    
    callback = [tf.keras.callbacks.LearningRateScheduler(scheduler)]
    
    loss = [tf.keras.losses.MeanSquaredError()]*NUM_LAYER;
    metrics = ["mae"]*NUM_LAYER;
    loss_weights = [1/NUM_LAYER]*NUM_LAYER;

    # Defining the model
    multirate_model = multirate_model_convmmnet(IMAGE_SIZE, BLOCK_SIZE, nB, NUM_CHANNEL);    # Calling the model
    multirate_model.summary();

    # Defining a dynamic learning rate with a scheduler function
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[training_step_size * 10, training_step_size * 10, training_step_size * 10],
        values=[INITIAL_LR, INITIAL_LR * 0.5, INITIAL_LR * 0.25, INITIAL_LR * 0.1]);

    # Setting hyperparameters and the loss
    multirate_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                            loss=loss,
                            metrics=metrics,
                            loss_weights=loss_weights);

    # Training the model
    history = multirate_model.fit(x=training_images,
                              y=[training_images]*NUM_LAYER,
                              epochs=EPOCH_NUM,
                              batch_size=BATCH_SIZE,
                              steps_per_epoch=ceil(training_sample_num/BATCH_SIZE));

# %% Saving the trained model
print("Saving the trained model"); 
OUTPUT_ADDRESS = "./models/multirate_convmmnet_"+str(BLOCK_SIZE);
if not os.path.exists(OUTPUT_ADDRESS):
    os.makedirs(OUTPUT_ADDRESS);

multirate_model.save(OUTPUT_ADDRESS+"/model.keras");
with open(OUTPUT_ADDRESS+"/history.txt", "w") as f:
    print(history.history, file=f);
