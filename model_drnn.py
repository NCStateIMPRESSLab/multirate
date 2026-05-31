import tensorflow as tf
from numpy.random import randn
from scipy.linalg import orth

# Convolutional layer based measurement block.
def measurement_csnet(input_image, image_size, scale):
    measurement = tf.keras.layers.Conv2D(filters=scale, kernel_size=(image_size,image_size),
                                         strides=(image_size,image_size), padding="valid",
                                         activation=None, use_bias=False)(input_image)
    return measurement

def create_measurement_matrix(im_size, mes_size):
    # Creating a normal sampling matrix and normalizing it
    measurement_matrix = randn(mes_size, im_size ** 2)
    measurement_matrix = orth(measurement_matrix.transpose()).transpose()
    
    return measurement_matrix

# The initial reconstruction block maps the measurement inputs to an initial solution.
def initial_reconstruction_drnn(measurements, filter_size, channels):
    mes = tf.keras.layers.Dense(units=1024)(measurements)
    mes_rshp = tf.keras.layers.Reshape(target_shape=(filter_size,filter_size,1))(mes)
    init_recon = tf.keras.layers.Conv2D(filters=channels, kernel_size=(3,3), 
                                        strides=1, padding="same",
                                        activation=None, use_bias=False)(mes_rshp)
    return init_recon
# Deep reconstruction modules refine the coarser input image.
def reconstruction_block_csnet(input_layer, filter_num, filter_size, aggr_num, channels):
    ftr_ext = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(filter_size,filter_size), 
                                     strides=1, padding="same", activation="relu", use_bias=True)(input_layer)
    shrink = tf.keras.layers.Conv2D(filters=aggr_num, kernel_size=(filter_size,filter_size), 
                                    strides=1, padding="same", activation="relu", use_bias=True)(ftr_ext)
    nl_map = tf.keras.layers.Conv2D(filters=aggr_num, kernel_size=(filter_size,filter_size), 
                                    strides=1, padding="same", activation="relu", use_bias=True)(shrink)
    for i in range(12):
        nl_map = tf.keras.layers.Conv2D(filters=aggr_num, kernel_size=(filter_size,filter_size), strides=1, padding="same",
                                    activation="relu", use_bias=True)(nl_map)
    expnd = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(filter_size,filter_size),
                                   strides=1, padding="same", activation="relu", use_bias=True)(nl_map)
    aggregate = tf.keras.layers.Conv2D(filters=channels, kernel_size=(filter_size,filter_size), 
                                       strides=1, padding="same", use_bias=True)(expnd)
    res = tf.keras.layers.Add()([input_layer, aggregate])
    return res

# Defining a DRNN-based multirate reconstruction model.
def multirate_model_drnn(im_size, mes_size):
    input_layer = tf.keras.Input(shape=(im_size**2))
    pseudoimage = initial_reconstruction_drnn(input_layer, im_size, 1)
    output_layer = reconstruction_block_csnet(pseudoimage, 128, 3, 32, 1)
    
    multirate_model = tf.keras.Model(inputs=input_layer, 
                                     outputs=output_layer)
    
    return multirate_model