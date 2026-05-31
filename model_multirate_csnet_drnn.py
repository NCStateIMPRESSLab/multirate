import tensorflow as tf;
from numpy.random import randn;
from scipy.linalg import orth;

def create_measurement_matrix(input_size, mes_size):
    # Creating a normal sampling matrix and normalizing it
    measurement_matrix = randn(mes_size, input_size);
    measurement_matrix = orth(measurement_matrix.transpose()).transpose();
    return measurement_matrix;

# The initial reconstruction block maps the measurement inputs to an initial solution.
def initial_reconstruction_csnet(measurements, filter_size, channels):
    init_recon = tf.keras.layers.Conv2D(filters=channels*filter_size**2, kernel_size=(1,1), 
                                        strides=1, padding="valid",
                                        activation=None, use_bias=False)(measurements);
    rescale = tf.keras.layers.Reshape((filter_size,filter_size,1))(init_recon);
    return rescale;

# Deep reconstruction modules refine the coarser input image.
def reconstruction_block_csnet(input_layer, filter_num, filter_size, aggr_num, channels):
    ftr_ext = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(filter_size,filter_size), 
                                     strides=1, padding="same", activation="relu", use_bias=True)(input_layer);
    shrink = tf.keras.layers.Conv2D(filters=aggr_num, kernel_size=(filter_size,filter_size), 
                                    strides=1, padding="same", activation="relu", use_bias=True)(ftr_ext);
    nl_map = tf.keras.layers.Conv2D(filters=aggr_num, kernel_size=(filter_size,filter_size), 
                                    strides=1, padding="same", activation="relu", use_bias=True)(shrink);
    for i in range(12):
        nl_map = tf.keras.layers.Conv2D(filters=aggr_num, kernel_size=(filter_size,filter_size), strides=1, padding="same",
                                    activation="relu", use_bias=True)(nl_map);
    expnd = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(filter_size,filter_size),
                                   strides=1, padding="same", activation="relu", use_bias=True)(nl_map);
    aggregate = tf.keras.layers.Conv2D(filters=channels, kernel_size=(filter_size,filter_size), 
                                       strides=1, padding="same", use_bias=True)(expnd);
    res = tf.keras.layers.Add()([input_layer, aggregate]);
    return res
# Base later of the multirate model.
def multirate_base_layer(measurements, image_size):
    pseudoimage = initial_reconstruction_csnet(measurements, image_size, 1);
    base_image = reconstruction_block_csnet(pseudoimage, 128, 3, 32, 1);
    return base_image;
# Enhancement later of the multirate model.
def multirate_enhancement_layer(prev_image, measurements, image_size):
    pseudoresiduals = initial_reconstruction_csnet(measurements, image_size, 1);
    residuals = reconstruction_block_csnet(pseudoresiduals, 128, 3, 32, 1);
    enhanced_image = tf.keras.layers.Add()([prev_image, residuals]);
    return enhanced_image;

# Defining a CSNET-based multirate reconstruction model.
def multirate_model_csnet_drnn(im_size, mes_size, mes_max):
    input_layer = tf.keras.Input(shape=(1,1,im_size**2));
    #input_layer = tf.keras.layers.Reshape(target_shape=(1,1,im_size**2))(input_vector);
    # Reconstruction layer
    base = multirate_base_layer(input_layer[:,:,:,:mes_size], im_size);
    en1 = multirate_enhancement_layer(base, input_layer[:,:,:,mes_size:2*mes_size], im_size);
    en2 = multirate_enhancement_layer(en1, input_layer[:,:,:,2*mes_size:3*mes_size], im_size);
    en3 = multirate_enhancement_layer(en2, input_layer[:,:,:,3*mes_size:4*mes_size], im_size);
    en4 = multirate_enhancement_layer(en3, input_layer[:,:,:,4*mes_size:5*mes_size], im_size);
    en5 = multirate_enhancement_layer(en4, input_layer[:,:,:,5*mes_size:6*mes_size], im_size);
    en6 = multirate_enhancement_layer(en5, input_layer[:,:,:,6*mes_size:7*mes_size], im_size);
    en7 = multirate_enhancement_layer(en6, input_layer[:,:,:,7*mes_size:8*mes_size], im_size);
    
    multirate_model = tf.keras.Model(inputs=input_layer, 
                                     outputs=[base, en1, en2, en3, en4, en5, en6, en7]);
    return multirate_model;

"""
def wrapper(weights): 
    def weighted_mse(y_true, y_pred):
        [batch, height, width] = y_pred.shape[:3]
        print(y_pred.shape)
        print(y_true.shape)
        s_dif = tf.square(y_pred - y_true)
        w_sum = tf.zeros(shape=(height, width, 1))
        for i in range(weights.shape[0]):
            w_sum = w_sum + weights[i]*tf.expand_dims(s_dif[:,:,:,i], axis=3)
        w_sum = tf.reduce_mean(w_sum, axis=0)
        print(w_sum.shape)
        return w_sum
    return weighted_mse

# Weighted output loss function for the implementation of weighted mean square error.
def custom_loss(weight_a, weight_b):
    def _custom_loss():
        # This can include any arbitrary logic
        loss = tf.norm(weight_a) + tf.norm(weight_b)
        return loss
    return _custom_loss"""