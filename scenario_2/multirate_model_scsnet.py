import tensorflow as tf

# Convolutional layer based measurement block.
def measurement_csnet(input_image, image_size, scale):
    measurement = tf.keras.layers.Conv2D(filters=scale, kernel_size=(image_size,image_size),
                                         strides=(image_size,image_size), padding="valid",
                                         activation=None, use_bias=False)(input_image)
    return measurement
# The initial reconstruction block maps the measurement inputs to an initial solution.
def initial_reconstruction_csnet(measurements, filter_size, channels):
    init_recon = tf.keras.layers.Conv2D(filters=channels*filter_size**2, kernel_size=(1,1), 
                                        strides=1, padding="valid",
                                        activation=None, use_bias=False)(measurements)
    rescale = tf.keras.layers.Reshape((filter_size,filter_size,1))(init_recon)
    return rescale
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
# Base later of the multirate model.
def multirate_base_layer(measurements, image_size):
    pseudoimage = initial_reconstruction_csnet(measurements, image_size, 1)
    base_image = reconstruction_block_csnet(pseudoimage, 128, 3, 32, 1)
    return base_image
# Enhancement later of the multirate model.
def multirate_enhancement_layer(prev_image, measurements, image_size):
    pseudoresiduals = initial_reconstruction_csnet(measurements, image_size, 1)
    residuals = reconstruction_block_csnet(pseudoresiduals, 128, 3, 32, 1)
    enhanced_image = tf.keras.layers.Add()([prev_image, residuals])
    return enhanced_image
# Weighted output layer for the implementation of weighted mean square error.
class WeightedMSE(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        w_init = tf.random_uniform_initializer(minval=0.01, maxval=0.99)
        self.w = tf.Variable(w_init(shape=(num_outputs, 1)))
        self.num_outputs = num_outputs
        super(WeightedMSE, self).__init__()
        
    def call(self, input_tensors):
        return input_tensors
    
    def get_config(self):
        return {"Weights": self.w}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
def wrapper(weights): 
    def weighted_mse(y_pred, y_true):
        print(y_pred.shape)
        print(y_true.shape)
        [batch, height, width] = y_pred.shape[:3]
        s_dif = tf.square(y_pred - y_true)
        w_sum = tf.zeros(shape=(batch, height, width, 1))
        for i in range(len(weights)):
            w_sum = w_sum + weights[i]*tf.expand_dims(s_dif[:,:,:,i], axis=3)
        w_sum = tf.reduce_mean(w_sum, axis=-1)
        return w_sum
    return weighted_mse