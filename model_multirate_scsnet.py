import tensorflow as tf

def wrapper(weights): 
    def weighted_mse(y_true, y_pred):
        [batch, height, width] = y_pred.shape[:3];
        
        print(y_pred.shape);
        print(y_true.shape);
        s_dif = tf.square(y_pred - y_true);
        w_sum = tf.zeros(shape=(height, width, 1));
        for i in range(weights.shape[0]):
            w_sum = w_sum + weights[i]*tf.expand_dims(s_dif[:,:,:,i], axis=3);
        w_sum = tf.reduce_mean(w_sum, axis=0);
        print(w_sum.shape);
        return w_sum;
    return weighted_mse;

# Convolutional layer based measurement block.
def sampling_block(input_layer, filter_num, filter_size):
    measurement = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(filter_size,filter_size), 
                                      strides=filter_size, padding="valid",
                                      activation=None, use_bias=False)(input_layer) # Sampling layer.
    return measurement;
# The initial reconstruction block maps the measurement inputs to an initial solution.
def initial_reconstruction_block(measurements, filter_size, channels):
    init_recon = tf.keras.layers.Conv2D(filters=channels*filter_size**2, kernel_size=(1,1), 
                                        strides=1, padding="valid",
                                        activation=None, use_bias=False)(measurements);
    rescale = tf.keras.layers.Reshape((filter_size,filter_size,channels))(init_recon);
    return rescale;
# Deep reconstruction modules refine the coarser input image.
def deep_reconstruction_block(input_layer, filter_num, filter_size, aggr_num, channels):
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
def multirate_base_layer(measurements, image_size, channel):
    pseudoimage = initial_reconstruction_block(measurements, image_size, channel);
    base_image = deep_reconstruction_block(pseudoimage, 128, 3, 32, channel);
    return base_image;
# Enhancement later of the multirate model.
def multirate_enhancement_layer(prev_image, measurements, image_size, channel):
    pseudoresiduals = initial_reconstruction_block(measurements, image_size, channel);
    enhanced_image = tf.keras.layers.Add()([prev_image, pseudoresiduals]);
    residuals = deep_reconstruction_block(enhanced_image, 128, 3, 32, channel);
    return residuals;

# Defining a SCSNET reconstruction model.
def multirate_model_scsnet(im_size, mes_step, mes_max, channels):
    samp_num = int(mes_max / mes_step);
    samp_rate = list(range(1,samp_num+1));
    for i in range(len(samp_rate)): samp_rate[i] = mes_step * samp_rate[i];

    # Returns a defined CSNet model
    input_layer = tf.keras.Input(shape=(im_size, im_size, channels)) # Input layer
    measurement = sampling_block(input_layer, mes_max, im_size) # Sampling block
    # Base layer
    base = multirate_base_layer(measurement[:,:,:,0:samp_rate[0]], im_size, channels);
    en1 = multirate_enhancement_layer(base, measurement[:,:,:,samp_rate[0]:samp_rate[1]], im_size, channels);
    en2 = multirate_enhancement_layer(en1, measurement[:,:,:,samp_rate[1]:samp_rate[2]], im_size, channels);
    en3 = multirate_enhancement_layer(en2, measurement[:,:,:,samp_rate[2]:samp_rate[3]], im_size, channels);
    en4 = multirate_enhancement_layer(en3, measurement[:,:,:,samp_rate[3]:samp_rate[4]], im_size, channels);
    en5 = multirate_enhancement_layer(en4, measurement[:,:,:,samp_rate[4]:samp_rate[5]], im_size, channels);
    en6 = multirate_enhancement_layer(en5, measurement[:,:,:,samp_rate[5]:samp_rate[6]], im_size, channels);
    en7 = multirate_enhancement_layer(en6, measurement[:,:,:,samp_rate[6]:samp_rate[7]], im_size, channels);
    en8 = multirate_enhancement_layer(en7, measurement[:,:,:,samp_rate[7]:samp_rate[8]], im_size, channels);
    en9 = multirate_enhancement_layer(en8, measurement[:,:,:,samp_rate[8]:samp_rate[9]], im_size, channels);
    en10 = multirate_enhancement_layer(en9, measurement[:,:,:,samp_rate[9]:samp_rate[10]], im_size, channels);
    en11 = multirate_enhancement_layer(en10, measurement[:,:,:,samp_rate[10]:samp_rate[11]], im_size, channels);
    en12 = multirate_enhancement_layer(en11, measurement[:,:,:,samp_rate[11]:samp_rate[12]], im_size, channels);
    en13 = multirate_enhancement_layer(en12, measurement[:,:,:,samp_rate[12]:samp_rate[13]], im_size, channels);
    en14 = multirate_enhancement_layer(en13, measurement[:,:,:,samp_rate[13]:samp_rate[14]], im_size, channels);
    en15 = multirate_enhancement_layer(en14, measurement[:,:,:,samp_rate[14]:samp_rate[15]], im_size, channels);
    
    multirate_model = tf.keras.Model(inputs=input_layer,outputs=[base, en1, en2, en3, en4, en5, en6, en7,
                                                                 en8, en9, en10, en11, en12, en13, en14, en15]);
    return multirate_model;
