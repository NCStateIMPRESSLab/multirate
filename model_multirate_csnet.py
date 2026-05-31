import tensorflow as tf

# Convolutional layer based measurement block.
def measurement_csnet(input_image, image_size, scale):
    measurement = tf.keras.layers.Conv2D(filters=scale, kernel_size=(image_size,image_size),
                                         strides=(image_size,image_size), padding="valid",
                                         activation=None, use_bias=False)(input_image);
    return measurement
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
def multirate_model_csnet(im_size, mes_size, mes_max):
    input_layer = tf.keras.Input(shape=(im_size,im_size,1));
    measurements = measurement_csnet(input_layer, image_size=im_size, scale=mes_max);
    # Reconstruction layer
    base = multirate_base_layer(measurements[:,:,:,:mes_size], im_size);
    en1 = multirate_enhancement_layer(base, measurements[:,:,:,mes_size:2*mes_size], im_size);
    en2 = multirate_enhancement_layer(en1, measurements[:,:,:,2*mes_size:3*mes_size], im_size);
    en3 = multirate_enhancement_layer(en2, measurements[:,:,:,3*mes_size:4*mes_size], im_size);
    #en4 = multirate_enhancement_layer(en3, measurements[:,:,:,4*mes_size:5*mes_size], im_size);
    #en5 = multirate_enhancement_layer(en4, measurements[:,:,:,5*mes_size:6*mes_size], im_size);
    #en6 = multirate_enhancement_layer(en5, measurements[:,:,:,6*mes_size:7*mes_size], im_size);
    #en7 = multirate_enhancement_layer(en6, measurements[:,:,:,7*mes_size:8*mes_size], im_size);
    #en8 = multirate_enhancement_layer(en7, measurements[:,:,:,8*mes_size:9*mes_size], im_size);
    #en9 = multirate_enhancement_layer(en8, measurements[:,:,:,9*mes_size:10*mes_size], im_size);
    #en10 = multirate_enhancement_layer(en9, measurements[:,:,:,10*mes_size:11*mes_size], im_size);
    #en11 = multirate_enhancement_layer(en10, measurements[:,:,:,11*mes_size:12*mes_size], im_size);
    #en12 = multirate_enhancement_layer(en11, measurements[:,:,:,12*mes_size:13*mes_size], im_size);
    #en13 = multirate_enhancement_layer(en12, measurements[:,:,:,13*mes_size:14*mes_size], im_size);
    #en14 = multirate_enhancement_layer(en13, measurements[:,:,:,14*mes_size:15*mes_size], im_size);
    #en15 = multirate_enhancement_layer(en14, measurements[:,:,:,15*mes_size:16*mes_size], im_size);
    #en16 = multirate_enhancement_layer(en15, measurements[:,:,:,16*mes_size:17*mes_size], im_size);
    #en17 = multirate_enhancement_layer(en16, measurements[:,:,:,17*mes_size:18*mes_size], im_size);
    #en18 = multirate_enhancement_layer(en17, measurements[:,:,:,18*mes_size:19*mes_size], im_size);
    #en19 = multirate_enhancement_layer(en18, measurements[:,:,:,19*mes_size:20*mes_size], im_size);
    #en20 = multirate_enhancement_layer(en19, measurements[:,:,:,20*mes_size:21*mes_size], im_size);
    #en21 = multirate_enhancement_layer(en20, measurements[:,:,:,21*mes_size:22*mes_size], im_size);
    #en22 = multirate_enhancement_layer(en21, measurements[:,:,:,22*mes_size:23*mes_size], im_size);
    #en23 = multirate_enhancement_layer(en22, measurements[:,:,:,23*mes_size:24*mes_size], im_size);
    #en24 = multirate_enhancement_layer(en23, measurements[:,:,:,24*mes_size:25*mes_size], im_size);
    #en25 = multirate_enhancement_layer(en24, measurements[:,:,:,25*mes_size:26*mes_size], im_size);
    #en26 = multirate_enhancement_layer(en25, measurements[:,:,:,26*mes_size:27*mes_size], im_size);
    #en27 = multirate_enhancement_layer(en26, measurements[:,:,:,27*mes_size:28*mes_size], im_size);
    #en28 = multirate_enhancement_layer(en27, measurements[:,:,:,28*mes_size:29*mes_size], im_size);
    #en29 = multirate_enhancement_layer(en28, measurements[:,:,:,29*mes_size:30*mes_size], im_size);
    #en30 = multirate_enhancement_layer(en29, measurements[:,:,:,30*mes_size:31*mes_size], im_size);
    #en31 = multirate_enhancement_layer(en30, measurements[:,:,:,31*mes_size:32*mes_size], im_size);
    
    multirate_model = tf.keras.Model(inputs=input_layer, 
                                     outputs=[base, en1, en2, en3]);#, en4, en5, en6, en7, 
                                              #en8, en9, en10, en11, en12, en13, en14, en15,
                                              #en16, en17, en18, en19, en20, en21, en22, en23,
                                              #en24, en25, en26, en27, en28, en29, en30, en31]);
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