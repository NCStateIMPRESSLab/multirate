import tensorflow as tf

# Weighted loss
def weighted_mse_loss(loss_weights):
    def _custom_loss(y_true, y_pred):
        loss = 0;
        for idx in range(loss_weights.shape[0]):
            diff = y_true[:,idx] - y_pred[:,idx];
            sqr = tf.math.multiply(diff, diff);
            loss = loss + loss_weights[idx] * tf.math.reduce_mean(sqr);
        return loss;
    return _custom_loss;

# FC MLP-based measurement block.
def measurement_convmmnet(input_image, scale):
    flatten = tf.keras.layers.Flatten()(input_image);
    measurement = tf.keras.layers.Dense(scale)(flatten);
    return measurement

# The initial reconstruction block maps the measurement inputs to an initial solution.
def initial_reconstruction_convmmnet(measurements, image_size, mes, channel):
    init_recon = tf.keras.layers.Dense(channel*image_size**2)(measurements)
    init_recon = tf.keras.layers.Reshape(target_shape=(image_size, image_size, channel))(init_recon)
    return init_recon
# Deep reconstruction modules refine the coarser input image.
def reconstruction_block_convmmnet(input_layer, filter_size, channel):
    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(filter_size, filter_size), 
                                  strides=1, padding="same",
                                  activation="relu", use_bias=True)(input_layer);
    conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(filter_size, filter_size), 
                                  strides=1, padding="same",
                                  activation="relu", use_bias=True)(conv1);
    conv3 = tf.keras.layers.Conv2D(filters=channel, kernel_size=(filter_size, filter_size), 
                                  strides=1, padding="same",
                                  activation="relu", use_bias=True)(conv2);
    res = tf.keras.layers.Add()([input_layer, conv3]);
    return res;

# Base later of the multirate model.
def multirate_base_layer_convmmnet(measurements, image_size, mes, channel):
    pseudoimage = initial_reconstruction_convmmnet(measurements, image_size, mes, channel);
    refinement = reconstruction_block_convmmnet(pseudoimage, 5, channel);
    """for i in range(9):
        refinement = reconstruction_block_convmmnet(refinement, 5)"""
    return refinement;

# Enhancement later of the multirate model.
def multirate_enhancement_layer_convmmnet(prev_image, measurements, image_size, mes, channel):
    pseudoresiduals = initial_reconstruction_convmmnet(measurements, image_size, mes, channel);
    residuals = reconstruction_block_convmmnet(pseudoresiduals, 5, channel);
    """for i in range(9):
        residuals = reconstruction_block_convmmnet(residuals, 5)"""
    enhanced_image = tf.keras.layers.Add()([prev_image, residuals]);
    return enhanced_image;

# Defining a ConvMMNet-based multirate reconstruction model.
def multirate_model_convmmnet(im_size, mes_step, mes_max, channel):
    input_image = tf.keras.Input(shape=(im_size, im_size, channel));
    measurements = measurement_convmmnet(input_image, mes_max);
    # Reconstruction layer
    base = multirate_base_layer_convmmnet(measurements[:,:mes_step], im_size, mes_step, channel);
    en1 = multirate_enhancement_layer_convmmnet(base, measurements[:,mes_step:2*mes_step], im_size, mes_step, channel);
    en2 = multirate_enhancement_layer_convmmnet(en1, measurements[:,2*mes_step:3*mes_step], im_size, mes_step, channel);
    en3 = multirate_enhancement_layer_convmmnet(en2, measurements[:,3*mes_step:4*mes_step], im_size, mes_step, channel);
    en4 = multirate_enhancement_layer_convmmnet(en3, measurements[:,4*mes_step:5*mes_step], im_size, mes_step, channel);
    en5 = multirate_enhancement_layer_convmmnet(en4, measurements[:,5*mes_step:6*mes_step], im_size, mes_step, channel);
    en6 = multirate_enhancement_layer_convmmnet(en5, measurements[:,6*mes_step:7*mes_step], im_size, mes_step, channel);
    en7 = multirate_enhancement_layer_convmmnet(en6, measurements[:,7*mes_step:8*mes_step], im_size, mes_step, channel);
    en8 = multirate_enhancement_layer_convmmnet(en7, measurements[:,8*mes_step:9*mes_step], im_size, mes_step, channel);
    en9 = multirate_enhancement_layer_convmmnet(en8, measurements[:,9*mes_step:10*mes_step], im_size, mes_step, channel);
    en10 = multirate_enhancement_layer_convmmnet(en9, measurements[:,10*mes_step:11*mes_step], im_size, mes_step, channel);
    en11 = multirate_enhancement_layer_convmmnet(en10, measurements[:,11*mes_step:12*mes_step], im_size, mes_step, channel);
    en12 = multirate_enhancement_layer_convmmnet(en11, measurements[:,12*mes_step:13*mes_step], im_size, mes_step, channel);
    en13 = multirate_enhancement_layer_convmmnet(en12, measurements[:,13*mes_step:14*mes_step], im_size, mes_step, channel);
    en14 = multirate_enhancement_layer_convmmnet(en13, measurements[:,14*mes_step:15*mes_step], im_size, mes_step, channel);
    en15 = multirate_enhancement_layer_convmmnet(en14, measurements[:,15*mes_step:16*mes_step], im_size, mes_step, channel);

    #base_out = tf.keras.layers.Reshape((1,) + base.shape[1:])(base);
    #en1_out = tf.keras.layers.Reshape((1,) + en1.shape[1:])(en1);
    #en2_out = tf.keras.layers.Reshape((1,) + en2.shape[1:])(en2);
    #en3_out = tf.keras.layers.Reshape((1,) + en3.shape[1:])(en3);
    #output = tf.keras.layers.Concatenate(axis=1)([base_out, en1_out, en2_out, en3_out]);

    multirate_model = tf.keras.Model(inputs=input_image,
                                     outputs=[base, en1, en2, en3]);#, en4, en5, en6, en7,
                                              #en8, en9, en10, en11, en12, en13, en14, en15]);
    return multirate_model

