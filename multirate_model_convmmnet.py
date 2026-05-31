import tensorflow as tf

# Convolutional layer based measurement block.
def measurement_convmmnet(input_image, scale):
    flatten = tf.keras.layers.Flatten()(input_image)
    measurement = tf.keras.layers.Dense(scale)(flatten)
    return measurement

# The initial reconstruction block maps the measurement inputs to an initial solution.
def initial_reconstruction_convmmnet(measurements, image_size):
    init_recon = tf.keras.layers.Dense(image_size**2)(measurements)
    init_recon = tf.keras.layers.Reshape(target_shape=(image_size, image_size, 1))(init_recon)
    return init_recon

# Deep reconstruction modules refine the coarser input image.
def reconstruction_block_convmmnet(input_layer, filter_size):
    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(filter_size, filter_size), 
                                  strides=1, padding="same",
                                  activation="relu", use_bias=True)(input_layer)
    conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(filter_size, filter_size), 
                                  strides=1, padding="same",
                                  activation="relu", use_bias=True)(conv1)
    conv3 = tf.keras.layers.Conv2D(filters=1, kernel_size=(filter_size, filter_size), 
                                  strides=1, padding="same",
                                  activation="relu", use_bias=True)(conv2)
    res = tf.keras.layers.Add()([input_layer, conv3])
    return res

# Base later of the multirate model.
def multirate_base_layer_convmmnet(measurements, image_size):
    pseudoimage = initial_reconstruction_convmmnet(measurements, image_size)
    refinement = reconstruction_block_convmmnet(pseudoimage, 5)
    for i in range(9):
        refinement = reconstruction_block_convmmnet(refinement, 5)
    return refinement

# Enhancement later of the multirate model.
def multirate_enhancement_layer_convmmnet(prev_image, measurements, image_size):
    residuals = initial_reconstruction_convmmnet(measurements, image_size)
    pseudoimage = tf.keras.layers.Add()([prev_image, residuals])
    refinement = reconstruction_block_convmmnet(pseudoimage, 5)
    for i in range(9):
        refinement = reconstruction_block_convmmnet(refinement, 5)
    return refinement

# Defining a CSNET-based multirate reconstruction model.
def multirate_model_convmmnet(im_size, feature_map, mes_sampling, channel):
    mes_size = [int(im_size**2*x) for x in mes_sampling];
    input_image = tf.keras.Input(shape=(im_size, im_size, 1));
    measurements = measurement_convmmnet(input_image, mes_size[-1]);
    base = multirate_base_layer_convmmnet(measurements[:mes_size[0]], im_size);
    en1 = multirate_enhancement_layer_convmmnet(base, measurements[mes_size[0]:mes_size[1]], im_size);
    en2 = multirate_enhancement_layer_convmmnet(en1, measurements[mes_size[1]:mes_size[2]], im_size);
    en3 = multirate_enhancement_layer_convmmnet(en2, measurements[mes_size[2]:mes_size[3]], im_size);
    en4 = multirate_enhancement_layer_convmmnet(en3, measurements[mes_size[3]:mes_size[4]], im_size);
    en5 = multirate_enhancement_layer_convmmnet(en4, measurements[mes_size[4]:mes_size[5]], im_size);
    en6 = multirate_enhancement_layer_convmmnet(en5, measurements[mes_size[5]:mes_size[6]], im_size);
    en7 = multirate_enhancement_layer_convmmnet(en6, measurements[mes_size[6]:mes_size[7]], im_size);
    en8 = multirate_enhancement_layer_convmmnet(en7, measurements[mes_size[7]:mes_size[8]], im_size);
    en9 = multirate_enhancement_layer_convmmnet(base, measurements[mes_size[8]:mes_size[9]], im_size);
    en10 = multirate_enhancement_layer_convmmnet(en1, measurements[mes_size[9]:mes_size[10]], im_size);
    en11 = multirate_enhancement_layer_convmmnet(en2, measurements[mes_size[10]:mes_size[11]], im_size);
    en12 = multirate_enhancement_layer_convmmnet(en3, measurements[mes_size[11]:mes_size[12]], im_size);
    en13 = multirate_enhancement_layer_convmmnet(en4, measurements[mes_size[12]:mes_size[13]], im_size);
    en14 = multirate_enhancement_layer_convmmnet(en5, measurements[mes_size[13]:mes_size[14]], im_size);
    en15 = multirate_enhancement_layer_convmmnet(en6, measurements[mes_size[14]:mes_size[15]], im_size);
    
    multirate_model = tf.keras.Model(inputs=input_image,
                                     outputs=[base, en1, en2, en3]);#, en4, en5, en6, en7,
                                              #en8, en9, en10, en11, en12, en13, en14, en15]);
    return multirate_model
