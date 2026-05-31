# Importing necessary libraries
import cv2
import numpy as np
import tensorflow as tf
import os

# Initializing constants
EXPORT_ADDRESS = ".\\datasets"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
IMAGE_SIZE = 32

# Turning RGB images to Grayscale images
def rgb_to_gray(images):
    gray_images = []
    for img in images:
        # Converting RGB image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_images.append(img)
    return gray_images

# Retrieving and calling the dataset.
cifar10 = tf.keras.datasets.cifar10.load_data()

training_set = cifar10[0]
training_images =  np.asarray(rgb_to_gray(training_set[0]))
training_labels = training_set[1]

test_set = cifar10[1]
test_images = np.asarray(rgb_to_gray(test_set[0]))
test_labels = test_set[1]

# Creating the validation set as dummy objects.
val_images = np.zeros(shape=(1,IMAGE_SIZE,IMAGE_SIZE)); train_images = np.zeros(shape=(1,IMAGE_SIZE,IMAGE_SIZE))
val_labels = np.zeros(shape=(1,1)); train_labels = np.zeros(shape=(1,1))

# Dividing the training.
for i in range(10):
    val_set = np.where(training_labels==i)[0]
    np.random.shuffle(val_set)
    images = np.asarray(rgb_to_gray(training_images[val_set]))
    labels = training_labels[val_set]
    val_images = np.concatenate((val_images, images[:int(VAL_RATIO*len(val_set))]), axis=0)
    val_labels = np.concatenate((val_labels, labels[:int(VAL_RATIO*len(val_set))]), axis=0)
    train_images = np.concatenate((train_images, images[int(VAL_RATIO*len(val_set)):]), axis=0)
    train_labels = np.concatenate((train_labels, labels[int(VAL_RATIO*len(val_set)):]), axis=0)
val_images = val_images[1:]; val_labels = val_labels[1:]
train_images = train_images[1:]; train_labels = train_labels[1:]

# Saving the dataset and the measurement matrix.
if not os.path.exists(EXPORT_ADDRESS):
    os.makedirs(EXPORT_ADDRESS)
np.save(os.path.join(EXPORT_ADDRESS,"train_images"), train_images)
np.save(os.path.join(EXPORT_ADDRESS,"train_labels"), train_labels)
np.save(os.path.join(EXPORT_ADDRESS,"test_images"), test_images)
np.save(os.path.join(EXPORT_ADDRESS,"test_labels"), test_labels)
np.save(os.path.join(EXPORT_ADDRESS,"val_images"), val_images)
np.save(os.path.join(EXPORT_ADDRESS,"val_labels"), val_labels)

# Displaying the image.
index = 1
image = training_images[index]
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image_res = cv2.resize(image_bgr, (6*image_bgr.shape[1], 6*image_bgr.shape[0]), interpolation=cv2.INTER_AREA)
cv2.imshow("image", image_res)
cv2.show()
