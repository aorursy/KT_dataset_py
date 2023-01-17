import os
import sys
import numpy as np
import tensorflow as tf
import random
import math
import warnings
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

print(os.listdir("../input"))
# Set some parameters
IMG_SIZE = 101

IMG_CHANNELS = 3

path_train = '../input/train'

path_test = '../input/test'
train_ids = next(os.walk(path_train + "/images"))[2]
test_ids = next(os.walk(path_test + "/images"))[2]
# Get and resize train images and masks
train_images = np.zeros((len(train_ids), IMG_SIZE, IMG_SIZE, IMG_CHANNELS), dtype=np.float32)
train_labels = np.zeros((len(train_ids), IMG_SIZE, IMG_SIZE, 1), dtype=np.bool)

test_images = np.zeros((len(test_ids), IMG_SIZE, IMG_SIZE, IMG_CHANNELS), dtype=np.float32)
print('Getting and resizing train images and masks without padding ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    img = imread(path_train + "/images/" + id_)[:,:,:IMG_CHANNELS]
    train_images[n] = img
    
    mask = imread(path_train + "/masks/" + id_)
    mask = np.expand_dims(mask, axis = -1)

    train_labels[n] = mask
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img = imread(path_test + "/images/" + id_)[:,:,:IMG_CHANNELS]
    test_images[n] = img
def deconv2d(input_tensor, filter_size, output_size, out_channels, in_channels, name, strides = [1, 1, 1, 1]):
    dyn_input_shape = tf.shape(input_tensor)
    batch_size = dyn_input_shape[0]
    out_shape = tf.stack([batch_size, output_size, output_size, out_channels])
    filter_shape = [filter_size, filter_size, out_channels, in_channels]
    w = tf.get_variable(name=name, shape=filter_shape)
    return tf.nn.conv2d_transpose(input_tensor, w, out_shape, strides, padding='VALID')
def conv2d(input_tensor, depth, kernel, name, strides=(1, 1), padding="VALID"):
    return tf.layers.conv2d(input_tensor, filters=depth, kernel_size=kernel, strides=strides, padding=padding, activation=None, name=name)
def cnn_model_fn(features, labels, mode):
    
    """CNN with five conv layers, and six transpose conv layers."""
    net = conv2d(features, 32, 1, "Y0") #101
    net = tf.nn.relu(net)

    net = conv2d(net, 40, 3, "Y1", strides=(2, 2)) #50
    net = tf.nn.relu(net)

    net = conv2d(net, 48, 2, "Y2", strides=(2, 2)) #25
    net = tf.nn.relu(net)

    net = conv2d(net, 64, 3, "Y3", strides=(2, 2)) #12
    net = tf.nn.relu(net)

    net = conv2d(net, 64, 2, "Y4") #11
    net = tf.nn.relu(net)


    net = deconv2d(net, 1, 11, 64, 64, "Y5_deconv") #11
    net = tf.nn.relu(net)

    net = deconv2d(net, 2, 12, 64, 64, "Y4_deconv") #12
    net = tf.nn.relu(net)

    net = deconv2d(net, 3, 25, 48, 64, "Y3_deconv", strides=[1, 2, 2, 1]) #25
    net = tf.nn.relu(net)

    net = deconv2d(net, 2, 50, 40, 48, "Y2_deconv", strides=[1, 2, 2, 1]) #50
    net = tf.nn.relu(net)

    net = deconv2d(net, 3, 101, 32, 40, "Y1_deconv", strides=[1, 2, 2, 1]) #101

    logits = deconv2d(net, 1, 101, 1, 32, "Y0_deconv") #101

    
    predictions = {
        'logits': logits
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
# Create the Estimator
cnn_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/convnet_40_0005_model")
# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=train_images,
    y=train_labels,
    batch_size=40,
    num_epochs=None,
    shuffle=True)
cnn_classifier.train(
    input_fn=train_input_fn,
    steps=15000)
# Predict input fuction
pred_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=test_images,
    batch_size=1,
    shuffle=False)
pred_result = cnn_classifier.predict(input_fn=pred_input_fn)
predicted_mask = next(pred_result)["logits"]
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
predicted_mask = np.reshape(np.squeeze(predicted_mask), [IMG_SIZE , IMG_SIZE, 1])
for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
                predicted_mask[i][j] = int(sigmoid(predicted_mask[i][j])*255)
imshow(predicted_mask.astype(np.uint8).squeeze())