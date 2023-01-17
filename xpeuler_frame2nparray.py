import json

import os

import time

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from IPython.display import YouTubeVideo

from subprocess import check_output



import tensorflow as tf

import tensorflow.contrib.slim as slim

from tensorflow import app

from tensorflow import flags

from tensorflow import gfile

from tensorflow import logging



#%% source starter code. Place this notebook in the same directory of starter code.

#import readers

#import utils

# use interactive session

sess = tf.InteractiveSession()

LABELs = pd.read_csv('../input/label_names.csv') #4716 labels
# helper functions.

# modified from sarter code. using readers.py and utils.py.



# from utils.py

def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):

  """Dequantize the feature from the byte format to the float format.



  Args:

    feat_vector: the input 1-d vector.

    max_quantized_value: the maximum of the quantized value.

    min_quantized_value: the minimum of the quantized value.



  Returns:

    A float vector which has the same shape as feat_vector.

  """

  assert max_quantized_value > min_quantized_value

  quantized_range = max_quantized_value - min_quantized_value

  scalar = quantized_range / 255.0

  bias = (quantized_range / 512.0) + min_quantized_value

  return feat_vector * scalar + bias



# from readers.py

def resize_axis(tensor, axis, new_size, fill_value=0):

  """Truncates or pads a tensor to new_size on on a given axis.



  Truncate or extend tensor such that tensor.shape[axis] == new_size. If the

  size increases, the padding will be performed at the end, using fill_value.



  Args:

    tensor: The tensor to be resized.

    axis: An integer representing the dimension to be sliced.

    new_size: An integer or 0d tensor representing the new value for

      tensor.shape[axis].

    fill_value: Value to use to fill any new entries in the tensor. Will be

      cast to the type of tensor.



  Returns:

    The resized tensor.

  """

  tensor = tf.convert_to_tensor(tensor)

  shape = tf.unstack(tf.shape(tensor))



  pad_shape = shape[:]

  pad_shape[axis] = tf.maximum(0, new_size - shape[axis])



  shape[axis] = tf.minimum(shape[axis], new_size)

  shape = tf.stack(shape)



  resized = tf.concat([

      tf.slice(tensor, tf.zeros_like(shape), shape),

      tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))

  ], axis)



  # Update shape.

  new_shape = tensor.get_shape().as_list()  # A copy is being made.

  new_shape[axis] = new_size

  resized.set_shape(new_shape)

  return resized



def get_video_matrix(features,

                     feature_size,

                     max_frames,

                     max_quantized_value,

                     min_quantized_value):

    """Decodes features from an input string and quantizes it.



    Args:

      features: raw feature values

      feature_size: length of each frame feature vector

      max_frames: number of frames (rows) in the output feature_matrix

      max_quantized_value: the maximum of the quantized value.

      min_quantized_value: the minimum of the quantized value.



    Returns:

      feature_matrix: matrix of all frame-features

      num_frames: number of frames in the sequence

    """

    decoded_features = tf.reshape(

        tf.cast(tf.decode_raw(features, tf.uint8), tf.float32),

        [-1, feature_size])



    num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)

    feature_matrix = Dequantize(decoded_features,

                                      max_quantized_value,

                                      min_quantized_value)

    feature_matrix = resize_axis(feature_matrix, 0, max_frames)

    return feature_matrix, num_frames
# consts

num_features=2

feature_names, feature_sizes = (['rgb','audio'], [1024,128])

max_frames=300

max_quantized_value=2

min_quantized_value=-2
# working with local files. change path to your own.

vidfilenames = "../input/frame_level/train-2.tfrecord"

examples = tf.python_io.tf_record_iterator(vidfilenames)
aa = next(examples)

print(len(aa))

print(type(aa))

bb = tf.decode_raw(aa,tf.uint8)

cc = sess.run([bb])

cc[0].shape
# from reader. Important to set context_features and sequance_features correctly

contexts, features = tf.parse_single_sequence_example(

        aa,

        context_features={"video_id": tf.FixedLenFeature(

            [], tf.string),

                          "labels": tf.VarLenFeature(tf.int64)},

        sequence_features={

            feature_name : tf.FixedLenSequenceFeature([], dtype=tf.string)

            for feature_name in ['rgb','audio']

        })



labels = (tf.cast(

        tf.sparse_to_dense(contexts["labels"].values, (4716,), 1,

            validate_indices=False),

        tf.bool))
num_frames = -1  # the number of frames in the video

feature_matrices = [None] * num_features  # an array of different features

for feature_index in range(num_features):

    feature_matrix, num_frames_in_this_feature = get_video_matrix(

        features[feature_names[feature_index]],

        feature_sizes[feature_index],

        max_frames,

        max_quantized_value,

        min_quantized_value)

    if num_frames == -1:

        num_frames = num_frames_in_this_feature

    else:

        tf.assert_equal(num_frames, num_frames_in_this_feature)



    feature_matrices[feature_index] = feature_matrix



# cap the number of frames at self.max_frames

num_frames = tf.minimum(num_frames, max_frames)



# concatenate different features

video_matrix = tf.concat(feature_matrices, 1)



# convert to batch format.

# TODO: Do proper batch reads to remove the IO bottleneck.

batch_video_ids = tf.expand_dims(contexts["video_id"], 0)

batch_video_matrix = tf.expand_dims(video_matrix, 0)

batch_labels = tf.expand_dims(labels, 0)

batch_frames = tf.expand_dims(num_frames, 0)
# sess.run to get data in numpy array.

[x1, x2, x3, x4] = sess.run([num_frames, video_matrix, batch_labels, batch_frames])

[z1, z2] = sess.run([labels, num_frames])

vid_byte = sess.run(batch_video_ids)

vid=vid_byte[0].decode()

print('vid = %s'%vid)
print(vid, x1)

print(LABELs[x3.T])

print(x2.shape)

print(contexts["labels"].values)

plt.figure(figsize=[12,8])

for i in range(4):

    k = int( (i+1)*2/10 * x1 )

    plt.subplot(2,4, i+1)

    plt.imshow(np.array(x2[k,:1024]).reshape([32,32]));

    plt.subplot(2,4, i+5)

    plt.imshow(np.array(x2[k,1024:]).reshape([16,8]));
# will run on local machine, not on kagglegym.

YouTubeVideo(vid)
sess.close()