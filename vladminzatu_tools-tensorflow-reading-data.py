# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
dataset = tf.data.Dataset.from_tensors((tf.random_uniform([4]), tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset.output_types)  # the types of elements in a dataset
print(dataset.output_shapes)  # the shapes of elements in a dataset
dataset = tf.data.Dataset.from_tensors({
    "a": tf.random_uniform([4]), 
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset.output_types)  # the types of elements in a dataset
print(dataset.output_shapes)  # the shapes of elements in a dataset
dataset = tf.data.Dataset.from_tensor_slices({
    "a": tf.random_uniform([4]), 
    "b": tf.random_uniform([4, 100],maxval=100, dtype=tf.int32)})
print(dataset.output_types)  # the types of elements in a dataset
print(dataset.output_shapes)  # the shapes of elements in a dataset
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    for i in range(4):
        print(sess.run(next_element))
# We have 3 training examples
features = np.array([[1,2,3],[4,5,6],[7,8,9]])
labels = np.array([1,0,1])

dataset = tf.data.Dataset.from_tensor_slices((features, labels))
print(dataset.output_types)
print(dataset.output_shapes)
features = np.array([[1,2,3],[4,5,6],[7,8,9]])
labels = np.array([1,0,1])

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
iterator = dataset.make_initializable_iterator()
with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})
filenames = ["../input/googleplaystore.csv"]
record_defaults = [tf.string] * 13
dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    print(sess.run(next_element))
