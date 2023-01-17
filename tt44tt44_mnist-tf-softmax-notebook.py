# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
import tempfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

from matplotlib import pyplot as plt 

import pickle
import gzip

def load_data():
    with gzip.open('../input/mnist.pkl.gz') as fp:
        training_data, valid_data, test_data = pickle.load(fp, encoding='bytes')
    return training_data, valid_data, test_data

training_data_0, valid_data_0, test_data_0 = load_data()

train_data = [training_data_0[0]]
valid_data = [valid_data_0[0]]
test_data = [test_data_0[0]]

from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit([[i] for i in range(10)])
train_data.append(enc.transform([[training_data_0[1][i]] for i in range(len(training_data_0[1]))]).toarray())
valid_data.append(enc.transform([[valid_data_0[1][i]] for i in range(len(valid_data_0[1]))]).toarray())
test_data.append(enc.transform([[test_data_0[1][i]] for i in range(len(test_data_0[1]))]).toarray())
train_data[0].shape
# 进行模型搭建

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

y_hat = tf.placeholder("float", [None,10])

cross_entropy = -tf.reduce_sum(y_hat * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 运行模型
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_hat,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

result = []

import math

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))
    
    #return shuffled_X[0:mini_batch_size, :], shuffled_Y[0:mini_batch_size, :]
    
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

seed = 3
epoch = 100

for i in range(epoch):
    seed += 1
    mini_batches = random_mini_batches(train_data[0], train_data[1], 32, seed)
    for j in range(len(mini_batches)):
        batch_xs, batch_ys = mini_batches[j][0], mini_batches[j][1]
        sess.run(train_step, feed_dict = {x: batch_xs, y_hat:batch_ys})
    epoch_result = sess.run(accuracy, feed_dict={x: test_data[0], y_hat: test_data[1]})
    result.append(epoch_result)

print(result)
plt.plot(result)
