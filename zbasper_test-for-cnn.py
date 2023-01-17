# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from sklearn.datasets import load_digits

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import OneHotEncoder

import numpy as np

import tensorflow as tf

import NLTK
digits = load_digits()

X_data = digits.data.astype(np.float32)

Y_data = digits.target.astype(np.float32).reshape(-1,1)

scaler = MinMaxScaler()

X_data = scaler.fit_transform(X_data)

Y = OneHotEncoder().fit_transform(Y_data).todense()

X = X_data.reshape(-1,8,8,1)

batch_size = 8
def generatebatch(X, Y, n_examples, batch_size):

    for batch_i in range(n_examples // batch_size):

        start = batch_i * batch_size

        end = start + batch_size

        batch_xs = X[start:end]

        batch_ys = Y[start:end]

        yield batch_xs, batch_ys

tf.reset_default_graph()

tf_X = tf.placeholder(tf.float32, [None, 8, 8, 1])

tf_Y = tf.placeholder(tf.float32, [None, 10])

conv_filter_w1 = tf.Variable(tf.random_normal([3, 3, 1, 10]))

conv_filter_b1 = tf.Variable(tf.random_normal([10]))

relu_feature_maps1 = tf.nn.relu(tf.nn.conv2d(tf_X, conv_filter_w1, strides=[1, 1, 1, 1],

                                            padding='SAME') + conv_filter_b1)

max_pool1 = tf.nn.max_pool(relu_feature_maps1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],

                           padding='SAME')

print(max_pool1)
conv_filter_w2 = tf.Variable(tf.random_normal([3, 3, 10, 5]))

conv_filter_b2 = tf.Variable(tf.random_normal([5]))

conv_out2 = tf.nn.conv2d(relu_feature_maps1, conv_filter_w2, strides=[1, 2, 2, 1],

                        padding='SAME') + conv_filter_b2

print(conv_out2)
batch_mean, batch_var = tf.nn.moments(conv_out2, [0, 1, 2], keep_dims=True)

shift = tf.Variable(tf.zeros([5]))

scale = tf.Variable(tf.ones([5]))

epsilon = 1e-3

BN_out = tf.nn.batch_normalization(conv_out2, batch_mean, batch_var, shift, scale, epsilon)

print(BN_out)

relu_BN_maps2 = tf.nn.relu(BN_out)

max_pool2 = tf.nn.max_pool(relu_BN_maps2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],

                          padding='SAME')

print(max_pool2)
max_pool2_flat = tf.reshape(max_pool2, [-1, 2*2*5])

fc_w1 = tf.Variable(tf.random_normal([2*2*5, 50]))

fc_b1 = tf.Variable(tf.random_normal([50]))

fc_out1 = tf.nn.relu(tf.matmul(max_pool2_flat, fc_w1) + fc_b1)

out_w1 = tf.Variable(tf.random_normal([50, 10]))

out_b1 = tf.Variable(tf.random_normal([10]))

pred = tf.nn.softmax(tf.matmul(fc_out1, out_w1) + out_b1)

loss = -tf.reduce_mean(tf_Y*tf.log(tf.clip_by_value(pred, 1e-11, 1.0)))

train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

y_pred = tf.arg_max(pred, 1)

bool_pred = tf.equal(tf.arg_max(tf_Y, 1), y_pred)

accuracy = tf.reduce_mean(tf.cast(bool_pred, tf.float32))



with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for epoch in range(1000):

        for batch_xs, batch_ys in generatebatch(X, Y, Y.shape[0], batch_size):

            sess.run(train_step, feed_dict={tf_X: batch_xs, tf_Y: batch_ys})

        if (epoch % 100 == 0):

            res = sess.run(accuracy, feed_dict={tf_X: X, tf_Y: Y})

            print(epoch, res)

    res_ypred = y_pred.eval(feed_dict={tf_X: X, tf_Y: Y}).flatten()

    print(res_ypred)
from sklearn.metrics import accuracy_score

print(accuracy_score(Y_data, res_ypred.reshape(-1,1)))