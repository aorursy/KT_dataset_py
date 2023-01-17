# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import tensorflow as tf

# Any results you write to the current directory are saved as output.
data = pd.DataFrame(pd.read_csv("../input/car.data.csv"))
data = pd.get_dummies(data, prefix=data.columns)
columns = list(data.keys())
X_train, X_test, y_train, y_test = train_test_split(data[columns[0:-4]].values,
                                                    data[columns[-4:]].values.reshape(-1, 4))

learning_rate = 0.01
n_train_samples = X_train.shape[0]
n_test_samples = y_test.shape[0]
batch_size = 10
n_features = 6
n_classes = 4

x = tf.placeholder(tf.float32, shape=[None, 21], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 4], name='y_')
l1 = tf.layers.dense(x, 128, tf.nn.relu, name="l1")
l2 = tf.layers.dense(l1, 128, tf.nn.relu, name="l2")
out = tf.layers.dense(l2, 4, name="l3")
with tf.name_scope('loss'):
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=out)
    loss = tf.reduce_mean(cross_entropy, name='loss')

with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

with tf.name_scope('eval'):
    correct_prediction = tf.metrics.accuracy(predictions=tf.argmax(out, axis=1), labels=tf.argmax(y_, axis=1))[1]
with tf.Session() as sess:
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    batch_index = np.random.randint(n_train_samples, size=10)
    batch_test_index = np.random.randint(n_test_samples, size=10)
    accuracy_l = []
    for epoch in range(2000):
        _, c, pred = sess.run([optimizer, cross_entropy, correct_prediction], feed_dict={
            x: X_train[batch_index],
            y_: y_train[batch_index]
        })

        ave_c = c
        print("epoch=", epoch, "loss=", ave_c)
        if epoch % 50 == 0:
            pred = correct_prediction.eval(feed_dict={
                x: X_test[batch_test_index],
                y_: y_test[batch_test_index]})
            print("pred=", pred)
            accuracy_l.append(pred)
    print('test accuracy %g' % np.mean(accuracy_l))