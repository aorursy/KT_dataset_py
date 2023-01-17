# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing MNIST data 

import tensorflow as tf

import numpy as np

from tensorflow.examples.tutorials.mnist import input_data 



mnist = input_data.read_data_sets('../input', one_hot=True)



# Learning rate, batch size and number of epochs

alpha = 0.0375

batch_size = 60

epochs = 15



# Placeholders for X and Y training 

X = tf.placeholder(tf.float32, [None, 784])

Y = tf.placeholder(tf.float32, [None, 10])





## FORWARD PROPOGATION

# Weights and Biases for hidden layers (5 hidden layers with 1024 hidden units). Using He initialization for W

W1 = tf.Variable(tf.random_normal([784,1024],stddev=1)*np.sqrt(2/784), name='w1') 

b1 = tf.Variable(tf.zeros([1024]), trainable=True, name="b1")



W2 = tf.Variable(tf.random_normal([1024,1024],stddev=1)*np.sqrt(2/1024), name='w2') 

b2 = tf.Variable(tf.zeros([1024]), trainable=True, name="b2")



W3 = tf.Variable(tf.random_normal([1024,1024],stddev=1)*np.sqrt(2/1024), name='w3') 

b3 = tf.Variable(tf.zeros([1024]), trainable=True, name="b3")



Wl = tf.Variable(tf.random_normal([1024,10],stddev=1)*np.sqrt(2/1024), name='wl') 

bl = tf.Variable(tf.zeros([10]), trainable=True, name="bl")



# Outputs of hidden layers

z1 = tf.add(tf.matmul(X,W1), b1)

a1 = tf.nn.relu(z1)



z2 = tf.add(tf.matmul(a1,W2), b2)

a2 = tf.nn.relu(z2)



z3 = tf.add(tf.matmul(a2,W3), b3)

a3 = tf.nn.relu(z3)



zl = tf.add(tf.matmul(a3,Wl), bl)

al = tf.nn.softmax(zl)



y = tf.clip_by_value(al, -0.00000001, 0.9999999)



# Cost as cross entropy and optimizer

cost = -tf.reduce_mean(tf.reduce_sum(Y * tf.log(y) + (1 - Y) * tf.log(1 - y), axis=1))



optimiser = optimiser = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(cost)





_init = tf.global_variables_initializer()



# Accuracy

predictions = tf.equal(tf.argmax(Y, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))



cost_dict = {}



with tf.Session() as sess:

    

    # Initialize variables

    

    sess.run(_init)

    m = len(mnist.train.labels)

    num_batches = int(m/batch_size)

    

    print("Number of training samples : ", m)

    print("Number of test samples : ", len(mnist.test.images))

    print("Number of Epochs : ", epochs)

    print("Number of batches : ", num_batches)

    

    # iterate through epochs

    for epoch in range(epochs):

        cost_ = 0.0

        

        # iterate through 

        for batch in range(num_batches):

            X_batch, Y_batch = mnist.train.next_batch(batch_size=batch_size)

            _,curr_cost = sess.run([optimiser, cost], feed_dict={X: X_batch, Y: Y_batch})

            cost_ = cost_ + curr_cost/num_batches

        cost_dict[epoch] = cost_

        print("Epoch : ", epoch + 1, "\tCost : ", cost_)

    

    X_test = mnist.test.images

    y_test = mnist.test.labels

    accuracy_ = sess.run(accuracy, feed_dict={X : X_test, Y : y_test})

    

    print("Accuracy on test set : " + str(accuracy_ * 100) + '%')
import matplotlib.pylab as plt



x, y = cost_dict.keys(),cost_dict.values()



plt.plot(x, y)

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.show()