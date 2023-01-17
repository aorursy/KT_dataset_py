# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from tensorflow.python.framework import ops

import math

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_X,train_y = pd.read_csv('../input/train.csv').iloc[:,1:].values,pd.read_csv('../input/train.csv').label.values

test_X = pd.read_csv('../input/test.csv').values
train_X1 = train_X.reshape((train_X.shape[0],28,28,1))

test_X1 = test_X.reshape((test_X.shape[0],28,28,1))
train_X1.shape
# Encoding labels Y:

def one_hot_matrix(labels, C):

    

    C = tf.constant(C,name='C')

    

    one_hot_matrix = tf.one_hot(labels,depth=C)

    

    sess = tf.Session()

    

    one_hot = sess.run(one_hot_matrix)

    

    sess.close()

        

    return one_hot
train_y1 = one_hot_matrix(train_y,10).T

train_y1[:,:10]
def initialize_parameters():

    weights = {

    'wc1': tf.Variable(tf.random_normal([5,5,1,32])),

    'wc2': tf.Variable(tf.random_normal([5,5,32,64])),

    'wd1': tf.Variable(tf.random_normal([7*7*64,1024])),

    'out': tf.Variable(tf.random_normal([1024,10]))

    }



    biases = {

    'bc1': tf.Variable(tf.random_normal([32])),

    'bc2': tf.Variable(tf.random_normal([64])),

    'bd1': tf.Variable(tf.random_normal([1024])),

    'out': tf.Variable(tf.random_normal([10]))

    }

    return weights,biases
def conv2d(X,W,b,s=1):

    out = tf.nn.conv2d(X,W,[1,s,s,1],padding='SAME')

    out = tf.add(out,b)

    return tf.nn.relu(out)
def maxpool(X,k=2):

    return tf.nn.max_pool(X,ksize = [1,k,k,1], strides = [1,k,k,1], padding = 'SAME')
def create_placeholders(n_x,n_y,n_c):

    X = tf.placeholder(dtype=tf.float32,shape=[None,n_x,n_x,n_c])

    Y = tf.placeholder(dtype=tf.float32,shape=[None,n_y])

    return X,Y
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    m = X.shape[0]                  # number of training examples

    mini_batches = []

    np.random.seed(seed)

    

    # Step 1: Shuffle (X, Y)

    permutation = list(np.random.permutation(m))

    shuffled_X = X[permutation]

    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))



    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning

    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size]

        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y.T)

        mini_batches.append(mini_batch)

    

    # Handling the end case (last mini-batch < mini_batch_size)

    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m]

        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]

        mini_batch = (mini_batch_X, mini_batch_Y.T)

        mini_batches.append(mini_batch)

    

    return mini_batches
def model(X_train, Y_train,learning_rate = 0.001,

          num_epochs = 1300, minibatch_size = 512, print_cost = True,keep_prob=1.0):

    costs=[]

    m = X_train.shape[0]

    n_x = X_train.shape[1]

    n_y=10

    ops.reset_default_graph()    

    weights,biases = initialize_parameters()

    (X,Y) = create_placeholders(n_x,n_y,1)

    A1 = conv2d(X,weights['wc1'],biases['bc1'])

    A1 = maxpool(A1,k=2)

    A2 = conv2d(A1,weights['wc2'],biases['bc2'])

    A2 = maxpool(A2,k=2)

    fc1 = tf.reshape(A2,[-1, weights['wd1'].get_shape().as_list()[0]])

    fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])

    fc1 = tf.nn.relu(fc1)

    fc1 = tf.nn.dropout(fc1,keep_prob)

    logits = tf.add(tf.matmul(fc1,weights['out']),biases['out'])

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()



    # Start the session to compute the tensorflow graph

    with tf.Session() as sess:

        

        # Run the initialization

        sess.run(init)

        for epoch in range(num_epochs):

            epoch_cost = 0.

            num_minibatches = int(m / minibatch_size)

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:

                

                minibatch_X,minibatch_Y = minibatch

                _ , minibatch_cost = sess.run([optimizer,cost],feed_dict={X:X_train,Y:Y_train.T})

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch

            if print_cost == True and epoch % 100 == 0:

                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

            if print_cost == True and epoch % 5 == 0:

                costs.append(epoch_cost)

        # plot the cost

        plt.plot(np.squeeze(costs))

        plt.ylabel('cost')

        plt.xlabel('iterations (per tens)')

        plt.title("Learning rate =" + str(learning_rate))

        plt.show()



        # lets save the parameters in a variable

        weights,biases = sess.run((weights,biases))

        print ("Parameters have been trained!")



        # Calculate the correct predictions

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))



        # Calculate accuracy on the test set

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train.T}))

        #print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test.T}))

        

    return weights,biases    

    
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_X1, train_y1.T, test_size=0.35, random_state=42)
y_train,y_test = y_train.T,y_test.T

x_train.shape,x_test.shape,y_train.shape,y_test.shape
weights,biases = model(train_X1,train_y1,minibatch_size=2**14,num_epochs=2000,keep_prob=0.77)
def forward_propagation_for_predict(X, weights,biases):

    

    # Retrieve the parameters from the dictionary "parameters" 

    A1 = conv2d(X,weights['wc1'],biases['bc1'])

    A1 = maxpool(A1,k=2)

    A2 = conv2d(A1,weights['wc2'],biases['bc2'])

    A2 = maxpool(A2,k=2)

    fc1 = tf.reshape(A2,[-1, weights['wd1'].get_shape().as_list()[0]])

    fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])

    fc1 = tf.nn.relu(fc1)

    z4 = tf.add(tf.matmul(fc1,weights['out']),biases['out'])

    

    return z4
def predict(X, weights,biases):

    

    wc1 = tf.convert_to_tensor(weights['wc1'])

    bc1 = tf.convert_to_tensor(biases["bc1"])

    wc2 = tf.convert_to_tensor(weights['wc2'])

    bc2 = tf.convert_to_tensor(biases["bc2"])

    wd1= tf.convert_to_tensor(weights['wd1'])

    bd1 = tf.convert_to_tensor(biases["bd1"])

    wout = tf.convert_to_tensor(weights['out'])

    bout = tf.convert_to_tensor(biases["out"])

    

    weights = {

    'wc1': wc1,

    'wc2': wc2,

    'wd1': wd1,

    'out': wout

    }



    biases = {

    'bc1': bc1,

    'bc2': bc2,

    'bd1': bd1,

    'out': bout

    }

    x = tf.placeholder("float", [None, 28,28,1])

    

    z4 = forward_propagation_for_predict(x, weights,biases)

    p = tf.argmax(z4,1)

    

    sess = tf.Session()

    prediction = sess.run(p, feed_dict = {x: X})

        

    return prediction
predictions = predict(test_X1,weights,biases)
sub = pd.read_csv('../input/sample_submission.csv')

sub['Label']=predictions

sub.to_csv('submission.csv',index=False)
train_X
