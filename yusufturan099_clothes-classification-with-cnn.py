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
import tensorflow as tf

from tensorflow.python.framework import ops

import matplotlib.pyplot as plt

train = pd.read_csv("../input/fashion/fashionmnist/fashion-mnist_train.csv")

test = pd.read_csv("../input/fashion/fashionmnist/fashion-mnist_test.csv")

train.head()
x_train = train.drop(["label"], axis=1).values.reshape(60000,28,28,1)

y_train = train[["label"]].values.reshape(-1,1)

x_test =  test.drop(["label"], axis=1).values.reshape(10000,28,28,1)   # LABELLARI AYIR ARRAYE DÖNÜŞTÜR 

y_test =  test[["label"]].values.reshape(-1,1)

def convert_to_one_hot(Y, C):

    Y = np.eye(C)[Y.reshape(-1)]

    return Y

y_train = convert_to_one_hot(y_train,10)

y_test = convert_to_one_hot(y_test,10)

print("Shape of x_train",x_train.shape)

print("Shape of y_train",y_train.shape)

print("Shape of x_test",x_test.shape)

print("Shape of y_test",y_test.shape)              #Shape of DATA

x_train = x_train/255

x_test = x_test/255  
plt.imshow(x_train[1,:,:].reshape(28,28))

print("class:",y_train[1,:])
def create_placeholders(n_h0,n_w0,n_c0,n_y0):

    X = tf.placeholder(tf.float32, shape= [None,n_h0,n_w0,n_c0])    

    Y = tf.placeholder(tf.float32, shape= [None, n_y0])

    return X,Y
def initialize_parameters():

    # lets make  filters W1 : [4, 4, 1, 8]  W2 : [2, 2, 8, 16]

    W1= tf.get_variable("W1", shape = [5, 5, 1, 8]  ,initializer= tf.contrib.layers.xavier_initializer() )   

    W2= tf.get_variable("W2", shape = [3, 3, 8, 16] ,initializer= tf.contrib.layers.xavier_initializer() )

    parameters = {"W1": W1,

                  "W2": W2}

    return parameters
def forward_propagation(X, parameters):

    Z1 = tf.nn.conv2d(X, parameters["W1"], strides= [1,2,2,1], padding="SAME"  )

    A1 = tf.nn.relu(Z1)

    P1 = tf.nn.max_pool( A1, ksize=[1,3,3,1], strides=[1,1,1,1], padding="SAME")

    Z2 = tf.nn.conv2d(P1, parameters["W2"], strides= [1,3,3,1], padding="SAME"  )

    A2 = tf.nn.relu(Z2)

    P2 = tf.nn.max_pool( A2, ksize=[1,2,2,1], strides=[1,1,1,1], padding="SAME")

    P2_flatten = tf.contrib.layers.flatten(P2)

    Z3 = tf.contrib.layers.fully_connected(P2_flatten,10, activation_fn=None)

    

    return Z3

    
def compute_cost(Z3,Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

    return cost
def random_mini_batches(X,Y, minibatch_size):

    m = X.shape[0]            # number of examples

    

    # Lets shuffle X and Y

    permutation = list(np.random.permutation(m))            # shuffled index of examples

    shuffled_X = X[permutation,:,:,:]

    shuffled_Y = Y[permutation,:]

    

    minibatches = []                                        # we will append all minibatch_Xs and minibatch_Ys to this minibatch list 

    number_of_minibatches = int(m/minibatch_size)           # number of mini batches 

    

    for k in range(number_of_minibatches):

        minibatch_X = shuffled_X[k*minibatch_size: (k+1)*minibatch_size,:,:,: ]

        minibatch_Y = shuffled_Y[k*minibatch_size: (k+1)*minibatch_size ,:]

        minibatch_pair = (minibatch_X , minibatch_Y)                        #tuple of minibatch_X and miinibatch_Y

        minibatches.append(minibatch_pair)

    if m%minibatch_size != 0 :

        last_minibatch_X = shuffled_X[number_of_minibatches*minibatch_size: m ,:,:,:]

        last_minibatch_Y = shuffled_Y[number_of_minibatches*minibatch_size: m ,:]

        last_minibatch_pair = (last_minibatch_X , last_minibatch_Y)

        minibatches.append(last_minibatch_pair)

    return minibatches
def model(X_train, Y_train, X_test, Y_test, learning_rate = 1,

          num_epochs = 150, minibatch_size = 64):

    ops.reset_default_graph()

    tf.set_random_seed(1)

    m,n_h0,n_w0,n_c0 = X_train.shape                          # (n_x: input size, m : number of examples in the train set)

    n_y0 = Y_train.shape[1]                            # n_y : output size

    costs = []                                        # To keep track of the cost

    

    X,Y = create_placeholders(n_h0,n_w0,n_c0,n_y0)



    parameters = initialize_parameters()

    zL = forward_propagation(X, parameters)



    cost = compute_cost(zL, Y)



    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



    init = tf.global_variables_initializer()

    minibatches = random_mini_batches(X_train,Y_train, minibatch_size)

    index = []

    with tf.Session() as sess:

        

        sess.run(init)

        

        for epoch in range(num_epochs):

            epoch_cost = 0

            num_minibatches = int(m / minibatch_size)

            for minibatch in minibatches:

                minibatch_X , minibatch_Y = minibatch

                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost/num_minibatches

                       

            if epoch % 10 == 0:

                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))  # Print the cost every epoch

            index.append(epoch)

            costs.append(epoch_cost)

                

        # plot the cost

        plt.plot(index , costs)

        plt.ylabel('cost')

        plt.xlabel('iterations (per tens)')

        plt.title("Learning rate =" + str(learning_rate))

        plt.show()



        # lets save the parameters in a variable

        parameters = sess.run(parameters)

        print("Parameters have been trained!")



        # Calculate the correct predictions

        predict_op = tf.argmax(zL, 1)

        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        

        # Calculate accuracy on the test set

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print(accuracy)

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})

        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

        print("Train Accuracy:", train_accuracy)

        print("Test Accuracy:", test_accuracy)

        

        
model(x_train, y_train, x_test, y_test, learning_rate = 0.005,

          num_epochs = 400, minibatch_size = 1024)