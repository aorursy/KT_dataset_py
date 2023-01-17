# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import tensorflow as tf

import math

# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/fashion/fashionmnist/fashion-mnist_train.csv")

test=pd.read_csv("../input/fashion/fashionmnist/fashion-mnist_test.csv")



train.head()             #datayı incele
train.describe()                     #datayı incele
train.info()                     #datayı incele
train = train[train["label"]==1].append(train[train["label"]==2]).append(train[train["label"]==3]).append(train[train["label"]==4]).append(train[train["label"]==5])

test = test[test["label"]==1].append(test[test["label"]==2]).append(test[test["label"]==3]).append(test[test["label"]==4]).append(test[test["label"]==5])                                                                                                                                        

x_train = train.drop(["label"], axis=1).values.T

y_train = train[["label"]].values.reshape(-1,1).T

x_test =  test.drop(["label"], axis=1).values.T    # LABELLARI AYIR ARRAYE DÖNÜŞTÜR 

y_test =  test[["label"]].values.reshape(-1,1).T

def convert_to_one_hot(Y, C):

    Y = np.eye(C)[Y.reshape(-1)].T

    return Y

y_train = convert_to_one_hot(y_train,10)

y_test = convert_to_one_hot(y_test,10)

print("Shape of x_train",x_train.shape)

print("Shape of y_train",y_train.shape)

print("Shape of x_test",x_test.shape)

print("Shape of y_test",y_test.shape)              #Shape of DATA

x_train = x_train/255

x_test = x_test/255                     # NORMALIZE DATA



def create_placeholders(n_x, n_y):

    X = tf.placeholder(tf.float32, [n_x, None], name="X")

    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")

    

    return X, Y
def initialize_params(layer_dims):

    

    parameters={}

    for l in range(1,len(layer_dims)):

        parameters["w" + str(l)] = tf.get_variable("w"+str(l), [layer_dims[l],layer_dims[l-1]], initializer = tf.contrib.layers.xavier_initializer())   

        parameters["b" + str(l)] = tf.get_variable("b"+str(l), [layer_dims[l],1], initializer = tf.zeros_initializer())   

    return parameters    

   
def forward_prop(x, parameters):

    z_a_deposu = {}

    z_a_deposu["A0"] = x

    for l in range(1,int(len(parameters)/2)+1):

        z_a_deposu["z" + str(l)] =  tf.add(tf.matmul(parameters["w" + str(l)],z_a_deposu["A" + str(l-1)] ), parameters["b" + str(l)])        

        if l == int(len(parameters)/2):

            break

        z_a_deposu["A" + str(l)] = tf.nn.relu(z_a_deposu["z" + str(l)])

    zL =z_a_deposu["z" + str(l)]

    return zL
def compute_cost(zL, y,parameters):

    

    logits = tf.transpose(zL)

    labels = tf.transpose(y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)) 

    

    #L2 REGULARIZATION

    num_parameters=len(parameters)

    L =int(num_parameters/2)

    reg = 0

    for l in range(L):

        reg += tf.nn.l2_loss(parameters["w"+ str(l+1)])

    LAMBDA = 0.01

    cost = tf.reduce_mean(cost + LAMBDA*reg)

    return cost
def random_mini_batches(X,Y, minibatch_size):

    m = Y.shape[1]            # number of examples

    

    # Lets shuffle X and Y

    permutation = list(np.random.permutation(m))            # shuffled index of examples

    shuffled_X = X[:, permutation]

    shuffled_Y = Y[:, permutation]

    

    minibatches = []                                        # we will append all minibatch_Xs and minibatch_Ys to this minibatch list 

    number_of_minibatches = int(m/minibatch_size)           # number of mini batches 

    

    for k in range(number_of_minibatches):

        minibatch_X = shuffled_X[:,k*minibatch_size: (k+1)*minibatch_size ]

        minibatch_Y = shuffled_Y[:,k*minibatch_size: (k+1)*minibatch_size ]

        minibatch_pair = (minibatch_X , minibatch_Y)                        #tuple of minibatch_X and miinibatch_Y

        minibatches.append(minibatch_pair)

    if m%minibatch_size != 0 :

        last_minibatch_X = shuffled_X[:,(k+1)*minibatch_size: m ]

        last_minibatch_Y = shuffled_Y[:,(k+1)*minibatch_size: m ]

        last_minibatch_pair = (last_minibatch_X , last_minibatch_Y)

        minibatches.append(last_minibatch_pair)

    return minibatches
def model(X_train, Y_train, X_test, Y_test, learning_rate = 1,

          num_epochs = 1500, minibatch_size = 32):

    tf.reset_default_graph()

    n_x, m = X_train.shape                          # (n_x: input size, m : number of examples in the train set)

    n_y = Y_train.shape[0]                            # n_y : output size

    costs = []                                        # To keep track of the cost

    

    X, Y = create_placeholders(n_x, n_y)



    parameters = initialize_params(layer_dims)

    zL = forward_prop(X, parameters)



    cost = compute_cost(zL, Y,parameters)



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

                       

            if epoch % 100 == 0:

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

        correct_prediction = tf.equal(tf.argmax(zL), tf.argmax(Y))



        # Calculate accuracy on the test set

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))

        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        

        

        return parameters
layer_dims= [x_train.shape[0],25,12,10]  

_ = model(x_train, y_train, x_test, y_test, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32)