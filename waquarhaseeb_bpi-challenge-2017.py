# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import random

import pprint

import pandas as pd

import tensorflow.compat.v1 as tf

from tensorflow.python.framework import ops

from sklearn.model_selection import train_test_split

tf.disable_eager_execution()
data = pd.read_csv(r"/kaggle/input/bpi-challenge-2017-dataset/BPI Challenge 2017.csv", engine="c")
data.keys()
data=data[data['EventOrigin']!='Workflow']
data=data.drop(['Action','org:resource','EventOrigin','EventID','lifecycle:transition','case:LoanGoal','case:ApplicationType','case:concept:name','OfferID'],axis=1)
useful=['A_Cancelled','A_Pending','O_Create Offer','A_Denied']

data=data[data['concept:name'].isin(useful)].reset_index()

data
statlist=[]

for i in range(len(data)):

    if data['concept:name'][i]=='O_Create Offer':

        stats=[]

        stats.append(data['case:RequestedAmount'][i])

        stats.append(data['FirstWithdrawalAmount'][i])

        stats.append(data['NumberOfTerms'][i])

        stats.append(data['Accepted'][i])

        stats.append(data['MonthlyCost'][i])

        stats.append(data['Selected'][i])

        stats.append(data['CreditScore'][i])

        stats.append(data['OfferedAmount'][i])

    elif data['concept:name'][i]=='A_Pending':

        stats.append(1)

        statlist.append(stats)

    elif data['concept:name'][i]=='A_Denied':

        stats.append(-1)

        statlist.append(stats)

    else:

        stats.append(0)

        statlist.append(stats)
statlist=pd.DataFrame(statlist,columns=['RequestedAmount','FirstWithdrawalAmount','NumberOfTerms','Accepted','MonthlyCost','Selected','CreditScore','OfferedAmount','Labels','NULL'])

statlist=statlist.drop(['NULL'],axis=1)
statlist
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label = LabelEncoder()

statlist['Accepted']=label.fit_transform(statlist['Accepted'])

statlist['Selected']=label.fit_transform(statlist['Selected'])
dataX = (statlist[['RequestedAmount','FirstWithdrawalAmount','NumberOfTerms','Accepted','MonthlyCost','Selected','CreditScore','OfferedAmount']])

dataY = statlist[['Labels']]

dataX = np.asarray(dataX)

dataY = np.asarray(dataY)


X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.33, random_state=42)

X_train=X_train.T

X_test=X_test.T

Y_train=Y_train.T

Y_test=Y_test.T
X_train.shape
Y_train.shape
def create_placeholders(n_x, n_y):

    """

   

    

    Arguments:

    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)

    n_y -- scalar, number of classes (from 0 to 5, so -> 6)

    

    Returns:

    X -- placeholder for the data input, of shape [n_x, None] and dtype "tf.float32"

    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "tf.float32"

    

    

    """



    

    X = tf.placeholder(tf.float32,[n_x,None],name='X')

    Y = tf.placeholder(tf.float32,[n_y,None],name='Y')

    

    

    return X, Y
def initialize_parameters():

    

    

    tf.set_random_seed(1)                   # so that your "random" numbers match ours

        

    

    W1 = tf.get_variable('W1',[25,8],initializer=tf.glorot_uniform_initializer)

    b1 = tf.get_variable('b1',[25,1],initializer=tf.zeros_initializer())

    W2 = tf.get_variable('W2',[12,25],initializer=tf.glorot_uniform_initializer)

    b2 = tf.get_variable('b2',[12,1],initializer=tf.zeros_initializer())

    W3 = tf.get_variable('W3',[6,12],initializer=tf.glorot_uniform_initializer)

    b3 = tf.get_variable('b3',[6,1],initializer=tf.zeros_initializer())

   



    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2,

                  "W3": W3,

                  "b3": b3}

    

    return parameters
def forward_propagation(X, parameters):

    """

    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    

    Arguments:

    X -- input dataset placeholder, of shape (input size, number of examples)

    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"

                  the shapes are given in initialize_parameters



    Returns:

    Z3 -- the output of the last LINEAR unit

    """

    

    # Retrieve the parameters from the dictionary "parameters" 

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    W3 = parameters['W3']

    b3 = parameters['b3']

    

               # Numpy Equivalents:

    Z1 = tf.add(tf.matmul(W1,X),b1)                                              # Z1 = np.dot(W1, X) + b1

    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)

    Z2 = tf.add(tf.matmul(W2,A1),b2)                                              # Z2 = np.dot(W2, A1) + b2

    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)

    Z3 = tf.add(tf.matmul(W3,A2),b3)                                              # Z3 = np.dot(W3, A2) + b3

    

    

    return Z3
def compute_cost(Z3, Y):

    """

    Computes the cost

    

    Arguments:

    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)

    Y -- "true" labels vector placeholder, same shape as Z3

    

    Returns:

    cost - Tensor of the cost function

    """

    

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)

    logits = tf.transpose(Z3)

    labels = tf.transpose(Y)

    

    

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))

   

    return cost
import math

import matplotlib.pyplot as plt
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

    

    m = X.shape[1]                  # number of training examples

    mini_batches = []

    np.random.seed(seed)

    

    # Step 1: Shuffle (X, Y)

    permutation = list(np.random.permutation(m))

    shuffled_X = X[:, permutation]

    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))



    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning

    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]

        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    # Handling the end case (last mini-batch < mini_batch_size)

    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]

        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    return mini_batches
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0000001,

          num_epochs = 1000, minibatch_size = 32, print_cost = True ):

    """

    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    

    

    Returns:

    parameters -- parameters learnt by the model. They can then be used to predict.

    """

    

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables

    seed = 3                                          # to keep consistent results

    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)

    n_y = Y_train.shape[0]                            # n_y : output size

    costs = []                                        # To keep track of the cost

    

    # Create Placeholders of shape (n_x, n_y)

    

    X, Y = create_placeholders(n_x, n_y)

    



    # Initialize parameters

    

    parameters = initialize_parameters()

    

    

    # Forward propagation: Build the forward propagation in the tensorflow graph

   

    Z3 = forward_propagation(X, parameters)

   

    

    # Cost function: Add cost function to tensorflow graph

    

    cost = compute_cost(Z3, Y)

   

    

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.

    

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    

    

    # Initialize all the variables

    init = tf.global_variables_initializer()



    # Start the session to compute the tensorflow graph

    with tf.Session() as sess:

        

        # Run the initialization

        sess.run(init)

        

        # Do the training loop

        for epoch in range(num_epochs):



            epoch_cost = 0.                       # Defines a cost related to an epoch

            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set

            seed = seed + 1

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)



            for minibatch in minibatches:



                # Select a minibatch

                (minibatch_X, minibatch_Y) = minibatch

                

                # IMPORTANT: The line that runs the graph on a minibatch.

                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).

                ### START CODE HERE ### (1 line)

                _ , minibatch_cost = sess.run([optimizer, cost],{X: minibatch_X, Y: minibatch_Y})

                ### END CODE HERE ###

                

                epoch_cost += minibatch_cost / minibatch_size



            # Print the cost every epoch

            if print_cost == True and epoch % 100 == 0:

                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

            if print_cost == True and epoch % 5 == 0:

                costs.append(epoch_cost)

                

        # plot the cost

        plt.plot(np.squeeze(costs))

        plt.ylabel('cost')

        plt.xlabel('iterations (per fives)')

        plt.title("Learning rate =" + str(learning_rate))

        plt.show()



        # lets save the parameters in a variable

        parameters = sess.run(parameters)

        print ("Parameters have been trained!")



        # Calculate the correct predictions

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))



        # Calculate accuracy on the test set

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



       

        return parameters
parameters = model(X_train, Y_train, X_test, Y_test)