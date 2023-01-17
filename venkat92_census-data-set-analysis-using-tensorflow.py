# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow.python.framework import ops

import sklearn

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import numpy as np

import pylab

from sklearn.preprocessing import LabelEncoder

from sklearn.base import BaseEstimator, TransformerMixin

import seaborn as sns

import math 

%matplotlib inline





# Input data files are available in the ".ut./inp/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/census-data-set/Census Income Dataset.csv',na_values = [' ?'])

dataset.head()
test_dataset = pd.read_csv('../input/census-test-dataset/Census Income Testset.csv', na_values =[' ?'])

test_dataset.head()
#Apply feature discretization to Income_bracket column

def Income_bracket_binarization(feat_val):

    if feat_val == '<=50K':

        return 0

    else:

        return 1

    

dataset['Income_bracket'] = dataset['Income_bracket'].apply(Income_bracket_binarization)

test_dataset['Income_bracket'] = test_dataset['Income_bracket'].apply(Income_bracket_binarization)



dataset.head()
test_dataset.head()
class CategoricalImputer():

  

    def __init__(self, columns = None, strategy='most_frequent'):

        self.columns = columns

        self.strategy = strategy

    

    

    def fit(self,X, y=None):

        if self.columns is None:

            self.columns = X.columns

            print(self.columns)

    

        if self.strategy is 'most_frequent':

            self.fill = {column: X[column].value_counts().index[0] for 

        column in self.columns}

            

        else:

              self.fill ={column: '0' for column in self.columns}



        return self

    

    def transform(self,X):

        for column in self.columns:

            X[column] = X[column].fillna(self.fill[column])

        return X

    
obj = CategoricalImputer(columns=

          ['workclass','Occupation', 'Native_Country'])

train_result = obj.fit(dataset[['workclass','Occupation', 'Native_Country']])



dataset[['workclass','Occupation', 'Native_Country']] = train_result.transform(dataset[['workclass','Occupation', 'Native_Country']])



test_obj = CategoricalImputer(columns=

          ['workclass','Occupation', 'Native_Country'])

test_result = test_obj.fit(test_dataset[['workclass','Occupation', 'Native_Country']])



test_dataset[['workclass','Occupation', 'Native_Country']] = test_result.transform(test_dataset[['workclass','Occupation', 'Native_Country']])

def Plot():

    #Find Indices where Income is >50K and <=50K

    fig = plt.figure(figsize=(15,15))

    fig.subplots_adjust(hspace=0.7, wspace=0.7)

    pylab.suptitle("Analyzing the dataset", fontsize="xx-large")

    plt.subplot(3,2,1)

    ax = sns.countplot(x='Age', hue='Income_bracket', data=dataset)

    plt.subplot(3,2,2)

    ax =sns.countplot(x='workclass', hue='Income_bracket', data=dataset)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

    plt.subplot(3,2,3)

    ax =sns.countplot(x='Education', hue='Income_bracket', data=dataset)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

    plt.subplot(3,2,4)

    ax = sns.countplot(x='Occupation', hue='Income_bracket', data=dataset)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

    plt.subplot(3,2,5)

    ax = sns.countplot(x='Gender', hue='Income_bracket', data=dataset)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

    plt.subplot(3,2,6)

    ax = sns.countplot(x='hours_per_week', hue='Income_bracket', data=dataset)

    

    return None

#Analyzing distribution for the dataset

dataset.hist(

    column=["Age","Education", "hours_per_week"],figsize=(6, 5))



pylab.suptitle("Analyzing distribution for the dataset", fontsize="xx-large")



Plot()



X = dataset.drop('Income_bracket',axis =1)

y = dataset['Income_bracket']



#Split data set into training set and test set

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

class Categorical_Encoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):

        self.columns  = columns

        self.encoders = None

    def fit(self, data, target=None):

        """

        Expects a data frame with named columns to encode.

        """

        # Encode all columns if columns is None

        if self.columns is None:

            self.columns = data.columns

        # Fit a label encoder for each column in the data frame

        self.encoders = {

            column: LabelEncoder().fit(data[column])

            for column in self.columns

        }

        return self

    def transform(self, data):

        """

        Uses the encoders to transform a data frame.

        """

        output = data.copy()

        for column, encoder in self.encoders.items():

            output[column] = encoder.transform(data[column])

        return output

    
categorical_features = {

        column: list(dataset[column].unique())

        for column in dataset.columns

        if dataset[column].dtype == 'object'

    }

encoder = Categorical_Encoder(categorical_features.keys())

dataset = encoder.fit_transform(dataset)
dataset.head()
data = dataset.values

X_train = np.float32(data[:,[0,1,2,3,5,6,7,8,9,10,11,12]])

Y_train = data[:,[13]]
categorical_features = {

        column: list(test_dataset[column].unique())

        for column in test_dataset.columns

        if test_dataset[column].dtype == 'object'

    }

encoder = Categorical_Encoder(categorical_features.keys())

test_dataset = encoder.fit_transform(test_dataset)
test_dataset.head()
def computeCost(A, y):



    """

    Computes the cost using the sigmoid cross entropy

    

    Arguments:

    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)

    labels -- vector of labels y (1 or 0) 

    

    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels" 

    in the TensorFlow documentation. So logits will feed into z, and labels into y. 

    

    Returns:

    cost -- runs the session of the cost (formula (2))

    """

    

    A= tf.cast(A,tf.float32)

    y = tf.cast(y,tf.float32)

    

    cross_entropy_cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = A, labels = y)

    

    return  cross_entropy_cost

    
def Xavier_Intializer(dim):

    

    tf.set_random_seed(1)

    w = tf.get_variable("w", [dim, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))

    b = 0



    assert(w.shape == (dim,1))

    assert(isinstance(b, float) or isinstance(b, int))

        

    return w,b

    
def propagate(X,y,w,b):

    

    #Implements the forward propagation for the model: LINEAR -> SIGMOID

    # Retrieve the parameters from the dictionary "parameters" 

    Z = tf.add(tf.matmul(X,w), b)                      # Z = np.dot(W, X) + b

    A =  tf.sigmoid(Z)                             # A1 = sigmoid(Z1)

    

    return A


def random_mini_batches(X, Y, mini_batch_size, seed = 0):

    

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
def model(X_train, Y_train,minibatch_size,learning_rate = 0.005,num_iterations = 1500,num_epochs = 1500, print_cost = True):

    

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables

    (m, n_x) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)

    print(m)

    n_y = Y_train.shape[0]                            # n_y : output size

    costs = []                                        # To keep track of the cost

    seed = 3

    

    # Initialize parameters

    w,b = Xavier_Intializer(n_x)

    

    # Forward propagation: Build the forward propagation in the tensorflow graph

    A = propagate(X_train, Y_train,w,b)

    # Cost function: Add cost function to tensorflow graph

    cost = computeCost(A,Y_train)    

    

    # Backpropagation: Define the tensorflow optimizer. Use a Gradient Descent

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

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

            minibatches = random_mini_batches(X_train.T, Y_train, minibatch_size, seed)



            for minibatch in minibatches:



                # Select a minibatch

                (minibatch_X, minibatch_Y) = minibatch

                

                # IMPORTANT: The line that runs the graph on a minibatch.

                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).

                ### START CODE HERE ### (1 line)

                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                ### END CODE HERE ###

                

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

        w = sess.run(w)

        b = sess.run(b)

        print("Parameters have been trained!")

        

        return w,b

w,b = model(X_train, Y_train,minibatch_size = 256)