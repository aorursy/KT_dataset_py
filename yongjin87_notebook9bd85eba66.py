# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.utils import shuffle

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Read Train data from CSV file

# Train and valid data set

df = pd.read_csv("../input/train.csv")

data = df.as_matrix().astype(np.float32)

data = shuffle(data)

# Make X in [0, 1]

Xtrain = data[:-1000,1:] / 255

Ytrain = data[:-1000,0].astype(np.int32)

Xvalid = data[-1000:,1:] / 255

Yvalid = data[-1000:,0].astype(np.int32)



# Read Test data from CSV file

# Test data set

df_test = pd.read_csv("../input/test.csv")

data_test = df_test.as_matrix().astype(np.float32)

Xtest = data_test



print("Xtrain", Xtrain.shape, "Xvalid", Xvalid.shape, "Xtest", Xtest.shape)
# Add usefull functions

def error_rate(targets, predictions):

    return np.mean(targets != predictions)



def sigmoid(A):

    return 1 / 1(1+np.exp(-A))
# AutoEncoder class

class AutoEncoder(object):

    def __init__(self, M1, M2, an_id):

        self.M1 = M1

        self.M2 = M2

        self.id = id

        

        # initialize variables

        w_init = np.random.randn(self.M1, self.M2) / np.sqrt(self.M1 + self.M2)

        bh_init = np.zeros(self.M2)

        bo_init = np.zeros(self.M1)

        

        self.w = tf.Variable(w_init.astype(np.float32))

        self.bh = tf.Variable(bh_init.astype(np.float32))

        self.bo = tf.Variable(bo_init.astype(np.float32))

        self.params = [self.w, self.bh, self.bo]

        self.forward_params = [self.w, self.bh]

        

    '''    

    def fit(self, X):

        # Set parameters

        N, D = X.shape

        lr = 0.5

        mu = 0.99

        max_iter = 1

        batch_sz = 100

        n_batches = int(N / batch_sz)

        

        #X_in = tf.placeholder(tf.float32, shape = (N, D), name="X_in")

        X_in = tf.placeholder(tf.float32, shape = (None, D), name="X_in")

        

        # Calculate X hat

        X_hat = self.forward_output(X_in)

        

        # Cost function

        #cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(X, X_hat))

        cost = X_in - X_hat

        

        # Operatin

        train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)

        init_op = tf.initialize_all_variables()

        

        with tf.Session() as session:

            session.run(init_op)

            

            for i in range(max_iter):

                for j in range(n_batches):

                    Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]

                    session.run(train_op, feed_dict={X_in:Xbatch})

                    c = session.run(cost, feed_dict={X_in:Xbatch})

                    #print("Cost :", c)

    '''

        

    def forward_hidden(self, X):

        Z = tf.sigmoid(tf.matmul(X, self.w) + self.bh)

        #Z = tf.nn.relu(tf.matmul(X, self.w) + self.bh)

        return Z

    

    def forward_output(self, X):

        Z = self.forward_hidden(X)

        Y = tf.sigmoid(tf.matmul(Z, tf.transpose(self.w) + self.bo))

        return Y
# Make Nueral Network

class DNN(object):

    def __init__(self, hidden_layer_sizes):

        self.hidden_layer_sizes = hidden_layer_sizes

        

    def fit(self, X, Y, Xvalid, Yvalid, pretrain=True):

        lr = 0.01

        mu = 0.99

        reg = 0.1

        max_iter = 1

        batch_sz = 100

        N, D = X.shape

        

        # Make hidden layers

        self.hidden_layers = []

        count = 0

        M1 = D

        for M2 in self.hidden_layer_sizes:

            ae = AutoEncoder(M1, M2, count)

            self.hidden_layers.append(ae)

            M1 = M2

            count += 1

        

        # Run Auto Encoder

        lr = 0.5

        mu = 0.99

        max_iter = 1

        batch_sz = 100

        n_batches = int(N / batch_sz)

        

        #X_in = tf.placeholder(tf.float32, shape = (N, D), name="X_in")

        X_in = tf.placeholder(tf.float32, shape = (None, D), name="X_in")

        

        # Calculate X hat

        X_hat = self.forward_output(X_in)

        

        # Cost function

        #cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(X, X_hat))

        cost = X_in - X_hat

        

        # Operatin

        train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)

        init_op = tf.initialize_all_variables()

        

        with tf.Session() as session:

            session.run(init_op)

            

            for i in range(max_iter):

                for j in range(n_batches):

                    Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]

                    session.run(train_op, feed_dict={X_in:Xbatch})

                    c = session.run(cost, feed_dict={X_in:Xbatch})

                    #print("Cost :", c)

        '''

        current_input = X

        for ae in self.hidden_layers:

            print("123")

            ae.fit(current_input)

            #current_input = ae.hdden_op(current_input)

            current_input = ae.forward_hidden(current_input)

        '''

        

        

        # Logistic regression layer

        N = len(Y)

        K = len(set(Y))

        w_init = np.random.randn(self.hidden_layers[-1].M, K) / np.sqrt(self.hidden_layers[-1].M + K)

        b_init = np.zeros(K)

        self.w = tf.Variable(w_init, "W_logreg")

        self.b = tf.Variable(b_init, "b_logreg")

        

        self.params = [self.w, self.b]

        for ae in self.hidden_layers:

            self.params += ae.forward_params

            

        labels = tf.placeholder(tf.float32, shape=(None, K), name='labels')

        pY = self.forward(X_in)

        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(pY, labels))

        

        train_op = tf.train.GradientDescentOptimizer(lr).minimiaze(cost)

        prediction = tf.argmax(logits, 1)

            

        init = tf.initialize_all_variables()     

        

        n_batches = int(N / batch_sz)

        with tf.Session() as session:

            session.run(init)

            

            for i in range(max_iter):

                X, Y = shuffle(X, Y)

                for j in range(n_batches):

                    Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]

                    Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                    session.run(train_op, feed_dict={X_in:Xbatch, labels:Ybatch})

            

            

    def predict(self, X):

        return tf.argmax(self.forward(X), axis=1)

    

    def forward(self, X):

        current_input = X

        for ae in self.hidden_layers:

            Z = ae.forward_hidden(current_input)

            current_input = Z

            

        Y = tf.nn.softmax(tf.matmul(current_input, self.w) + self.b)

        return Y

        

        
# Run 

dnn  = DNN([1000, 750, 500])

dnn.fit(Xtrain, Ytrain, Xvalid, Yvalid)