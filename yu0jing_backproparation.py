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
import numpy as np

# X = (hours sleeping, hours studying), y = score on test
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# scale units
X = X/np.amax(X, axis=0) # maximum of X array
y = y/100 # max test score is 100

class Neural_Network(object):
    def __init__(self):
        #parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

    def forward(self, X):
        #forward propagation through our network
        self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.sigmoid(self.z3) # final activation function
        return o 

    def sigmoid(self, s):
        # activation function 
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        # backward propgate through the network
        self.o_error = y - o # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

NN = Neural_Network()
for i in range(1000): # trains the NN 1,000 times
    print ("Input: \n" + str(X)) 
    print ("Actual Output: \n" + str(y)) 
    print ("Predicted Output: \n" + str(NN.forward(X)))
    print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
    print ("\n")
NN.train(X, y)
#x = [hours of studying,hours of sleeping], y = score of test
x = np.array(([2,9],[1,5],[3,6]),dtype = float)
y = np.array(([92],[86],[89]),dtype = float)

#scale units
x = x/np.amax(x,axis = 0)#maximum of x array
y = y/100#max test score is 100

class neural_network(object):
    def __init__(self):
        #paremeters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        
        #weights
        self.w1 = np.random.randn(self.inputSize,self.hiddenSize)#2*3 weight matrix from input to hidden
        self.w2 = np.random.randn(self.hiddenSize,self.outputSize)#3*1 weight matrix from hidden to input
    def forward(self,x):
        #forward propagation 
        self.z = np.dot(x,self.w1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2,self.w2)
        o = self.sigmoid(self.z3)
        return o
    
    def sigmoid(self,s):
        #activation function
        return 1/(1+np.exp(-s))
    
    def sigmoid_primr(self,s):
        return s*(1-s)
    
    def backward(self,x,y,o):
        self.o_error = y-o
        self.o_delta = self.o_error*self.sigmoid_primr(o)
        
        self.z2_error = self.o_delta.dot(self.w2.T)
        self.z2_delta = self.z2_error*self.sigmoid_primr(self.z2)
        
        self.w1 += x.T.dot(self.z2_delta)
        self.w2 += self.z2.T.dot(self.o_delta)
        
    def train(self,x,y):
        o = self.forward(x)
        self.backward(x,y,o)
        
nn = neural_network()
for i in range(1000):
    print("input:"+str(x))
    print("actual output:"+str(y))
    print("predict output:"+str(nn.forward(x)))
    print("loss:"+str(np.mean(np.square(y-nn.forward(x)))))
    nn.train(x,y)
    

         
