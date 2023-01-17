# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/voice.csv")
data.info()
# predicted label value cannot be object . It must be integer or category . So we convert label column to integer 
print(data.label.unique())
data.label = [1 if each =='female' else 0 for each in data.label ]

y = data.label.values.reshape(-1,1)
x_data = data.drop(["label"], axis=1)
x = (x_data - np.min(x_data)) /(np.max(x_data) - np.min(x_data)).values
x.head()

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split (x,y, test_size =0.2, random_state = 42)

# we get transposes since in the next step we will use Transposes 
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train shape: ",x_train.shape )
print("y_train shape: ",y_train.shape )
print("x_test shape: ",x_test.shape )
print("y_test shape: ",y_test.shape )

# we have 20 features and 2534 sample for train data
# we have 20 features and 634  sample for test data
def initialize_weights_and_bias (dimension):
    w = np.full( (dimension, 1 ), 0.01)
    b =0.0
    return w,b

def sigmoid (z):
    y_head = 1 /(1 + np.exp(-z))
    return y_head
def forward_backward_propagation (w,b, x_train, y_train ):
    #forward propagation
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)    
    
    # loss = -(1-y)log(1-y^) - y*og(y^)
    # cost = sum of loss 
    loss =  - (1-y_train)*np.log(1-y_head) - y_train * np.log(y_head)
    cost = (np.sum(loss)) / x_train.shape[1] #  x_train.shape[1]  for scaling 
    
    #backward propagation
    derivative_weight = (np.dot(x_train, (( y_head - y_train).T))) / x_train.shape[1]  # x_train.shape[1] for scaling. x.shape deki axis =1 adetini veriyor. Su anda xaxis=1'de ornek adeti var , xaxis=0 da feature adeti 
    derivative_bias = np.sum(y_head - y_train)/ x_train.shape[1]
    gradients = {"derivative_weight":derivative_weight, "derivative_bias": derivative_bias }
    
    return cost, gradients
def update(w, b, x_train, y_train, learning_rate,number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iteration):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]  #derivative_weight weight in costa gore turevi
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 100 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list
def predict(w,b,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        #print ("probabilty of tumor for " , i , " : " ,  z[0,i] )  
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 30
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    
    # Print test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

#lets  run
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 1000)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train.T, y_train.T)

print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))