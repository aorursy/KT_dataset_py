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
data = pd.read_csv("../input/voice.csv") #read data
data.head() #first five datas
print(data.label.unique())
data.label = [1 if i =='female' else 0 for i in data.label ]
y = data.label.values.reshape(-1,1)
x_data = data.drop(["label"], axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values
data.head()
x.head()
# Do you see the difference?
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)
# y => label
#x => feature
#test_size => %20
#random state => something like id

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train shape: ",x_train.shape )
print("y_train shape: ",y_train.shape )
print("x_test shape: ",x_test.shape )
print("y_test shape: ",y_test.shape )
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    #make dimensionx1 matrix full of 0.01. We use 0.01 but if we write 0, 0*x = 0 and that will cause our
    #code can't learn
    b = 0.0 # we want float
    return w,b
    
    #b is initial bias
def sigmoid(z):    
    y_head = 1/(1 + np.exp(-z)) #formula of sigmoid 
    return y_head

# If z = 0, function must give us 0.5 mathematically
sigmoid(0)

def forward_backward_propagation(w,b,x_train,y_train):
    
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    
    #backward prop.
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # derivative weight
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1] #derivative bias
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias} #parameters
    
    return cost,gradients

#number of it. = how many time backward-forward
def update(w, b, x_train, y_train, learning_rate, number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    
    #update parameters is number-of-iter. times
    for i in range(number_of_iteration):
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        #we need cost for  know how many time make iteration
        cost_list.append(cost)
        
        #update
        w = w - learning_rate * gradients["derivative_weight"]
        b= b - learning_rate * gradients["derivative_bias"]
        #stop when derivatives approach to zero

        if i % 10 == 0:
            cost_list2.append(cost) 
            index.append(i)
            print("cost after iteration %i: %f" %(i,cost))

    parameters = {"weight": w, "bias": b} #important part
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation = 'vertical')
    plt.xlabel("number of cost iteration")
    plt.ylabel("cost")
    plt.show()
    
    return parameters, gradients, cost_list
def predict(weight, bias, x_test):
    #x_test input for forward propagation    
    z = sigmoid(np.dot(weight.T,x_test)+bias)
    y_prediction = np.zeros((1,x_test.shape[1]))
    # if z > 0.5, prediction = 1 (y_head = 1)
    # if z < 0.5, prediction = 0 (y_head = 0)
    for i in range(z.shape[1]):
        if z[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    return y_prediction
def logistic_reg(x_train, y_train, x_test, y_test, learning_rate, num_iteration):
    dimension = x_train.shape[0] #need for initialize weight, that is 30
    w,b = initialize_weights_and_bias(dimension)
    parameters, gradients, cost_list = update(w,b,x_train,y_train,learning_rate,num_iteration)
    y_prediction_test = predict(parameters["weight"],parameters["bias"], x_test)
    
    #print errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_reg(x_train, y_train, x_test, y_test,learning_rate = 1, num_iteration = 500)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))