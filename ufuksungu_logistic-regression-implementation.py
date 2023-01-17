# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/PimaIndians.csv") 
data.info() 

data.head()
data.test=[0 if i=="negatif" else 1 for i in data.test ] 

y=data.test.values
data
x_data=data.drop(["test"],axis=1) 
x_data
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values 
from sklearn.model_selection import train_test_split #Arrange train&test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42) 
x_train=x_train.T

x_test=x_test.T

y_train=y_train.T

y_test=y_test.T

print("x train:",x_train.shape)

print("x test:",x_test.shape)

print("y train:",y_train.shape)

print("y test:",y_test.shape)
# initialize parameters

# dimension=counts of examples

def initialize_weight_and_bias(dimension):

    w=np.full((dimension,1),0.01)

    b=0.0

    return w,b

#calculation of z

def sigmoid(z):

    y_head=1/(1+np.exp(-z))

    return y_head

#x_train for features & y_train for loss method.

def forward_backward_propagation(w,b,x_train,y_train):

    #fp

    z=np.dot(w.T,x_train)+b

    y_head=sigmoid(z)

    loss=-(1-y_train)*np.log(1-y_head)-y_train*np.log(y_head) 

    #-((1-y_train)*np.log(1-y_head)+y_train*np.log(y_head))

    #(1-y_train)*np.log(1-y_head)+y_train*np.log(y_head) 

    #-(1-y_train)*np.log(1-y_head)-y_train*np.log(y_head) 

    cost=(np.sum(loss))/x_train.shape[1]

    #bp

    derivative_weight=(np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias=np.sum(y_head-y_train)/x_train.shape[1]

    gradients={"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}

    return cost,gradients

#Learning parameters

def update(w,b,x_train,y_train,learning_rate,number_of_iteration):

    cost_list=[]

    cost_list1=[]

    index=[]

    #1 Forward-Backward Propagation=1 Number of iteration

    #we do that for learning gradients,cost func and the most optimized parameters

    for i in range(number_of_iteration):

        cost,gradients=forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        w=w-learning_rate*gradients["derivative_weight"]

        b=b-learning_rate*gradients["derivative_bias"]

        #if i%2==0:

        cost_list1.append(cost)

        index.append(i)

        print("Cost after iteration %i: %f"%(i,cost))

    #update parameters with the most optimized parameters

    parameters={"weight":w,"bias":b}

    plt.plot(index,cost_list1)

    plt.xticks(index,rotation="vertical")

    plt.xlabel("Number of Iteration")

    plt.ylabel("Cost")

    plt.show()

    return parameters,gradients,cost_list



def predict(w,b,x_test):

    #x_test input for forward propagation

    z=sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction=np.zeros((1,x_test.shape[1]))

    for i in range(z.shape[1]):

        if z[0,i]<=0.5:

            Y_prediction[0,i]=0

        else:

            Y_prediction[0,i]=1

    return Y_prediction

def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,num_iterations):

    #initializng for w and b

    dimension=x_train.shape[0]

    w,b=initialize_weight_and_bias(dimension)

    #update parameters, learning gradients & cost func

    parameters,gradients,cost_list=update(w,b,x_train,y_train,learning_rate,num_iterations)

    y_prediction_test=predict(parameters["weight"],parameters["bias"],x_test)

    #y_test for learning accuracy of our model

    print("test accuracy:{}%".format(100-np.mean(np.abs(y_prediction_test-y_test))*100))

    

#train datalarını içeren tüm metodlar öğrenmek için, test datalarını içerenler ise tahmin için   
logistic_regression(x_train,y_train,x_test,y_test,learning_rate=4,num_iterations=200)
#Logistic Regression with sklearn

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train.T,y_train.T)

print("accuracy {}".format(lr.score(x_test.T,y_test.T)*100))