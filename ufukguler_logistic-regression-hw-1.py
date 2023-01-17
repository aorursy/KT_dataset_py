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
data = pd.read_csv("../input/heart.csv")

data.info()

data.head()
y = data.target.values

x_data = data.drop(['target'],axis = 1)

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)



x_train = x_train.T

y_train = y_train.T

x_test = x_test.T

y_test = y_test.T
def initialize_prams(dimensions):

    w = np.full((dimensions,1),0.01)

    b = 0.0

    return w,b
def sigmoid(z):

    y_head = 1/(1 + np.exp(-z))

    return y_head
def Forward_Backward_Propagation(w,b,x_train,y_train):

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]

    derivative_weights = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias = (np.sum(y_head-y_train)/x_train.shape[1])

    gradients = {"derivative_weights":derivative_weights,"derivative_bias":derivative_bias}

    return cost, gradients
def update(w,b,x_train,y_train,learning_rate,number_of_iteration):

    cost_list  = []

    cost_list2 = []

    index = []

    for i in range(number_of_iteration):

        cost,gradients = Forward_Backward_Propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        

        w = w-learning_rate * gradients["derivative_weights"]

        b = b-learning_rate * gradients["derivative_bias"]

        

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print("cost after iteration %i : %f"%(i,cost))

            

        parameters = {"weights":w,"bias":b}

        plt.plot(index,cost_list2)

        plt.xticks(index,rotation = "vertical")

        plt.show()

        return parameters,gradients,cost_list

    

    
def predict(w,b,x_test):

    z = sigmoid(np.dot(w.T,x_test)+b)

    y_prediction = np.zeros((1,x_test.shape[1]))

    

    for i in range(z.shape[1]):

        if z[0,i] <= 0.5:

            y_prediction[0,1] = 0

        else:

            y_prediction[0,1] = 1

    return y_prediction
def logistic_regression(x_train,y_train,_test,y_test,learning_rate,number_of_iteration):

    

    dimensions = x_train.shape[0]

    w,b = initialize_prams(dimensions)

    parameters,gradients,cost_list = update(w,b,x_train,y_train,learning_rate,number_of_iteration)

    y_prediction_test = predict(parameters["weights"],parameters["bias"],x_test)

    y_prediction_train = predict(parameters["weights"],parameters["bias"],x_train)

    

    print("test accuracy : {}%".format(100-np.mean(np.abs(y_prediction_test-y_test))*100))

    print("train accuracy : {}%".format(100-np.mean(np.abs(y_prediction_train-y_train))*100))

    

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 3, number_of_iteration = 250) 
from sklearn.linear_model import LogisticRegression

lr= LogisticRegression()

lr.fit(x_train.T,y_train.T)

print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))