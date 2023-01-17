# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd .read_csv('../input/voice.csv')
data.head()
data.info()
data.label = [1 if i == "male" else 0 for i in data.label]    #changing "label" values

y = data.label.values

x_data = data.drop(["label"], axis=1)

x = ((x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))).values    #normalization
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T
def initialize_w_and_b(dimension):

    w = np.full((dimension,1), 0.01)

    b = 0.0

    return w, b
def sigmoid(z):

    y_head = 1 / (1+np.exp(-z))

    return y_head
def forward_backward_propagation(w, b, x_train, y_train):

    z = np.dot(w.T, x_train) + b

    y_head = sigmoid(z)

    loss = -((1-y_train)*np.log(1-y_head))-(y_train*np.log(y_head))

    cost = (np.sum(loss)) / x_train.shape[1]

    

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    return cost, gradients
def update(w, b, x_train, y_train, learning_rate, num_iterations):

    cost_list = []

    cost_list2 = []

    index = []

    

    for i in range(num_iterations):

        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)

        cost_list.append(cost)

        

        w = w - learning_rate * gradients["derivative_weight"]

        b = b- learning_rate * gradients["derivative_bias"]

        

        if i%10 == 0:

            cost_list2.append(cost)

            index.append(i)

            

    parameters = {"weight":w, "bias":b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list
def predict(w, b, x_test):

    z = sigmoid(np.dot(w.T, x_test)+b)

    y_prediction = np.zeros((1, x_test.shape[1]))

    for i in range(z.shape[1]):

        if z[0,i] <= 0.5:

            y_prediction[0,i] = 0

        else:

            y_prediction[0,i] = 1

    return y_prediction
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):

    dimension = x_train.shape[0]

    w, b = initialize_w_and_b(dimension)

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)

    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
logistic_regression(x_train, y_train, x_test, y_test, learning_rate=1, num_iterations=3000)