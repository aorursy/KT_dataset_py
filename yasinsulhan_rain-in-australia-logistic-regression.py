# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings

warnings.warn_explicit

        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Loading Data

data = pd.read_csv("/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv")
data.head()
data.describe()
data.info()
#lets drop some variables

data.drop(["Date","Location","WindGustDir","WindDir9am","WindDir3pm","RainToday"],axis=1,inplace=True)
data.RainTomorrow.value_counts()
#RainTomorrow must be object type

data.astype({'RainTomorrow': 'object'}).dtypes
data.RainTomorrow = [1 if each == "Yes" else 0 for each in data.RainTomorrow]
#done

data.RainTomorrow.value_counts()
#done

data.info()
#to be taken mean values each  variables

variable_list = [each for each in data.columns]

#except RainTomorrow because it is not numeric values

variable_list.remove("RainTomorrow")

variable_list
data["MinTemp"].mean()
#Let's get rid of missing values :)

for i in variable_list:

    average_value = data[i].mean()

    data[i].fillna(average_value,inplace=True)
#Done

data.info()
y = data.RainTomorrow.values

x_data = data.drop(["RainTomorrow"],axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T
print("x_train: ",x_train.shape)

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)
def initialize_weights_and_bias(dimension):

    

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w,b

def sigmoid(z):

    

    y_head = 1/(1 + np.exp(-z))

    return y_head

def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]

    

    # backward propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    

    return cost,gradients

def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):

    cost_list = []

    cost_list2 = [] #her 10 adimda bir costlari depolamak icin(ilerde analiz edebilmek icin...)

    index = []

    

    # updating(learning) parameters is number_of_iterarion times

    for i in range(number_of_iterarion): #kac kere forward-backward iteration yapacagimizi dongu icinde...

        # make forward and backward propagation and find cost and gradients

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        # lets update ->weight bias burada guncellenecek adım sayısına gore(number of iterations)

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

            

    # we update(learn) parameters weights and bias

    parameters = {"weight": w,"bias": b}

    plt.figure(figsize=(15,9))

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost



def predict(w,b,x_test):

    # x_test is a input for forward propagation

    z = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    # initialize

    dimension =  x_train.shape[0]

    w,b = initialize_weights_and_bias(dimension)

    # do not change learning rate

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)



    # Print test Errors

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

#learning rate arttikca hizli ogrenir...

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 3, num_iterations = 300)    
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train.T,y_train.T)

print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))