# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/breast-cancer/breast-cancer.csv')

data.head()
data.info()
data.drop(['id','Unnamed: 32'],axis=1,inplace=True)
data.head()
data.describe()
data2= data[['diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean',

                 'smoothness_mean','area_worst','concavity_mean','concave points_mean',

                 'symmetry_mean','fractal_dimension_mean']]
color_list = ['cyan' if i=='M' else 'lime' for i in data2.loc[:,'diagnosis']]

pd.plotting.scatter_matrix(data2.loc[:, data2.columns != 'diagnosis'],

                           c=color_list,

                           figsize= [20,20],

                           diagonal='hist',

                           alpha=0.5,

                           s = 200,

                           marker = '*',

                           edgecolor= "black")

                                        

plt.show()
# Values of 'Benign' and 'Malignant' cancer cells

data.diagnosis.value_counts()
#Visualization

sns.countplot(x="diagnosis", data=data)

plt.show()
g = sns.jointplot(data.radius_mean, data.smoothness_mean, kind="kde", size=7)

plt.savefig('graph.png')

plt.show()
plt.figure(figsize=(25,25))

sns.heatmap(data.corr(),annot=True,cmap='RdBu_r')
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

print(data.info())

#Firstly M and B values update 0 and zero. Because not using string.
y = data.diagnosis.values

x_data = data.drop(["diagnosis"],axis=1)
#normalization

#This is a formul>>   (x - min(x))/(max(x)-min(x))

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
x.shape

# change value (matrix)
# %30 testing %70 training ///  random constant=42

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=42)



x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T



print("x_train: ",x_train.shape)

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)
# dimension =feauture values



def initialize_weights_and_bias(dimension):

    

    w = np.full((dimension,1),0.01)

    b = 0.0  #float values

    return w,b
look = np.full((6,1),0.01)

print(look)
# Let's calculating z

# z = np.dot(w.T,x_train)+b



def sigmoid(z):

    

    y_head = 1/(1+ np.exp(-z))

    return y_head
print(sigmoid(-6))

print(sigmoid(0))

print(sigmoid(6))
x_train.shape[1] #for scaling
def forward_backward_propagation(w,b,x_train,y_train):

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]  

    

    # backward propagation

    #weight turev

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] 

    #bias turev

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]           

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    

    return cost,gradients
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    

    # updating(learning) parameters is number_of_iterarion times

    for i in range(number_of_iterarion):

        # make forward and backward propagation and find cost and gradients

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        # lets update

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 10 == 0:

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

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction
x_test.shape[1]
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    # initialize

    dimension =  x_train.shape[0]  # that is 30

    w,b = initialize_weights_and_bias(dimension)

    # do not change learning rate

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)



    # Print test Errors

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 3, num_iterations = 300)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train.T,y_train.T)

print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))