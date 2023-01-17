# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data_dir = "/kaggle/input/heart-disease-prediction-using-logistic-regression"



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





data = pd.read_csv(data_dir+"/framingham.csv")

data.drop(['education'],axis=1,inplace=True)

data.rename(columns={'male':'Sex_male'},inplace=True)

data.head()
#Check dataset for missing values

data.isnull().sum()
count = 0

for i in data.isnull().sum(axis=1):

    if i>0:

        count = count + 1

data.dropna(axis = 0 , inplace =True)

data.describe()
#Check values again.

data.isnull().sum()
print(data.info())

y = data.TenYearCHD.values

x_data = data.drop(["TenYearCHD"],axis = 1)

#Normalize data

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T

def initialize_weights_and_bias(dimension):

    w = np.full((dimension),0.01)

    b = 0.0

    return w,b



w,b = initialize_weights_and_bias(14) #For debugging.
def sigmoid (z):

    y_head = 1/(1+np.exp(-z))

    return y_head

print(sigmoid(0)) #debug

def forward_backward_propagation(w,b,x_train,y_train):

    

    z = np.dot(w.T,x_train) + b #•In tutorial we have learned fp's formula  z = W1X1+W2X2+....WnXn + bias(b). np.Dot does matrix multiplication

    y_head = sigmoid(z) # y_head means our predictions. We find this with sigmoid function. We implement it to the z.

    loss =-y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head) #loss function.

    cost = (np.sum(loss))/(x_train.shape[1]) # cost function is the sum of loss and divide by sample count.

    #now we will implement bp. in bp we need derivative of w's and b's

    

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias = derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients =gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

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

    w = w.reshape(-1,1)

    b = b.reshape(-1,1)

    z = sigmoid(np.dot(w.T,x_test)+b)

    z= z.reshape(-1,1)

    print(w.shape)

    Y_prediction = np.zeros((1,x_test.shape[1]))



   

    # if z is bigger than 0.5, our prediction is TenYearsCHD 1 (y_head=1),

    # if z is smaller than 0.5, our prediction is TenYearsCHD 0 (y_head=0),

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    # initialize

    dimension =  x_train.shape[0]  # that is 14

    print(dimension)

    w,b = initialize_weights_and_bias(dimension)

    # do not change learning rate

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)



    # Print test Errors

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 5000)  
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train.T,y_train.T)

print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))  