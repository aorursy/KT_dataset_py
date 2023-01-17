# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv",sep=",")
# Dropping null and non numeretic columns

data.dropna(inplace=True)

data.drop(["age","country","country-year","generation"],axis=1,inplace=True)
# convert male and femala 1 and 0

data.sex = [1 if each == "male" else 0 for each in data.sex]
# deleting spaces

data.rename(columns={' gdp_for_year ($) ':'gdp_year'}, inplace=True)

data.rename(columns={'HDI for year':'HDI_year'}, inplace=True)

data.rename(columns={'suicides/100k pop':'suicides/100k_pop'}, inplace=True)

data.rename(columns={'gdp_per_capita ($)':'gdp_per_capita_dollar'}, inplace=True)
# deleting comas

data.gdp_year = data.gdp_year.str.replace(',','')
# defining x and y

y = data.sex

x_data = data.drop(["sex"],axis=1)
# I should convert type of sixth column to float from string.

x_data.gdp_year = data.gdp_year.apply(lambda x: float(x))
# normalization

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
# train test split

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.15, random_state = 42)



x_train = x_train.values.T

x_test = x_test.values.T

y_test = y_test.values.reshape(1,y_test.shape[0])

y_train = y_train.values.reshape(1,y_train.shape[0])
# sigmoid function

def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head
# intialize parameters and layer sizes

def initialize_parameters_and_layer_sizes_NN(x_train, y_train):

    parameters = {"weight1": np.random.randn(3,x_train.shape[0]) * 0.1,

                  "bias1": np.zeros((3,1)),

                  "weight2": np.random.randn(y_train.shape[0],3) * 0.1,

                  "bias2": np.zeros((y_train.shape[0],1))}

    return parameters
# forward propagation

def forward_propagation_NN(x_train, parameters):



    Z1 = np.dot(parameters["weight1"],x_train) +parameters["bias1"]

    A1 = np.tanh(Z1)

    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]

    A2 = sigmoid(Z2)



    cache = {"Z1": Z1,

             "A1": A1,

             "Z2": Z2,

             "A2": A2}

    

    return A2, cache
# Compute cost

def compute_cost_NN(A2, Y, parameters):

    logprobs = np.multiply(np.log(A2),Y)

    cost = -np.sum(logprobs)/Y.shape[1]

    return cost
# Backward Propagation

def backward_propagation_NN(parameters, cache, X, Y):



    dZ2 = cache["A2"]-Y

    dW2 = np.dot(dZ2,cache["A1"].T)/X.shape[1]

    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1]

    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2))

    dW1 = np.dot(dZ1,X.T)/X.shape[1]

    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]

    grads = {"dweight1": dW1,

             "dbias1": db1,

             "dweight2": dW2,

             "dbias2": db2}

    return grads
# update parameters

def update_parameters_NN(parameters, grads, learning_rate = 0.01):

    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"],

                  "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],

                  "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],

                  "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}

    

    return parameters
# prediction

def predict_NN(parameters,x_test):

    # x_test is a input for forward propagation

    A2, cache = forward_propagation_NN(x_test,parameters)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    for i in range(A2.shape[1]):

        if A2[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction
# 2 - Layer neural network

def two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations):

    cost_list = []

    index_list = []

    #initialize parameters and layer sizes

    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)



    for i in range(0, num_iterations):

         # forward propagation

        A2, cache = forward_propagation_NN(x_train,parameters)

        # compute cost

        cost = compute_cost_NN(A2, y_train, parameters)

         # backward propagation

        grads = backward_propagation_NN(parameters, cache, x_train, y_train)

         # update parameters

        parameters = update_parameters_NN(parameters, grads)

      

        if i % 100 == 0:

            cost_list.append(cost)

            index_list.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))    

    plt.plot(index_list,cost_list)

    plt.xticks(index_list,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    

    

    # predict

    y_prediction_test = predict_NN(parameters,x_test)

    y_prediction_train = predict_NN(parameters,x_train)



    # Print train/test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    return parameters



parameters = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=3000)
# reshaping

x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T
# Evaluating the ANN

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential # initialize neural network library 

from keras.layers import Dense # build our layers library

def build_classifier():

    classifier = Sequential() # initialize neural network

    classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1])) # we use dimension of x_train as input

    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu')) # we use 4 nodes in first layer

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) # if we use sigmoid function it means we add output layer

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # we will use accuracy as metrics

    return classifier



classifier = KerasClassifier(build_fn = build_classifier, epochs = 100) # epochs means that is number of iteration 

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)

mean = accuracies.mean()

variance = accuracies.std()

print("Accuracy mean: "+ str(mean))

print("Accuracy variance: "+ str(variance))
dict_of_values = {'ANN_Num': [], 'CV': [],'epochs': [] , 'accuracy': [] }

data_temproray = pd.DataFrame.from_dict(dict_of_values)
data_temproray.ANN_Num = [2]

data_temproray.CV = [3]

data_temproray.epochs = [100]

data_temproray.accuracy = [0.6558003964712986]
data_temproray.head()
# Evaluating the ANN V2

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential # initialize neural network library 

from keras.layers import Dense # build our layers library

def build_classifier2():

    classifier = Sequential() # initialize neural network

    classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1])) # we use dimension of x_train as input

    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu')) # we use 4 nodes in second layer

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) # if we use sigmoid function it means we add output layer

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # we will use accuracy as metrics

    return classifier



classifier = KerasClassifier(build_fn = build_classifier2, epochs = 150) # epochs means that is number of iteration 

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 4)

mean = accuracies.mean()

variance = accuracies.std()

print("Accuracy mean: "+ str(mean))

print("Accuracy variance: "+ str(variance))
data_temproray.loc[-1] = [2, 4, 150,  0.6640782611279521]  # adding a row

data_temproray.index = data_temproray.index + 1  # shifting index

data_temproray = data_temproray.sort_index()  # sorting by index



data_temproray.loc[-1] = [3, 3, 100, 0.492614638565769]  # adding a row

data_temproray.index = data_temproray.index + 1  # shifting index

data_temproray = data_temproray.sort_index()  # sorting by index



data_temproray.loc[-1] = [3, 4, 150, 0.49022363005586056]  # adding a row

data_temproray.index = data_temproray.index + 1  # shifting index

data_temproray = data_temproray.sort_index()  # sorting by index



data_temproray.loc[-1] = [2, 4, 150, 0.7318890200427076]  # adding a row

data_temproray.index = data_temproray.index + 1  # shifting index

data_temproray = data_temproray.sort_index()  # sorting by index



data_temproray.loc[-1] = [2, 5, 150, 0.7341438597594337]  # adding a row

data_temproray.index = data_temproray.index + 1  # shifting index

data_temproray = data_temproray.sort_index()  # sorting by index



data_temproray.loc[-1] = [2, 4, 200, 0.7310445842269522]  # adding a row

data_temproray.index = data_temproray.index + 1  # shifting index

data_temproray = data_temproray.sort_index()  # sorting by index



data_temproray
data_temproray.accuracy = data_temproray.accuracy.apply(lambda x: x*100)
data_temproray
# plotly

#import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go





international_color = [float(each) for each in data_temproray.epochs]

data2 = [

    {

        'y':data_temproray.CV,

        'x': data_temproray.ANN_Num,

        'mode': 'markers',

        'marker': {

            'color': international_color,

            'size': data_temproray.accuracy,

            'showscale': True

        },

        "text" :  data_temproray.accuracy    

    }

]

iplot(data2)