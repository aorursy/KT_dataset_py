# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #data visualization





from PIL import Image

import cv2 

from tqdm import tqdm 

# Last three are for import and tidying data

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
validation_horses = "../input/horses-or-humans-dataset/horse-or-human/horse-or-human/validation/horses"

validation_humans = "../input/horses-or-humans-dataset/horse-or-human/horse-or-human/validation/humans"

train_horses = "../input/horses-or-humans-dataset/horse-or-human/horse-or-human/train/horses"

train_humans = "../input/horses-or-humans-dataset/horse-or-human/horse-or-human/train/humans"



image_size = 64



X1 = []

X2 = []

X3 = []

X4 = []



for image in tqdm(os.listdir(validation_horses)):

    p = os.path.join(validation_horses, image)

    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE) 

    img = cv2.resize(img, (image_size, image_size)).flatten() 

    X1.append(img) # list of images

    imgshow1 = np.asarray(img) # just taking one image for visualize

    

for image in tqdm(os.listdir(validation_humans)):

    p = os.path.join(validation_humans, image)

    img2 = cv2.imread(p, cv2.IMREAD_GRAYSCALE) 

    img2 = cv2.resize(img2, (image_size, image_size)).flatten() 

    X2.append(img2)

    imgshow2 = np.asarray(img2)

    

for image in tqdm(os.listdir(train_horses)):

    p = os.path.join(train_horses, image)

    img3 = cv2.imread(p, cv2.IMREAD_GRAYSCALE) 

    img3 = cv2.resize(img3, (image_size, image_size)).flatten() 

    X3.append(img3)

    imgshow3 = np.asarray(img3)

    

for image in tqdm(os.listdir(train_humans)):

    p = os.path.join(train_humans, image)

    img4 = cv2.imread(p, cv2.IMREAD_GRAYSCALE) 

    img4 = cv2.resize(img4, (image_size, image_size)).flatten() 

    X4.append(img4)

    imgshow4 = np.asarray(img4)

    

# Convert to array    

X1 = np.asarray(X1)    

X2 = np.asarray(X2)

X3 = np.asarray(X3)    

X4 = np.asarray(X4)    

plt.figure(figsize=(6, 6))

plt.subplot(2, 2, 1)

plt.imshow(imgshow1.reshape(image_size, image_size))

plt.axis('off')

plt.subplot(2, 2, 2)

plt.imshow(imgshow2.reshape(image_size, image_size))

plt.axis('off')

plt.subplot(2, 2, 3)

plt.imshow(imgshow3.reshape(image_size, image_size))

plt.axis('off')

plt.subplot(2, 2, 4)

plt.imshow(imgshow4.reshape(image_size, image_size))

plt.axis('off')

plt.show()
x = np.concatenate((X1,X3,X2,X4), axis = 0)



zero = np.zeros(X1.shape[0] + X3.shape[0])

one = np.ones(X2.shape[0] + X4.shape[0])



y = np.concatenate((zero,one), axis = 0).reshape(-1,1)

print("x shape :", x.shape)

print("y shape :", y.shape)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)



print("x_train shape :",x_train.shape)

print("x_test shape :",x_test.shape)

print("y_train shape :",y_train.shape)

print("y_test shape :",y_test.shape)
def initializeParameters(x_train, y_train):

    parameters = {"weight1" : np.random.randn(4, x_train.shape[1]) * 0.1, 

                  "bias1" : np.zeros((4,1)) ,

                  "weight2" : np.random.randn(1, 4) * 0.1,

                  "bias2" : np.zeros((1,1))}

    return parameters



def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head



def forwardPropagation(x_train, parameters):

    z1 = np.dot(parameters["weight1"], x_train.T) + parameters["bias1"]

    a1 = np.tanh(z1)

    z2 = np.dot(parameters["weight2"], a1) + parameters["bias2"]

    a2 = sigmoid(z2)

    

    spinoff = {"z1" : z1,

               "a1" : a1,

               "z2" : z2,

               "a2" : a2}

    return a2, spinoff



def calculateCost(a2, y_train):

    k = np.multiply(np.log(a2), y_train.T)

    cost = -np.sum(k) / y_train.shape[0]

    

    return cost



def backwardPropagation(parameters, spinoff, x, y):

    

    dz2 = spinoff["a2"].T-y

    dw2 = np.dot(dz2.T,spinoff["a1"].T)/x.shape[0]

    db2 = np.sum(dz2.T,axis =1,keepdims=True)/x.shape[0]

    dz1 = np.dot(parameters["weight2"].T,dz2.T)*(1 - np.power(spinoff["a1"], 2))

    dw1 = np.dot(dz1,x)/x.shape[0]

    db1 = np.sum(dz1,axis =1,keepdims=True)/x.shape[0]

    

    gradients = {"dWeight1": dw1,

                 "dBias1": db1,

                 "dWeight2": dw2,

                 "dBias2": db2}

    return gradients



def update(param, grad, lr):

    param = {"weight1" : param["weight1"] - lr * grad["dWeight1"],

             "bias1"   : param["bias1"]   - lr * grad["dBias1"],

             "weight2" : param["weight2"] - lr * grad["dWeight2"],                             

             "bias2"   : param["bias2"]   - lr * grad["dBias2"]}  

                                         

    return param



def predict(parameters, x_test):

    a2, spinoff = forwardPropagation(x_test, parameters)

    predicted = np.zeros((x_test.shape[0], 1))

    

    for i in range(a2.shape[1]):

        if a2[0, i] < 0.5 :

            predicted[i, 0] = 0

        else:

            predicted[i, 0] = 1

            

    return predicted    
def annModel(x_train, x_test, y_train, y_test, iterationNumber, learningRate):

    

    costList = []

    iterationList = []

    

    parameters = initializeParameters(x_train, y_train)

    

    for i in range(iterationNumber):

        a2, spinoff = forwardPropagation(x_train, parameters)

        cost = calculateCost(a2, y_train)

        gradients = backwardPropagation(parameters, spinoff, x_train, y_train)

        parameters = update(parameters, gradients, learningRate)

        

        if i % 500 == 0:

            costList.append(cost)

            iterationList.append(i)

            print("Cost after {}. iteration : {}".format(i, cost))

            

        

    plt.plot(iterationList, costList)

    plt.xticks(iterationList, rotation='vertical')

    plt.xlabel("Number of iteration")

    plt.ylabel("Cost")

    plt.show()

       

    trainPrediction = predict(parameters, x_train)

    testPrediction  = predict(parameters, x_test)

    

    print("Train accuracy : {}%".format(100 - np.mean(np.abs(trainPrediction - y_train)) * 100))

    print("Test accuracy : {}%".format(100 - np.mean(np.abs(testPrediction - y_test)) * 100))     

    

    return parameters   
parameters = annModel(x_train, x_test, y_train, y_test, iterationNumber = 2500, learningRate = 0.008) 
from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import EarlyStopping



earlyStop = EarlyStopping(patience = 3)



n_cols = x_train.shape[1]



ann = Sequential()

ann.add(Dense(units = 50, kernel_initializer = "uniform", activation = "relu", input_dim = (n_cols)))

ann.add(Dense(units = 20, kernel_initializer = "uniform", activation = "relu"))

ann.add(Dense(units = 20, kernel_initializer = "uniform", activation = "relu"))

ann.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

ann.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])



ann.fit(x_train, y_train, epochs = 100, callbacks = [earlyStop])
trainPrediction2 = ann.predict(x_train)

testPrediction2 = ann.predict(x_test)



print("Train accuracy : {}%".format(100 - np.mean(np.abs(trainPrediction2 - y_train)) * 100))

print("Test accuracy : {}%".format(100 - np.mean(np.abs(testPrediction2 - y_test)) * 100))     