import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
import fnmatch
import cv2
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
img_dog=Image.open("../input/catpic/cat.10.jpg")
img_cat=Image.open("../input/dogpic/dog.1.jpg")

fig, ax = plt.subplots(1,2)
ax[0].imshow(img_dog);
ax[1].imshow(img_cat);
filename = "cats" # filename in my pycharm
# This is a function for finding images in my file 
def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename
cat_files=[] # empty list 
for filename in find_files(filename,"*jpg"): # Defined function begin to magic show
    file,ext = os.path.split(filename)
    img = Image.open(filename)
    img = img.resize((64,64), Image.ANTIALIAS) # Arrange images size 64x64
    data = np.array( img,np.uint8) # Convert images to matrix
    data=data.sum(axis=2) # This code deacrease column two from three for each array
    np.save("cat",data)
    c = np.load("cat.npy")
    cat_files.append(c)  
np.asarray(cat_files) # Convert list to array
np.save("cat1",cat_files) # final form 
x1 = np.load("../input/images/cat1.npy")
x2 =np.load("../input/images/dog1.npy")

xc = np.concatenate((x1,x2),axis=0)

print(xc.shape)

img_size = 64
plt.subplot(1,2,1)
plt.imshow(xc[1].reshape(img_size,img_size))
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(xc[300].reshape(img_size,img_size))
plt.axis("off")
plt.show()
x1 = np.load("../input/images/cat1.npy")
x2 =np.load("../input/images/dog1.npy")

x = np.concatenate((x1,x2),axis=0)

z = np.zeros(300) # Create 300 zeros for 300 cat file
o = np.ones(300)  # Create 300 ones for 300 dog file
y = np.concatenate((z,o),axis=0).reshape(x.shape[0],1) # do vector
print("x shape",x.shape)
print("y shape",y.shape)

#Normalization
x = (x-np.min(x))/(np.max(x)-np.min(x))

from sklearn.model_selection import train_test_split # from sklearn module, create train and test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.45,random_state=42) # %45 will be test data
print("x train shape",x_train.shape)
print("x test shape",x_test.shape)
print("y train shape",y_train.shape)
print("y test shape",y_test.shape)
# now we have 3 dimensional input array x so we need to make it flatten (2d) in order to use as input for our fist deep learning model.

number_of_train = x_train.shape[0]
number_of_test = x_test.shape[0]

x_train_flat = x_train.reshape(number_of_train,x_train.shape[1]*x_train.shape[2])
x_test_flat = x_test.reshape(number_of_test,x_test.shape[1]*x_test.shape[2]) # now in two dimension
print("x train flat",x_train_flat.shape)
print("x test flat",x_test_flat.shape)
# Transpose to our test and train datas for matrix calculation
x_train = x_train_flat.T
y_train = y_train.T
x_test  = x_test_flat.T
y_test  = y_test.T

print("x train shape",x_train.shape)
print("x test shape",x_test.shape)
print("y train shape",y_train.shape)
print("y test shape",y_test.shape)
print("x train :",x_train)
print("y train: ",y_train)
from sklearn import linear_model
lr = linear_model.LogisticRegression(random_state =42,max_iter= 15000)
print("test accuracy: {} ".format(lr.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))
print("train accuracy: {} ".format(lr.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))
# Define sigmoid function to predict y values between 0 and 1 
def sigmoid(z):
    y_head=1/(1+np.exp(-z))
    return y_head
# Define the inital parameters 
def initialize_parameters_and_layer_sizes_NN(x_train, y_train):
    parameters = {"weight1": np.random.randn(3,x_train.shape[0]) * 0.1,
                  "bias1": np.zeros((3,1)),
                  "weight2": np.random.randn(y_train.shape[0],3) * 0.1,
                  "bias2": np.zeros((y_train.shape[0],1))}
    return parameters
# Begin to process with initial parameters
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
def compute_cost_NN(A2, Y, parameters):
    logprobs = np.multiply(np.log(A2),Y)
    cost = -np.sum(logprobs)/Y.shape[1]
    return cost
# Bacward propagation
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
# Update parameters with information which came from backward process
def update_parameters_NN(parameters, grads, learning_rate = 0.001):
    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"],
                  "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],
                  "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],
                  "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}
    
    return parameters
# Create prediction between 0 and 1
def predict_NN(parameters,x_test):
    
    A2, cache = forward_propagation_NN(x_test,parameters)
    Y_prediction = np.zeros((1,x_test.shape[1]))

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
        
        if i % 10000 == 0:
            cost_list.append(cost)
            index_list.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    plt.plot(index_list,cost_list)
    plt.xticks(index_list,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    
    y_prediction_test = predict_NN(parameters,x_test)
    y_prediction_train = predict_NN(parameters,x_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    return parameters

parameters = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=100000)
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator= lr, X= x_train.T, y=y_train.T,cv=10)
print("average accuracy" ,np.mean(accuracy))
print("average standard deviation",np.std(accuracy))

lr.fit(x_train.T,y_train.T)
print("test accuracy:",lr.score(x_test.T,y_test.T))
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(x_train.T,y_train.T)
print("Random Forest Classification Test Score:",rf.score(x_test.T,y_test.T))

import seaborn as sns
print(x_test.shape)

y_prediction = rf.predict(x_test.T)
y_true = y_test.T

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_prediction)


f, ax= plt.subplots(figsize=(6,6))
sns.heatmap(cm,annot= True,linewidths=0.4,linecolor="red",fmt= ".0f",ax=ax)
plt.xlabel("Prediction number of y")
plt.ylabel("True number of y")
plt.show()