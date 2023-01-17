# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import csv

import scipy
from sklearn.model_selection import KFold

import keras
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,BatchNormalization, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from scipy import stats


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import math
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv (r'/kaggle/input/digit-recognizer/train.csv')
df_test = pd.read_csv (r'/kaggle/input/digit-recognizer/test.csv')

X_train = df_train.to_numpy()
Y_train = X_train[:,0].reshape(X_train.shape[0],1)
X_train = X_train[:,1:]


X_train = X_train.T
Y_train = Y_train.T

X_train = X_train / 255

X_test = df_test.to_numpy()
X_test = X_test.T
X_test = X_test / 255

print(X_train.shape)
print(Y_train.shape)
    
Y_train_final = np.zeros([10, Y_train.shape[1]])
for i in range(Y_train.shape[1]):
    C = np.zeros([10,1])
    x = np.squeeze(Y_train[:,i])
    C[x,:] = 1
    C = C.reshape(10,)
    Y_train_final[:,i] = C
    
Y_train = Y_train_final
print(Y_train.shape)
def relu(Z):
    
    A = np.maximum(0, Z)
    cache = Z
    
    return A,cache

def softmax(Z):
    
    t = np.exp(Z - np.max(Z))
    A = t / np.sum(t, axis = 0, keepdims = True)
    cache = Z
    
    return A, cache


def initialize_parameters(layer_dims):
    
    L = len(layer_dims)
    parameters = {}
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters

def linear_forward(A, W, b):
    
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    
    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)
        
    cache = (linear_cache, activation_cache)
    
    return A, cache

def L_model_forward(X, parameters):
    
    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
                                             
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "softmax")
    caches.append(cache)
     
    return AL, caches
def cost_function(AL, Y_train, parameters, lambd):
    
    m = Y_train.shape[1]
    L = len(parameters) // 2
    
    cost = - np.sum((Y_train * np.log(AL)), axis = 0, keepdims = True)
    cost = (1/m) * np.sum(cost, axis = 1, keepdims = True)
    
    reg_term = 0
    
    for l in range(1, L+1):
        reg_term = reg_term + np.sum(np.square(parameters['W' + str(l)]))
    
    reg_term = (lambd/(2*m)) * reg_term
    cost = float(np.squeeze(cost))
    
    cost = cost + reg_term
    return cost
    
def relu_backward(dA, activation_cache):
    
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    # When z <= 0, we should set dz to 0 as well.
    dZ[Z <= 0] = 0
    
    return dZ

def softmax_backward(Y_train, AL):
    
    dZ = AL - Y_train
    
    return dZ

def linear_backward(dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation, Y_train, AL):
    
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "softmax":
        #print(Y_train.shape[1])
        dZ = softmax_backward(Y_train, AL)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    return dA_prev, dW, db


def L_model_backward(AL, Y_train, caches):
    
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    #print(m)
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(0, current_cache, "softmax", Y_train, AL)
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu", Y_train, AL)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        
    return grads

def update_parameters(parameters, grads, learning_rate, lambd, m):
    
    L = len(parameters) // 2
    
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * (grads["dW" + str(l+1)] + ((lambd/m) * (parameters["W" + str(l+1)])))
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters
layer_dims = [X_train.shape[0], 512, 264, 10]
def final_NN_model(X_train, Y_train, layer_dims, learning_rate = 0.075, num_iterations = 3000, mini_batch_size = 64, print_cost=True):
    
    costs = []
    parameters = initialize_parameters(layer_dims)
    no_of_complete_mini_batches = X_train.shape[1] // mini_batch_size
    
    lambd = 0.08
    m = mini_batch_size
    
    for i in range(0, num_iterations):
        
        for j in range(no_of_complete_mini_batches):
            
            AL, caches = L_model_forward(X_train[:,mini_batch_size*j:mini_batch_size*(j+1)], parameters)
            
            cost = cost_function(AL, Y_train[:,mini_batch_size*j:mini_batch_size*(j+1)], parameters, lambd)
            #print(Y_train[:,mini_batch_size*j:mini_batch_size*(j+1)].shape)
            grads = L_model_backward(AL, Y_train[:,mini_batch_size*j:mini_batch_size*(j+1)], caches)
            
            parameters = update_parameters(parameters, grads, learning_rate, lambd, m)
        
            if print_cost and j % 50 == 0:
                print ("Cost after iteration %i, on minibatch %i :%f" %(i, j, cost))
            if print_cost and j % 50 == 0:
                costs.append(cost)
            
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per fifty minibatch)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
        
    return parameters

X_train_actual = X_train[:,:37024]
X_train_cv = X_train[:,37024:]



Y_train_actual = Y_train[:,:37024]
Y_train_cv = Y_train[:,37024:]



parameters = final_NN_model(X_train_actual, Y_train_actual, layer_dims, num_iterations = 3, mini_batch_size = 32, print_cost = True)
def predict(parameters, X):
    
    AL, caches = L_model_forward(X, parameters)
    
    return AL


pred = predict(parameters, X_train_cv)

def final_pred(pred):
    
    pred_final = np.zeros([pred.shape[0], pred.shape[1]])

    for i in range(pred.shape[1]):
        k = 0
        index = 0
        C = np.zeros([10,1])
        for j in range(pred.shape[0]):
            l = np.squeeze(pred[j,i])
            if(l>k):
                index = j
                k = l
        C[index,:] = 1
        C = C.reshape(10,)
        pred_final[:,i] = C
        
    return pred_final
train_final_pred = final_pred(pred)
err = Y_train_cv - train_final_pred
counter = 0
for i in range(err.shape[1]):
    for j in range(err.shape[0]):
        if(np.squeeze(err[j,i]) == 1 or np.squeeze(err[j,i]) == -1):
            counter+=1
        break            
print("train accuracy on cross validation set: ")
print((4976-counter)/4976)
test_pred = predict(parameters, X_test)
test_final_pred = final_pred(test_pred)

test_pred_last = np.zeros([1, test_final_pred.shape[1]])

for i in range(test_final_pred.shape[1]):
    index = 0
    for j in range(test_final_pred.shape[0]):
        if(np.squeeze(test_final_pred[j,i]) == 1):
            index = j
            test_pred_last[:,i] = index
            break
    
test_pred_last = test_pred_last.T
print(test_pred_last.shape)
final_csv = []
csv_title = ['ImageId', 'Label']
final_csv.append(csv_title)

for i in range(test_pred_last.shape[0]):
    image_id = i + 1
    label = np.squeeze(test_pred_last[i,:])
    temp = [image_id, label]
    final_csv.append(temp)
    
print(len(final_csv))

with open('submission_csv20.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(final_csv)
file.close()
df_train = pd.read_csv (r'/kaggle/input/digit-recognizer/train.csv')
df_test = pd.read_csv (r'/kaggle/input/digit-recognizer/test.csv')


labels = df_train["label"].values.tolist()
labels = np.array(labels)

n_classes = 10
labels = keras.utils.to_categorical(labels)


df_train = df_train.drop(["label"], axis = 1)
data = df_train.values.tolist() # converting image data to list
data = np.array(data)

#print(data.shape)


X_train_cnn = data.reshape(len(data), 28, 28, 1)

print(X_train_cnn.shape)
print(labels.shape)
index = 1589
label = np.squeeze(labels[index])

pixels = data[index,:]
pixels = np.array(pixels, dtype='uint8')
pixels = pixels.reshape((28, 28))

plt.title('Label in one hot is {label}'.format(label=label))
plt.imshow(pixels, cmap='gray')
plt.show()
X_train_cnn = X_train_cnn/255.0 #normalizing the inputs 
def define_model():
    cnn_model = Sequential()
    cnn_model.add(Conv2D(32, kernel_size = [3,3], activation = 'relu', input_shape = (28,28,1)))
    cnn_model.add(MaxPooling2D(pool_size = [2,2], strides = 2))
    cnn_model.add(Conv2D(64, kernel_size = [3,3], activation = 'relu'))
    cnn_model.add(Conv2D(64, kernel_size = [3,3], activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = [2,2], strides = 2))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(100, activation = 'relu'))
    cnn_model.add(Dense(10, activation = 'softmax'))
    #print("CNN MODEL :-")
    #cnn_model.summary()
    cnn_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
      
    return cnn_model
def define_model_batchnorm():
    cnn_model = Sequential()
    cnn_model.add(Conv2D(32, kernel_size = [3,3], activation = 'relu', input_shape = (28,28,1)))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(pool_size = [2,2], strides = 2))
    cnn_model.add(Conv2D(64, kernel_size = [3,3], activation = 'relu'))
    cnn_model.add(Conv2D(64, kernel_size = [3,3], activation = 'relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(pool_size = [2,2], strides = 2))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(100, activation = 'relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dense(10, activation = 'softmax'))
    #print("CNN MODEL :-")
    #cnn_model.summary()
    cnn_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
      
    return cnn_model
def define_model_accurate():
    
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = 5, strides = 1, activation = 'relu', input_shape = (28,28,1), kernel_regularizer=regularizers.l2(0.0005)))
    model.add(Conv2D(filters = 32, kernel_size = 5, strides = 1, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = 2, strides = 2))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 64, kernel_size = 3, strides = 1, activation = 'relu', kernel_regularizer=regularizers.l2(0.0005)))
    model.add(Conv2D(filters = 64, kernel_size = 3, strides = 1, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = 2, strides = 2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(128, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(84, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation = 'softmax'))
    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model
model = define_model()

model_hist = model.fit(X_train_cnn, labels, epochs=10, batch_size=32, validation_split = 0.1)

model2 = define_model()
model2_hist = model2.fit(X_train_cnn, labels, epochs=10, batch_size=16, validation_split = 0.1)
model3 = define_model_batchnorm()
model3_hist = model3.fit(X_train_cnn, labels, epochs=10, batch_size=32, validation_split = 0.1)

model4 = define_model_batchnorm()
model4_hist = model4.fit(X_train_cnn, labels, epochs=10, batch_size=16, validation_split = 0.1)
model5 = define_model_accurate()
model5_hist = model5.fit(X_train_cnn, labels, epochs=30, batch_size=32, validation_split = 0.1)

plt.plot(model2_hist.history["accuracy"])
plt.plot(model2_hist.history["val_accuracy"])
plt.title("Training vs Validation Accuracy (CNN Model 2)")
plt.legend(["Training","Validation"], loc = 'lower right')
plt.show()
plt.plot(model5_hist.history["accuracy"])
plt.plot(model5_hist.history["val_accuracy"])
plt.title("Training vs Validation Accuracy (CNN Model 5)")
plt.legend(["Training","Validation"], loc = 'lower right')
plt.show()
testdata = df_test.values.tolist() #Load and reshape test data
testdata = np.array(testdata)
testdata_reshaped = testdata.reshape(testdata.shape[0], 28, 28, 1)
testdata_reshaped = testdata_reshaped.astype('float')/255
print(testdata_reshaped.shape)
Y_test = model5.predict_classes(testdata_reshaped)
final_csv = [] #Making final output file
csv_title = ['ImageId', 'Label']
final_csv.append(csv_title)

for i in range(28000):
    image_id = i + 1
    label = Y_test[i]
    temp = [image_id, label]
    final_csv.append(temp)
    
print(len(final_csv))

with open('submission_csv130.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(final_csv)
file.close()
