# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import matplotlib.image as implt
from PIL import Image 
import seaborn as sns
import cv2 as cs2
import os

import warnings
warnings.filterwarnings('ignore')
train_data = "/kaggle/input/messy-vs-clean-room/images/train"
test_data = "/kaggle/input/messy-vs-clean-room/images/val"

train_messy_path = "/kaggle/input/messy-vs-clean-room/images/train/messy"
train_clean_path = "/kaggle/input/messy-vs-clean-room/images/train/clean"

test_messy_path = "/kaggle/input/messy-vs-clean-room/images/val/messy"
test_clean_path = "/kaggle/input/messy-vs-clean-room/images/val/clean"
#Visualization
category_names = os.listdir(train_data) # output: ['messy', 'clean'] -> listing into folder
nb_categories = len(category_names) # output: 2
train_images = []

for category in category_names:
    folder = train_data + "/" + category
    train_images.append(len(os.listdir(folder))) #sort of datas in folder
    
sns.barplot(y=category_names, x=train_images).set_title("Number Of Training Images Per Category");
test_images = []
for caregory in category_names:
    folder = test_data + "/" + folder
    test_images.append(len(os.listdir(test_data)))
    
sns.barplot(y=category_names, x=test_images).set_title("Number Of Test Images Per Category");
img1 = implt.imread("../input/messy-vs-clean-room/images/train/messy/0.png") #messy
img2 = implt.imread("../input/messy-vs-clean-room/images/train/clean/0.png") #clean

plt.subplot(1, 2, 1)
plt.title('messy')
plt.imshow(img1)       
plt.subplot(1, 2, 2)
plt.title('clan')
plt.imshow(img2)
plt.show()
img_size = 50
messy_train = []
clean_train = []
label = []

for i in os.listdir(train_messy_path): # all train messy images
    if os.path.isfile(train_data + "/messy/" + i): # check image in file
        messy = Image.open(train_data + "/messy/" + i).convert("L") # converting grey scale 
        messy = messy.resize((img_size,img_size), Image.ANTIALIAS) # resizing to 50,50
        messy = np.asarray(messy)/255 # bit format (RGB)
        messy_train.append(messy)
        label.append(1)
        
for i in os.listdir(train_clean_path): # all train clean images
    if os.path.isfile(train_data + "/clean/" + i): # check image in file
        clean = Image.open(train_data + "/clean/" + i).convert("L") # converting grey scale 
        clean = clean.resize((img_size,img_size), Image.ANTIALIAS) # resizing to 50,50
        clean = np.asarray(clean)/255 # bit format
        clean_train.append(clean)
        label.append(0)
x_train = np.concatenate((messy_train,clean_train),axis=0)
x_train_label = np.asarray(label)
x_train_label = x_train_label.reshape(x_train_label.shape[0],1)

print("messy:",np.shape(messy_train) , "clean:",np.shape(clean_train))
print("train_dataset:",np.shape(x_train), "train_values:",np.shape(x_train_label))
img_size = 50
messy_test = []
clean_test = []
label = []

for i in os.listdir(test_messy_path): # all train messy images
    if os.path.isfile(test_data + "/messy/" + i): # check image in file
        messy = Image.open(test_data + "/messy/" + i).convert("L") # converting grey scale 
        messy = messy.resize((img_size,img_size), Image.ANTIALIAS) # resizing to 50,50
        messy = np.asarray(messy)/255 # bit format
        messy_test.append(messy)
        label.append(1)
        
for i in os.listdir(test_clean_path): # all train clean images
    if os.path.isfile(test_data + "/clean/" + i): # check image in file
        clean = Image.open(test_data + "/clean/" + i).convert("L") # converting grey scale 
        clean = clean.resize((img_size,img_size), Image.ANTIALIAS) # resizing to 50,50
        clean = np.asarray(clean)/255 # bit format
        clean_test.append(clean)
        label.append(0)
x_test = np.concatenate((messy_test,clean_test),axis=0)
x_test_label = np.asarray(label)
x_test_label = x_test_label.reshape(x_test_label.shape[0],1)

print("messy:",np.shape(messy_test) , "clean:",np.shape(clean_test))
print("train_dataset:",np.shape(x_test), "train_values:",np.shape(x_test_label))
x = np.concatenate((x_train,x_test),axis=0) #train data
# x.shape: 
y = np.concatenate((x_train_label,x_test_label),axis=0)
x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]) # flatten 3D image array to 2D, count
print("images:",np.shape(x), "labels:",np.shape(y))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

print("Train Number: ", number_of_train)
print("Test Number: ", number_of_test)
x_train = X_train.T
x_test = X_test.T
y_train = y_train.T
y_test = y_test.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01) # np.full((row,column),value)
    b = 0.0
    return w,b

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -(1-y_train)*np.log(1-y_head)-y_train*np.log(y_head)
    cost = (np.sum(loss))/x_train.shape[1]  # x_train.shape[1]  is for scaling, x_train.shape[1] = 1090
    
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {
        "derivative_weight": derivative_weight,
        "derivative_bias": derivative_bias
    }
    return cost,gradients

def update(w, b, x_train, y_train, learning_rate,number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    # updating(learning) parameters is number_of_iteration times
    for i in range(number_of_iteration):
        
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 50 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
        # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation="vertical")
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list

def predict(w,b,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    y_prediction = np.zeros((1,x_test.shape[1]))
    
    # if z is bigger than 0.5, our prediction is human (y_head=1)
    # if z is smaller than 0.5, our prediction is horse (y_head=0)
    for i in range(z.shape[1]):
        if z[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    return y_prediction


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    # initialize
    dimension = x_train.shape[0] # 2500
    w,b = initialize_weights_and_bias(dimension)
    parameters, gradients, cost_list = update(w,b,x_train,y_train,learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)
    
    test_acc_lr = round((100 - np.mean(np.abs(y_prediction_test - y_test)) * 100),2)
    train_acc_lr = round((100 - np.mean(np.abs(y_prediction_train - y_train))*100),2)
    
    # Print train/test Errors
    print("train accuracy: %", train_acc_lr)
    print("test accuracy: %", test_acc_lr)
    return train_acc_lr, test_acc_lr
#you can adjust learning_rate and num_iteration to check how the result is affected
#(for learning rate, try exponentially lower values:0.001 etc.) 
train_acc_lr, test_acc_lr = logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.11, num_iterations = 2000)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
test_acc_logregsk = round(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)* 100, 2)
train_acc_logregsk = round(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)* 100, 2)
# with GridSearchCV
from sklearn.model_selection import GridSearchCV

grid = {
    "C": np.logspace(-4, 4, 20),
    "penalty": ["l1","l2"]
}
lg=LogisticRegression(random_state=42)
log_reg_cv=GridSearchCV(lg,grid,cv=10,n_jobs=-1,verbose=2)
log_reg_cv.fit(x_train.T,y_train.T)
print("accuracy: ", log_reg_cv.best_score_)
models = pd.DataFrame({
    'Model': ['LR without sklearn','LR with sklearn','LR with GridSearchCV' ],
    'Train Score': [train_acc_lr, train_acc_logregsk, "-"],
    'Test Score': [test_acc_lr, test_acc_logregsk, log_reg_cv.best_score_*100]
})
models.sort_values(by='Test Score', ascending=False)
# Reshaping
x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library

def build_classifier():
    classifier = Sequential() # initialize neural network
    # units: number of nodes 
    classifier.add(Dense(units= 8, kernel_initializer="uniform", activation="relu", input_dim=x_train.shape[1]))
    classifier.add(Dense(units= 4, kernel_initializer="uniform", activation="relu", input_dim=x_train.shape[1]))
    classifier.add(Dense(units= 1, kernel_initializer="uniform", activation="sigmoid")) # end node
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    
    
    
    return classifier
classifier = KerasClassifier(build_fn=build_classifier, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=5)
mean = accuracies.mean()
print("Accuracies: ", accuracies)
print("Accuracy mean: ",mean)