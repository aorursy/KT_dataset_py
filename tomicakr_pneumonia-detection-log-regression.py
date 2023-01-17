import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2 

import os 

import warnings

from random import shuffle 

from tqdm import tqdm 

from PIL import Image

warnings.filterwarnings('ignore')



in_path = '../input/chest-xray-pneumonia//chest_xray/chest_xray/'
image_size = 150



train_data_log = []

train_labels_log = []



for normal_image in tqdm(os.listdir(in_path + 'train' + '/NORMAL/')): 

    path = os.path.join(in_path + 'train' + '/NORMAL/', normal_image)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 

    if img is None:

        continue

    img = cv2.resize(img, (image_size, image_size)).flatten()   

    np_img=np.asarray(img)

    train_data_log.append(img)

    train_labels_log.append(0)

    

for pneumonia_image in tqdm(os.listdir(in_path + 'train' + '/PNEUMONIA/')): 

    path = os.path.join(in_path + 'train' + '/PNEUMONIA/', pneumonia_image)

    img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 

    if img2 is None:

        continue

    img2 = cv2.resize(img2, (image_size, image_size)).flatten() 

    np_img2=np.asarray(img2)

    train_data_log.append(img2)

    train_labels_log.append(1)
test_data_log = []

test_labels_log = []



for normal_image in tqdm(os.listdir(in_path + 'test' + '/NORMAL/')): 

    path = os.path.join(in_path + 'test' + '/NORMAL/', normal_image)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 

    if img is None:

        continue

    img = cv2.resize(img, (image_size, image_size)).flatten()   

    np_img=np.asarray(img)

    test_data_log.append(img)

    test_labels_log.append(0)

    

for pneumonia_image in tqdm(os.listdir(in_path + 'test' + '/PNEUMONIA/')): 

    path = os.path.join(in_path + 'test' + '/PNEUMONIA/', pneumonia_image)

    img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 

    if img2 is None:

        continue

    img2 = cv2.resize(img2, (image_size, image_size)).flatten() 

    np_img2=np.asarray(img2)

    test_data_log.append(img2)

    test_labels_log.append(1)
train_data = np.array(train_data_log).T

train_labels = np.array(train_labels_log).reshape(-1, 1).T

test_data = np.array(test_data_log).T

test_labels = np.array(test_labels_log).reshape(-1, 1).T



train_data = (train_data-np.min(train_data))/(np.max(train_data)-np.min(train_data))

test_data = (test_data-np.min(test_data))/(np.max(test_data)-np.min(test_data))



print("Train data shape: {}".format(train_data.shape))

print("Train labels shape: {}".format(train_labels.shape))

print("Test data shape: {}".format(test_data.shape))

print("Test labels shape: {}".format(test_labels.shape))
def sigm(x):

    return 1/(1+np.exp(-x))
def propagate(w,b,train_data,train_labels):

    y = sigm(np.dot(w.T,train_data) + b)

    loss = -train_labels*np.log(y)-(1-train_labels)*np.log(1-y)

    cost = (np.sum(loss))/train_data.shape[1]

    dW = (np.dot(train_data,((y-train_labels).T)))/train_data.shape[1]

    dB = np.sum(y-train_labels)/train_data.shape[1]

    return cost, {"dW": dW,"dB": dB}
def update(w, b, train_data, train_labels, learning_rate, max_iter):

    costs = []

    costs2 = []

    index = []

    

    for i in range(max_iter):

        cost, deltas = propagate(w,b,train_data,train_labels)

        costs.append(cost)

        w = w - learning_rate * deltas["dW"]

        b = b - learning_rate * deltas["dB"]

        

        if i % 50 == 0 and i != 0:

            costs2.append(cost)

            index.append(i)

            print("Iteration {} > cost = {}".format(i, cost))

    

    plt.plot(index,costs2)

    plt.xticks(index)

    plt.xlabel("Iteration")

    plt.ylabel("Cost")

    plt.show()

    return {"w": w,"b": b}, deltas, costs
def predict(w, b, test_data):

    z = sigm(np.dot(w.T, test_data)+b)

    pred = np.zeros((1, test_data.shape[1]))

    for i in range(z.shape[1]):

        if z[0,i] <= 0.5:

            pred[0,i] = 0

        else:

            pred[0,i] = 1

    return pred
def log_regression(train_data, train_labels, test_data, test_labels, learning_rate, max_iter):

    dim =  train_data.shape[0]

    w = np.full((dim,1),0.01)

    b = 0.0



    params, gradients, cost_list = update(w, b, train_data, train_labels, learning_rate, max_iter)

    

    test_labels_prediction = predict(params["w"],params["b"],test_data)

    train_labels_prediction = predict(params["w"],params["b"],train_data)

    

    print("Test Accuracy: {} %".format(round(100 - np.mean(np.abs(test_labels_prediction - test_labels)) * 100,2)))

    print("Train Accuracy: {} %".format(round(100 - np.mean(np.abs(train_labels_prediction - train_labels)) * 100,2)))
log_regression(train_data, train_labels, test_data, test_labels, learning_rate = 0.001, max_iter = 1000)