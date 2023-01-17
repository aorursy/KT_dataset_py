import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cv2 import imread, resize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
path = '../input/horses-or-humans-dataset/horse-or-human'
classes = np.array(os.listdir(path + '/train'))
print(dict(zip(classes, [0,1])))
train_X = []
train_Y = []

for i, j in enumerate(classes):
    folder = path + '/train/' + str(j)
    for image in os.listdir(folder):
        img = imread((os.path.join(folder, image)), 1)
        img = resize(img, (100, 100)).flatten()
        train_X.append(img)
        train_Y.append(i)
test_X = []
test_Y = []

for i, j in enumerate(classes):
    folder = path + '/validation/' + str(j)
    for image in os.listdir(folder):
        img = imread((os.path.join(folder, image)), 1)
        img = resize(img, (100, 100)).flatten()
        test_X.append(img)
        test_Y.append(i)
MLP1 = MLPClassifier(hidden_layer_sizes = (1000, 500), max_iter = 300, activation = 'logistic').fit(train_X, train_Y)
predictionMLP1 = MLP1.predict(test_X)
MLP1.score(test_X, test_Y)
print(classification_report(test_Y, predictionMLP1))
MLP2 = MLPClassifier(hidden_layer_sizes = (1000, 500), max_iter = 300, activation = 'relu').fit(train_X, train_Y)
predictionMLP2 = MLP2.predict(test_X)
MLP2.score(test_X, test_Y)
print(classification_report(test_Y, predictionMLP2))
MLP3 = MLPClassifier(hidden_layer_sizes = (1000, 500), max_iter = 300, activation = 'tanh').fit(train_X, train_Y)
predictionMLP3 = MLP3.predict(test_X)
MLP3.score(test_X, test_Y)
print(classification_report(test_Y, predictionMLP3))
MLP4 = MLPClassifier(hidden_layer_sizes = (1000, 500, 250, 100, 50), max_iter = 300, activation = 'logistic').fit(train_X, train_Y)
predictionMLP4 = MLP4.predict(test_X)
MLP4.score(test_X, test_Y)
print(classification_report(test_Y, predictionMLP4))