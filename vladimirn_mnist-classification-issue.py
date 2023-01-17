# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import cv2

import numpy as np

import matplotlib.pyplot as plt
train = np.loadtxt('/kaggle/input/digit-recognizer/train.csv', delimiter=',', skiprows=1)

test = np.loadtxt('/kaggle/input/digit-recognizer/test.csv', delimiter=',', skiprows=1)
# сохраняем разметку в отдельную переменную

train_label = train[:, 0]

# приводим размерность к удобному для обаботки виду

train_img = np.resize(train[:, 1:], (train.shape[0], 28, 28))

test_img = np.resize(test, (test.shape[0], 28, 28))
fig = plt.figure(figsize=(20, 10))

for i, img in enumerate(train_img[0:5], 1):

    subplot = fig.add_subplot(1, 7, i)

    plt.imshow(img, cmap='gray');

    subplot.set_title('%s' % train_label[i - 1]);
train_sobel_x = np.zeros_like(train_img)

train_sobel_y = np.zeros_like(train_img)

for i in range(len(train_img)):

    train_sobel_x[i] = cv2.Sobel(train_img[i], cv2.CV_64F, dx=1, dy=0, ksize=3)

    train_sobel_y[i] = cv2.Sobel(train_img[i], cv2.CV_64F, dx=0, dy=1, ksize=3)
test_sobel_x = np.zeros_like(test_img)

test_sobel_y = np.zeros_like(test_img)

for i in range(len(test_img)):

    test_sobel_x[i] = cv2.Sobel(test_img[i], cv2.CV_64F, dx=1, dy=0, ksize=3)

    test_sobel_y[i] = cv2.Sobel(test_img[i], cv2.CV_64F, dx=0, dy=1, ksize=3)
train_g, train_theta = cv2.cartToPolar(train_sobel_x, train_sobel_y)

test_g, test_theta = cv2.cartToPolar(test_sobel_x, test_sobel_y)
fig = plt.figure(figsize=(20, 10))

for i, img in enumerate(train_g[:5], 1):

    subplot = fig.add_subplot(1, 7, i)

    plt.imshow(img, cmap='gray');

    subplot.set_title('%s' % train_label[i - 1]);

    subplot = fig.add_subplot(3, 7, i)

    plt.hist(train_theta[i - 1].flatten(),

             bins=16, weights=train_g[i - 1].flatten())
# Гистограммы вычисляются с учетом длины вектора градиента

train_hist = np.zeros((len(train_img), 16))

for i in range(len(train_img)):

    hist, borders = np.histogram(train_theta[i],

                                 bins=16,

                                 range=(0., 2. * np.pi),

                                 weights=train_g[i])

    train_hist[i] = hist
test_hist = np.zeros((len(test_img), 16))

for i in range(len(test_img)):

    hist, borders = np.histogram(test_theta[i],

                                 bins=16,

                                 range=(0., 2. * np.pi),

                                 weights=test_g[i])

    test_hist[i] = hist
train_hist = train_hist / np.linalg.norm(train_hist, axis=1)[:, None]
test_hist = test_hist / np.linalg.norm(test_hist, axis=1)[:, None]
from sklearn.model_selection import train_test_split

y_train, y_val, x_train, x_val = train_test_split(

    train_label, train_hist, test_size=0.2,

    random_state=42)
centroids = np.zeros((10, train_hist.shape[1]), dtype=np.float32)

for i in range(10):

    centroids[i] = np.mean(x_train[y_train == i], axis=0)
pred_val = np.zeros_like(y_val)

for i in range(len(y_val)):

    distances = np.linalg.norm(centroids - x_val[i], axis=1)

    pred_val[i] = np.argmin(distances)
from sklearn.metrics import accuracy_score

print('Accuracy: %s' % accuracy_score(y_val, pred_val))
from sklearn.metrics import classification_report

print(classification_report(y_val, pred_val))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_val, pred_val))
pred_test = np.zeros(len(test_img), np.uint8)

for i in range(len(test_img)):

    distances = np.linalg.norm(centroids - test_hist[i], axis=1)

    pred_test[i] = np.argmin(distances)
fig = plt.figure(figsize=(20, 10))

for i, img in enumerate(test_img[0:5], 1):

    subplot = fig.add_subplot(1, 7, i)

    plt.imshow(img, cmap='gray');

    subplot.set_title('%s' % pred_test[i - 1]);
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train, y_train)

pred_val = model.predict(x_val)

from sklearn.metrics import accuracy_score

print('Accuracy: %s' % accuracy_score(y_val, pred_val))
pred_test = model.predict(test_hist)

fig = plt.figure(figsize=(20, 10))

for i, img in enumerate(test_img[0:5], 1):

    subplot = fig.add_subplot(1, 7, i)

    plt.imshow(img, cmap='gray');

    subplot.set_title('%s' % pred_test[i - 1]);
from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

clf_rf.fit(x_train, y_train)

pred_val = clf_rf.predict(x_val)

from sklearn.metrics import accuracy_score

print('Accuracy: %s' % accuracy_score(y_val, pred_val))
pred_test = model.predict(test_hist)

fig = plt.figure(figsize=(20, 10))

for i, img in enumerate(test_img[0:5], 1):

    subplot = fig.add_subplot(1, 7, i)

    plt.imshow(img, cmap='gray');

    subplot.set_title('%s' % pred_test[i - 1]);
from catboost import CatBoostClassifier

model = CatBoostClassifier(iterations=10000,

                           task_type="GPU",

                           devices='0:0')

model.fit(x_train, y_train, verbose = False)

pred_val = model.predict(x_val)

from sklearn.metrics import accuracy_score

print('Accuracy: %s' % accuracy_score(y_val, pred_val))
from sklearn.metrics import classification_report

print(classification_report(y_val, pred_val))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_val, pred_val))
pred_test = model.predict(test_hist)

fig = plt.figure(figsize=(20, 10))

for i, img in enumerate(test_img[0:5], 1):

    subplot = fig.add_subplot(1, 7, i)

    plt.imshow(img, cmap='gray');

    subplot.set_title('%s' % pred_test[i - 1]);
with open('submit.txt', 'w') as dst:

    dst.write('ImageId,Label\n')

    for i, p in enumerate(pred_test, 1):

        dst.write('%s,%s\n' % (i, int(p)))
!head submit.txt