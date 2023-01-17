# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import *
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head()
train.describe()
plt.figure(figsize=(15,15)) 
sns.countplot(train['label'])
train['label'].value_counts()
image_size = train.drop(columns = ['label']).copy().shape[1]
print ('image total pixels = ' + str(image_size))

image_width = image_height = np.ceil(np.sqrt(image_size))

print ('image_width = {0}\nimage_height = {1}'.format(image_width,image_height))
images = train.iloc[:,1:].values
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(4,3,i+1)
    plt.figure
    ex_image = images[i].reshape(int(image_width),int(image_height))
    plt.axis('off')
    plt.imshow(ex_image)
trainData, testData = train_test_split(train, test_size=0.2, shuffle=False)
trainData.describe()
images_train = trainData.iloc[:,1:].values
images_test = testData.iloc[:,1:].values
train_images = images_train.reshape(-1,28,28)
train_images.shape
test_images = images_test.reshape(-1,28,28)
test_images.shape
train_labels = trainData['label'].values
test_labels = testData['label'].values
mode = train['label'].mode()
print(mode)
y_pred = test['label'].copy()
y_pred.describe()
y_actual =[]
for i in range(len(test)):
    y_actual.append(mode)
print(len(y_actual))
score = accuracy_score(y_pred,y_actual)
print(score)
matrix = confusion_matrix(y_pred, y_actual)
print(matrix)
report = classification_report(y_pred, y_actual)
print(report)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
train_images_normalized = np.multiply(train_images.copy(), 1.0 / 255.0)
test_images_normalized = np.multiply(test_images.copy(), 1.0 / 255.0)
model.fit(train_images_normalized, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images_normalized, test_labels)
print('Test accuracy:', test_acc)
test_set = pd.read_csv('../input/test.csv')
test_set_np = test_set.values
test_set_np = np.multiply(test_set_np, 1.0 / 255.0)

print(test_set_np.shape)

predictions = model.predict(test_set_np.reshape(-1,28,28))
y_classes = predictions.argmax(axis=-1)
y_classes
np.savetxt('submission.csv', 
           np.c_[range(1,len(test_set)+1),y_classes], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')
