from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import os

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

from glob import glob

import seaborn as sns

from PIL import Image

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
print(os.listdir('../input/data/'))
folder_benign_train = '../input/data/train/benign'

folder_malignant_train = '../input/data/train/malignant'



folder_benign_test = '../input/data/test/benign'

folder_malignant_test = '../input/data/test/malignant'



read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))



# Load in training pictures 

ims_benign = [read(os.path.join(folder_benign_train, filename)) for filename in os.listdir(folder_benign_train)]

X_benign = np.array(ims_benign, dtype='uint8')

ims_malignant = [read(os.path.join(folder_malignant_train, filename)) for filename in os.listdir(folder_malignant_train)]

X_malignant = np.array(ims_malignant, dtype='uint8')



# Load in testing pictures

ims_benign = [read(os.path.join(folder_benign_test, filename)) for filename in os.listdir(folder_benign_test)]

X_benign_test = np.array(ims_benign, dtype='uint8')

ims_malignant = [read(os.path.join(folder_malignant_test, filename)) for filename in os.listdir(folder_malignant_test)]

X_malignant_test = np.array(ims_malignant, dtype='uint8')



# Create labels

y_benign = np.zeros(X_benign.shape[0])

y_malignant = np.ones(X_malignant.shape[0])



y_benign_test = np.zeros(X_benign_test.shape[0])

y_malignant_test = np.ones(X_malignant_test.shape[0])





# Merge data 

X_train = np.concatenate((X_benign, X_malignant), axis = 0)

y_train = np.concatenate((y_benign, y_malignant), axis = 0)



X_test = np.concatenate((X_benign_test, X_malignant_test), axis = 0)

y_test = np.concatenate((y_benign_test, y_malignant_test), axis = 0)



# Shuffle data

s = np.arange(X_train.shape[0])

np.random.shuffle(s)

X_train = X_train[s]

y_train = y_train[s]



s = np.arange(X_test.shape[0])

np.random.shuffle(s)

X_test = X_test[s]

y_test = y_test[s]
print(X_test[1])

plt.imshow(X_test[1], interpolation='nearest')

plt.show()
X_train = X_train/255

X_test = X_test/255

print(X_test[1])
model = SVC()

model.fit(X_train.reshape(X_train.shape[0],-1), y_train)
y_pred = model.predict(X_test.reshape(X_test.shape[0],-1))
print(accuracy_score(y_test, y_pred))
logreg = LogisticRegression(C= 1.0, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=100,

                   multi_class='warn', n_jobs=None, penalty='l2',

                   random_state=None, solver='warn', tol=0.0001, verbose=0,

                   warm_start=False)

logreg.fit(X_train.reshape(X_train.shape[0],-1), y_train)
y_pred = logreg.predict(X_test.reshape(X_test.shape[0],-1))
print(accuracy_score(y_test, y_pred))
import tensorflow as tf

X_Train = tf.keras.utils.normalize(X_train)

X_Test = tf.keras.utils.normalize(X_test)
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(128,(3,3), input_shape = X_Train.shape[1:] ,activation = tf.nn.relu ))

model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=None))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(64,activation=tf.nn.relu))

model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(32,activation=tf.nn.relu))

model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Dense(2,activation=tf.nn.softmax))
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
X_Train.shape
model.fit(X_Train, y_train, epochs = 20, verbose=2, batch_size=32, validation_split = 0.1)
y_pred = model.predict(X_Test)
yp =[]

for i in range(0,660):

    if y_pred[i][0] >= 0.5:

        yp.append(0)

    else:

        yp.append(1)
print(accuracy_score(y_test, yp))