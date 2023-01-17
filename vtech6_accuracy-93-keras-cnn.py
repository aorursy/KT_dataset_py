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
train_data = pd.read_csv(r'/kaggle/input/fashionmnist/fashion-mnist_train.csv')

test_data = pd.read_csv(r'/kaggle/input/fashionmnist/fashion-mnist_test.csv')
from keras.utils import to_categorical

y_train = train_data['label']

y_train = to_categorical(y_train, num_classes=10)

X_train = train_data.drop('label', axis=1)



y_test = test_data['label']

y_test = to_categorical(y_test, num_classes=10)

X_test = test_data.drop('label', axis=1)
X_train_p = np.array((X_train/255)**2).reshape(-1, 28, 28, 1)

X_test_p = np.array((X_test/255)**2).reshape(-1, 28, 28, 1)



import matplotlib.pyplot as plt

plt.imshow(X_train_p[1].reshape(28, 28))

plt.imshow(X_test_p[0].reshape(28, 28))
import tensorflow as tf

from tensorflow import keras



model = keras.models.Sequential()



model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same'))

model.add(keras.layers.BatchNormalization())



model.add(keras.layers.Dropout(0.1))



model.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 2), activation='tanh', padding='same'))

model.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 2), activation='tanh', padding='same'))

model.add(keras.layers.BatchNormalization())



model.add(keras.layers.Conv2D(filters=16, kernel_size=(4, 1), activation='tanh', padding='same'))

model.add(keras.layers.Conv2D(filters=16, kernel_size=(4, 1), activation='tanh', padding='same'))

model.add(keras.layers.BatchNormalization())



model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(keras.layers.MaxPooling2D(strides=(2, 2), padding='valid'))

model.add(keras.layers.BatchNormalization())





model.add(keras.layers.Flatten())



model.add(keras.layers.Dense(700, activation='relu'))

model.add(keras.layers.Dense(500, activation='relu'))

model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.BatchNormalization())



model.add(keras.layers.Dense(300, activation='tanh'))

model.add(keras.layers.Dense(200, activation='tanh'))

model.add(keras.layers.BatchNormalization())



model.add(keras.layers.Dense(100, activation='softsign'))

model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(50, activation='tanh'))

model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])

model.fit(X_train_p, y_train, epochs=25, batch_size=500)
predictions = model.predict(X_test_p)
from sklearn.metrics import accuracy_score

y_test_args=[]

for i in y_test:

    y_test_args.append(i.argmax())

preds_args=[]

for i in predictions:

    preds_args.append(i.argmax())

accuracy = accuracy_score(y_test_args, preds_args)

accuracy
plt.imshow(X_train[1].reshape(28, 28))