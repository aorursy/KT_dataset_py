# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

from keras.utils import to_categorical



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv(r'/kaggle/input/Kannada-MNIST/train.csv')
test_data = pd.read_csv(r'/kaggle/input/Kannada-MNIST/test.csv')
train_data.head()
test_data.describe()
X = train_data.drop('label', axis=1)

y = train_data['label']





X_test = test_data.drop('id', axis=1)

X_test = np.array(X_test/255.0)

X_test = X_test.reshape(-1, 28, 28,1)

X_train = np.array(X/255.0)

X_train = X_train.reshape(-1, 28, 28,1)

y_train = to_categorical(y, num_classes=10)



print (y.shape)

X_test[0]
model = keras.models.Sequential()



model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(keras.layers.Conv2D(filters=64, kernel_size=(4, 4), activation='relu'))

model.add(keras.layers.AveragePooling2D(pool_size=(3, 3),strides=(1, 1), padding='same'))

model.add(keras.layers.BatchNormalization())



model.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 2), activation='tanh', padding='same'))

model.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 2), activation='tanh', padding='valid'))

model.add(keras.layers.BatchNormalization())



model.add(keras.layers.Flatten())



model.add(keras.layers.Dense(700, activation='relu'))

model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.Dense(500, activation='relu'))

model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.BatchNormalization())



model.add(keras.layers.Dense(300, activation='tanh'))

model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.Dense(200, activation='tanh'))

model.add(keras.layers.BatchNormalization())



model.add(keras.layers.Dense(100, activation='softsign'))

model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(50, activation='relu'))

model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=1000, shuffle=True)

test_data.head()
test_pred = model.predict(X_test)



solution = []

for i in test_pred:

    solution.append(np.argmax(i))



solution[0]
output = pd.DataFrame({'id': test_data.id, 'label': solution})

output.to_csv('submission.csv', index=False)