# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import keras



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
y = df_train['label']

df_train.drop(['label'], axis=1, inplace=True)

df_train = df_train.values.reshape(-1,28,28,1)

df_test = df_test.values.reshape(-1,28,28,1)
x = df_train

x_predict = df_test
g = plt.imshow(x[0][:,:,0])
import keras

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

y = to_categorical(y, 10)
model = Sequential()

model.add(Conv2D(input_shape=(28,28,1), padding='same', activation='relu', filters = 64, kernel_size=3))

model.add(Conv2D(padding='same', activation='relu', filters=128, kernel_size=3))

model.add(MaxPool2D(pool_size=2))

model.add(Conv2D(padding='same', activation='relu', filters=64, kernel_size=3))

model.add(MaxPool2D(pool_size=2))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dense(10, activation='softmax'))



model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size = 0.9)
#x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8)

history = model.fit(x, y, epochs=3, validation_data=(x_valid, y_valid), verbose=1, batch_size=100)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
predictions = model.predict_classes(x_predict)
submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submission.head()
submission['Label'] = predictions

submission.to_csv("mysub.csv", index=False)