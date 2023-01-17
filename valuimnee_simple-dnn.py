# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt
import tensorflow

print(tensorflow.__version__)

import tensorflow.keras as keras

print(keras.__version__)

from keras import Model, Input

from keras.layers import Dense, Dropout
def load_data(filename):

  data = pd.read_csv(filename)

  y_data = data['label'].values

  data.drop(columns=['label'], inplace=True)

  rows, columns = 4, 4

  _, ax = plt.subplots(nrows=rows, ncols=columns, figsize=(10, 11))

  for i, curr_ax in enumerate(ax.flat):

    curr_ax.imshow(np.array(data.iloc[i, :]).reshape((28, 28)))

    curr_ax.set_title(str(y_data[i]))

  plt.show()

  x_data = data.values

  return x_data, y_data
x_train, y_train = load_data('/kaggle/input/Kannada-MNIST/train.csv')
# split to train and validation sets

train_coef = 0.7

train_size = int(x_train.shape[0]*train_coef)

x_train, x_valid = x_train[:train_size], x_train[train_size:]

y_train, y_valid = y_train[:train_size], y_train[train_size:]
# normalize data

x_train = x_train / 255 * 2 - 1

x_valid = x_valid / 255 * 2 - 1
x = Input(shape=(784,))

y = Dense(60, activation='relu')(x)

y = Dense(100, activation='relu')(y)

y = Dense(100, activation='relu')(y)

y = Dense(10, activation='softmax')(y)
model = Model(inputs=[x], output=[y])

model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, batch_size=10, verbose=1, epochs=20,

          validation_data=(x_valid, y_valid), 

          callbacks=[keras.callbacks.CSVLogger('log.csv'), 

                     keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True)])