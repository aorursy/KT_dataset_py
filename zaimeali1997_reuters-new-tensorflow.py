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
import tensorflow as tf

from tensorflow.keras.datasets import reuters

from tensorflow.keras import models, layers, optimizers, utils      # utils for one hot encode

import matplotlib.pyplot as plt
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print(len(train_data))

print(len(test_data))
train_data[0]
def vectorize_sequences(sequences, dimension=10000):

    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):

        results[i, sequence] = 1.

    return results
x_train = vectorize_sequences(train_data)

x_test = vectorize_sequences(test_data)
len(train_labels)
# dimension = 46 bcz no. of column is 46

def to_one_hot(labels, dimension=46):

    results = np.zeros((len(labels), dimension))

    for i, label in enumerate(labels):

        results[i, label] = 1.

    return results
x_train_one_hot = to_one_hot(train_labels)

x_test_one_hot = to_one_hot(test_labels)
# or you can do one hot encode through utils lib

one_hot_train_labels = utils.to_categorical(train_labels)

one_hot_test_labels = utils.to_categorical(test_labels)
one_hot_train_labels
# Validation

x_val = x_train[:1000]

par_x_train = x_train[1000:]



y_val = one_hot_train_labels[:1000]

par_y_train = one_hot_train_labels[1000:]
# Building Network



model = models.Sequential()

model.add(layers.Dense(64, activation = 'relu', input_shape = (10000, )))

model.add(layers.Dense(64, activation = 'relu'))

model.add(layers.Dense(46, activation = 'softmax')) # 46 classes that's why 46 outputs 

# softmax: match with every class and give probability
len(par_x_train)
# compile



model.compile(

    optimizer = 'rmsprop',

    loss = 'categorical_crossentropy', # multiclass classification

    metrics = ['acc']

)



history = model.fit(

    par_x_train,

    par_y_train,

    epochs = 20,

    batch_size = 512,

    validation_data = (x_val, y_val)

)
history_dict = history.history
loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)



plt.plot(epochs, loss_values, 'bo', label = 'Training Loss')

plt.plot(epochs, val_loss_values, 'b', label = 'Validation Loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()



plt.clf()



acc_values = history_dict['acc']

val_acc_values = history_dict['val_acc']



plt.plot(epochs, acc_values, 'bo', label = 'Training Accuracy')

plt.plot(epochs, val_acc_values, 'b', label = 'Validation Accuracy')

plt.title('Training and Validation Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
prediction = model.predict(x_test)
prediction.shape
prediction[1]
prediction[1].sum()
prediction[0].argmax()