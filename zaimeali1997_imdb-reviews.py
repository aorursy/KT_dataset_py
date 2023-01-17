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

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb

from tensorflow.keras import models, layers, optimizers
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) 

# most frequent words which occurs 10000 will be retrieve
print('Words: ', train_data[0])

print('Labels: ', train_labels[0])



# Tokenization performed that's why there are numbers
word_index = imdb.get_word_index()

print(word_index)
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
reverse_word_index[1]
review_at_0 = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

print(review_at_0)

# 0, 1, 2 is related to space, margin
def vectorize_sequences(sequences, dimension = 10000):

    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):

        results[i, sequence] = 1.

    return results
x_train = vectorize_sequences(train_data)

x_test = vectorize_sequences(test_data)
x_train.shape
print(x_train[0])
y_train = np.asarray(train_labels).astype('float32')

y_test = np.asarray(test_labels).astype('float32')
model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid')) # sigmoid returns the value between 0 and 1
# there should be no bottleneck

'''

means first layer should have larger output

and last layer should have smaller

'''
x_val = x_train[:10000]

partial_x_train = x_train[10000:]



y_val = y_train[:10000]

partial_y_train = y_train[10000:]
print(x_val.shape)

print(partial_x_train.shape)
model.compile(

    optimizer='rmsprop', # gradient descent is also known as rmsprop

    loss='binary_crossentropy', # loss func is coorelated to it's problem

    metrics=['acc']

)



history = model.fit(

    partial_x_train,

    partial_y_train,

    epochs=20,

    batch_size=512,

    validation_data=(x_val, y_val)

)
history.params
history.model
history.history
history_dict = history.history
loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label = 'Training Loss')

plt.plot(epochs, val_loss_values, 'b', label = 'Validation Loss')

plt.title("Training and Validation Loss")

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
model.predict(x_test)
# Validation Loss is getting increase after 4 epochs where Training Loss is getting decrease in every epochs



# Training Accuracy is getting increase in every epoch
'''So the epoch value should be 4'''
history = model.fit(

    partial_x_train,

    partial_y_train,

    epochs=4,

    batch_size=512,

    validation_data=(x_val, y_val)

)
history_dict = history.history



loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label = 'Training Loss')

plt.plot(epochs, val_loss_values, 'b', label = 'Validation Loss')

plt.title("Training and Validation Loss")

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
model.predict(x_test)