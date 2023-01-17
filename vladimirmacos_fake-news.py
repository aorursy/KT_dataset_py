# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import csv
import numpy as np

titles = []
labels = []
with open('/kaggle/input/fake-and-real-news-dataset/True.csv', 'r') as f:
    csvreader = csv.reader(f, delimiter=',')
    next(csvreader)
    for row in csvreader:
        titles.append(row[0].strip())
        labels.append(1)
        
fake_titles = []
with open('/kaggle/input/fake-and-real-news-dataset/Fake.csv', 'r') as f:
    csvreader = csv.reader(f, delimiter=',')
    next(csvreader)
    for row in csvreader:
        titles.append(row[0].strip())
        labels.append(0)

labels = np.array(labels)
        
print(len(titles))
print(len(labels))
print(titles[0])
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

oov_token = '<OOV>'
num_words = 10000
max_len = 35
train_size = 32000
test_size = 8000

tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\'\t\n’”…')
tokenizer.fit_on_texts(titles)
print(len(tokenizer.word_index))
# print(tokenizer.word_index)

sequences = tokenizer.texts_to_sequences(titles)
padded = pad_sequences(sequences, maxlen=max_len)

print(padded.shape)

def get_indices(from_ind1, to_ind1, from_ind2, to_ind2):
    indices = np.concatenate((np.arange(from_ind1, to_ind1), np.arange(from_ind2, to_ind2)))
    return indices

padded_len = len(padded)

train_indices = get_indices(0, 
                             train_size // 2, 
                             padded_len - (train_size // 2), 
                             padded_len)
train_x = padded[train_indices]
train_y = labels[train_indices]

test_indices = get_indices((train_size // 2), 
                            (train_size // 2) + (test_size // 2), 
                            padded_len - (train_size // 2) - (test_size // 2), 
                            padded_len - (train_size // 2))
test_x = padded[test_indices]
test_y = labels[test_indices]

print(len(train_x))
print(len(test_x))
from keras import models, layers
from keras.optimizers import RMSprop
from keras import regularizers
import keras

keras.backend.clear_session()

model = models.Sequential([
    layers.Embedding(num_words, 256, input_length=max_len),
    layers.Conv1D(32, 
                    4, 
                    activation='relu', 
                    kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01)
                ),
    layers.MaxPooling1D(),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(32, 
                    activation='relu', 
                    kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01),
                ),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid'),
])

optimizer = RMSprop(lr=1e-5)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
model.summary()
epochs = 500
batch_size = 128
history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_x, test_y))
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

from_index = 100
epochs = range(from_index, len(acc) + 1)

plt.plot(epochs, acc[from_index - 1:], 'r', label='Training acc')
plt.plot(epochs, val_acc[from_index - 1:], 'b', label='Validation acc')
plt.legend()

plt.show()

plt.clf()

plt.plot(epochs, loss[from_index - 1:], 'r', label='Training loss')
plt.plot(epochs, val_loss[from_index - 1:], 'b', label='Validation loss')
plt.legend()

plt.show()