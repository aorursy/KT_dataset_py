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


import matplotlib.pyplot as plt

import keras

import os

import tensorflow as tf

import cv2

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
train = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/train.csv')

test = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/test.csv')
train
train['language'].unique()
unique_languages = ['English', 'French', 'Thai', 'Turkish', 'Urdu', 'Russian',

       'Bulgarian', 'German', 'Arabic', 'Chinese', 'Hindi', 'Swahili',

       'Vietnamese', 'Spanish', 'Greek']



plt.figure(figsize=(9, 5))

plt.barh(unique_languages, train['language'].value_counts(), color = 'green')
plt.figure(figsize=(7, 7))

plt.pie(train['language'].value_counts(), shadow=True, frame=True, labels=unique_languages, autopct='%1.1f%%')

plt.show()
train['label'].value_counts()
train.info()
x = train['hypothesis']

y = train['label']

x.shape, y.shape
y = np.asarray(y, dtype = np.uint8).reshape(-1, 1)

y
vocab_size = 10000

oov_tok = '<OOV>'

emb_dim = 32

max_len = 150
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(x)

train_seq = tokenizer.texts_to_sequences(x)

padded_train = pad_sequences(train_seq, maxlen = max_len)
x_test = test['hypothesis']

x_test.shape
test_seq = tokenizer.texts_to_sequences(x_test)

padded_test = pad_sequences(test_seq, maxlen = max_len)
model = tf.keras.Sequential()

model.add(tf.keras.layers.Embedding(vocab_size, output_dim=emb_dim, input_length=max_len))

model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)))

model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,)))

model.add(tf.keras.layers.Dropout(0.2))



model.add(tf.keras.layers.Dense(16, activation='relu'))

model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(3, activation='softmax'))

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(padded_train, y, epochs=50)
prediction = model.predict_classes(padded_test)

prediction
pred_ds = pd.DataFrame(test['id'])

pred_ds = pd.concat([pred_ds, pd.DataFrame(prediction, columns = ['prediction'])], axis = 1)

pred_ds
pred_ds.to_csv('submission.csv', index = False)