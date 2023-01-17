# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv(r'/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv')
data.head()
#so we are classifying the Message into either "ham" or "spam", we convert the categorical output to numerical, 

#"ham" = 0

#"spam" = 1

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

Category_en = label_encoder.fit_transform(data['Category'])
Category_en
message = data['Message'].to_list()

message[:5]
train_size = int(len(message)*0.7)

train_size
train_message = message[:train_size]

test_message = message[train_size:]

train_cat = Category_en[:train_size]

test_cat = Category_en[train_size:]
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
vocab_size = 1000

embedding_dim = 16

max_len = 75

trunc_type='post'

padding_type='post'

oov_tok = "<OOV>"



tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)

tokenizer.fit_on_texts(train_message)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(train_message)

padded = pad_sequences(sequences, maxlen = max_len, padding = padding_type, truncating = trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_message)

test_padded = pad_sequences(test_sequences, maxlen = max_len, padding = padding_type, truncating = trunc_type)
padded[12]
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_len),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(16, activation = 'relu'),

    tf.keras.layers.Dense(1, activation = 'sigmoid')

])



model.compile(optimizer = 'adam',

             loss = 'binary_crossentropy',

             metrics = ['accuracy'])

model.summary()
num_epochs = 10

model_fit = model.fit(padded, train_cat, epochs = num_epochs, validation_data=(test_padded, test_cat))
#checking our predicts

classes = model.predict(test_padded)

for i in range(15):

    print(test_message[i])

    print(test_padded[i])

    print(classes[i])

    if classes[i]>0.7:

        print("It is a spam")

    else:

        print("It is not a spam")

        print("____________________________")
import matplotlib.pyplot as plt





def plot_graphs(model, string):

  plt.plot(model.history[string])

  plt.plot(model.history['val_'+string])

  plt.xlabel("Epochs")

  plt.ylabel(string)

  plt.legend([string, 'val_'+string])

  plt.show()

  

plot_graphs(model_fit, "accuracy")

plot_graphs(model_fit, "loss")