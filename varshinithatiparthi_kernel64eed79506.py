import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model

from keras.layers import GlobalAveragePooling1D

from keras.layers import LSTM

from keras.layers import Bidirectional
vocab_size = 20000

max_length = 120

embedding_dim = 50

trunc_type = 'post'

padding_type = 'post'

oov_tok = "<OOV>"

train = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")

test = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")
train.isnull().sum()
test.isnull().sum()
label = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[label].values

test_list = test["comment_text"].fillna("_na_").values

train_sentences = train["comment_text"].fillna("_na_").values

train_sentences
y.shape
tokenizer = Tokenizer(num_words=vocab_size,oov_token= oov_tok)

tokenizer.fit_on_texts(list(train_sentences))

# word_index = tokenizer.word_index
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_sentences)

train_padded = pad_sequences(train_sequences,padding=padding_type,maxlen = max_length)

test_sequences = tokenizer.texts_to_sequences(test_list)

test_padded = pad_sequences(test_sequences, padding = padding_type,maxlen = max_length)

print("train sequences: ",len(train_sequences[0]))

print("train padded: ",len(train_padded[0]))



print(len(train_sequences[1]))

print(len(train_padded[1]))



# print(len(train_sequences[10]))

# print(len(train_padded[10]))
# inp = Input(shape=(maxlen,))

model = tf.keras.Sequential([

tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),

tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50,return_sequences=True)),

tf.keras.layers.GlobalMaxPooling1D(),

tf.keras.layers.Dense(50,activation='relu'),

tf.keras.layers.Dense(6,activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
train_padded.shape
num_epochs = 5

history = model.fit(train_padded,y,epochs=num_epochs)
test_pred = model.predict([test_padded],verbose=2)

sample_submission = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")

sample_submission[label] = test_pred

sample_submission.to_csv('submission.csv', index=False)