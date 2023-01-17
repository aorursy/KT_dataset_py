import numpy as np 

import pandas as pd 



import os

import re

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import torch

import torch.optim as optim

import os

from sklearn.model_selection import train_test_split

import keras

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential, load_model

from keras.initializers import Constant

from keras.layers import (LSTM, 

                          Embedding, 

                          BatchNormalization,

                          Dense, 

                          TimeDistributed, 

                          Dropout, 

                          Bidirectional,

                          Flatten, 

                          GlobalMaxPool1D)

import nltk

nltk.download('punkt')

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords 

import seaborn as sns

import scipy

from scipy import spatial

from sklearn.manifold import TSNE

from mpl_toolkits.mplot3d import Axes3D

from keras import regularizers

import matplotlib.pyplot as plt

train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')
print(train.head())

print(test.head())
test_id = test['id']

test.drop(labels = ['id','keyword','location'], axis = 1, inplace = True)

train.drop(labels = ['id','location','keyword'], axis = 1, inplace = True)
def clean_data(data):

    data = data.lower()



    # remove unknow characters

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    data = emoji_pattern.sub(r'', data)



    # remove http links

    url = re.compile(r'https?://\S+|www\.\S+')

    data = url.sub(r'',data)



    # remove other symbols

    data = data.replace('#',' ')

    data = data.replace('@',' ')

    symbols = re.compile(r'[^A-Za-z0-9 ]')

    data = symbols.sub(r'',data)



    return data

    
train['text'] = train['text'].apply(lambda x: clean_data(x)).apply(lambda x:x.split(' '))

test['text'] = test['text'].apply(lambda x:clean_data(x)).apply(lambda x:x.split(' '))

stop = set(stopwords.words('english'))

train['text'] = train['text'].apply(lambda x: [word for word in x if word not in stop])

test['text'] = test['text'].apply(lambda x: [word for word in x if word not in stop])
print(train.head())

print(test.head())
sns.catplot('target',data = train, kind = 'count',palette = 'Set3')

train.target.value_counts()
sample_size = 3271



train_norm = train[train.target == 1].sample(sample_size).append(train[train.target == 0].sample(sample_size)).reset_index()
sns.catplot('target',data = train_norm, kind = 'count',palette = 'Accent')

train_norm.target.value_counts()
train_x = train_norm.text.values

train_y = train_norm.target.values

test_x = test.text.values
word_tokenizer = Tokenizer()

word_tokenizer.fit_on_texts(train_x)

vocab_length = len(word_tokenizer.word_index) + 1
word_tokenizer_test = Tokenizer()

word_tokenizer_test.fit_on_texts(test_x)
max_sentence_length = len(max(train_x, key = lambda sentence:len(word_tokenize(str(sentence)))))



# integer encode the data

def sequence(corpus):

  return word_tokenizer.texts_to_sequences(corpus)



train_pad = pad_sequences(sequence(train_x),

                          max_sentence_length,

                          padding = 'post')

test_pad = pad_sequences(sequence(test_x),

                         max_sentence_length,

                         padding = 'post')
embedding_dict={}



with open('../input/glove6b100dtxt/glove.6B.100d.txt','r') as f: 

    for line in f:

        values = line.split()

        word = values[0]

        vectors = np.asarray(values[1:],'float32')

        embedding_dict[word]= vectors

f.close()
embedding_dim = 100

embedding_matrix =  np.zeros((vocab_length, embedding_dim))

hits = 0

misses = 0



for word, index in word_tokenizer.word_index.items():

  # If word is in our vocab, then update the corresponding weights

  embedding_vector = embedding_dict.get(word)

  if embedding_vector is not None:

    embedding_matrix[index] = embedding_vector

    hits += 1

  else:

    misses += 1

print("Converted %d words (%d misses)" % (hits, misses))
tsne_2d = TSNE(n_components=2, random_state=0)

words =  list(embedding_dict.keys())

vectors = [embedding_dict[word] for word in words]

embeddings = tsne_2d.fit_transform(vectors[:500])



sns.set_palette('summer')

plt.figure(figsize=(20,15))

plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.5)

for label, x, y in zip(words, embeddings[:, 0], embeddings[:, 1]):

    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")

plt.show()
x_train, x_val, y_train, y_val = train_test_split(train_pad,

                                                  train_y,

                                                  test_size = 0.25)
def BLSTM():

  model = Sequential()

  model.add(Embedding(input_dim = embedding_matrix.shape[0],

                      output_dim = embedding_matrix.shape[1],

                      weights = [embedding_matrix],

                      trainable=False,

                     input_length = max_sentence_length))

  model.add(Bidirectional(LSTM(max_sentence_length,

                               return_sequences = True,

                               recurrent_dropout = 0.5)))

  model.add(GlobalMaxPool1D())

  model.add(BatchNormalization())

  model.add(Dropout(0.5))

  model.add(Dense(max_sentence_length, activation = 'relu',kernel_regularizer=regularizers.l2(0.01)))

  model.add(Dropout(0.5))

  model.add(Dense(max_sentence_length, activation = 'relu', kernel_regularizer=regularizers.l2(0.001)))

  model.add(Dropout(0.5))

  model.add(Dense(1, activation = 'sigmoid', kernel_regularizer=regularizers.l2(0.01)))

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  return model
# define the model

model = BLSTM()



# early stopping to avoid overfitting

es = EarlyStopping(monitor='val_loss', mode='min', 

                   verbose=1, patience= 3)



mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', 

                     mode='max', verbose=1, save_best_only=True)



# fit the model

train_history = model.fit(x_train, y_train, epochs = 15,

                          validation_data = (x_val, y_val), 

                          verbose = 0, callbacks=[es, mc])



# load the saved model

saved_model = load_model('best_model.h5')



# evaluate the model

_, train_acc = model.evaluate(x_train, y_train, verbose=0)

_, val_acc = model.evaluate(x_val, y_val, verbose = 0)

print('Train: %.3f,  Validation: %.3f' % (train_acc, val_acc))



# plot training history

plt.figure(figsize=(10,10))

sns.set_palette('Pastel1')

plt.plot(train_history.history['loss'], label='train')

plt.plot(train_history.history['val_loss'], label='validation')

plt.legend()

plt.show()
submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

submission.target = model.predict_classes(test_pad)

submission.to_csv('submission00.csv', index = False)