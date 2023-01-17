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
from keras import Input

import keras

from keras.optimizers import Adam

import keras.backend as K

import numpy as np

from keras.layers import *

from keras.layers.core import Dense, Dropout

from keras.models import Sequential, Model

from keras.layers.recurrent import LSTM

import pandas as pd
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
MAX_SENT_LENGTH = 100

MAX_SENTS = 15

MAX_NB_WORDS = 20000

EMBEDDING_DIM = 100

VALIDATION_SPLIT = 0.2
data_train = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip', sep='\t')

print(data_train.shape)



from nltk import tokenize

from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical



reviews = []

labels = []

texts = []



for idx in range(data_train.review.shape[0]):

    text = data_train.review[idx]

    texts.append(text)

    sentences = tokenize.sent_tokenize(text)

    reviews.append(sentences)

    

    labels.append(data_train.sentiment[idx])



tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(texts)



data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')



for i, sentences in enumerate(reviews):

    for j, sent in enumerate(sentences):

        if j< MAX_SENTS:

            wordTokens = text_to_word_sequence(sent)

            k=0

            for _, word in enumerate(wordTokens):

                if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:

                    data[i,j,k] = tokenizer.word_index[word]

                    k=k+1                    

                    

word_index = tokenizer.word_index

print('Total %s unique tokens.' % len(word_index))



labels = to_categorical(np.asarray(labels))

print('Shape of data tensor:', data.shape)

print('Shape of label tensor:', labels.shape)



indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]

labels = labels[indices]

nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])



x_train = data[:-nb_validation_samples]

y_train = labels[:-nb_validation_samples]

x_val = data[-nb_validation_samples:]

y_val = labels[-nb_validation_samples:]



print('Number of positive and negative reviews in traing and validation set')

print(y_train.sum(axis=0))

print(y_val.sum(axis=0))
embedding_layer = Embedding(len(word_index) + 1,

                            EMBEDDING_DIM,

                            input_length=MAX_SENT_LENGTH)



sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sentence_input)

l_lstm = Bidirectional(LSTM(100))(embedded_sequences)

sentEncoder = Model(sentence_input, l_lstm)



review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')

review_encoder = TimeDistributed(sentEncoder)(review_input)

l_lstm_sent = Bidirectional(LSTM(100))(review_encoder)

preds = Dense(2, activation='softmax')(l_lstm_sent)

model = Model(review_input, preds)



model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['acc'])



print("model fitting - Hierachical LSTM")

model.summary()
class AttentionLayer(Layer):

    def __init__(self, **kwargs):

        super(AttentionLayer, self).__init__(** kwargs)

    

    def build(self, input_shape):

        assert len(input_shape)==3

        # W.shape = (time_steps, time_steps)

        self.W = self.add_weight(name='att_weight', 

                                 shape=(input_shape[1], input_shape[1]),

                                 initializer='uniform',

                                 trainable=True)

        super(AttentionLayer, self).build(input_shape)



    def call(self, inputs, mask=None):

        # inputs.shape = (batch_size, time_steps, seq_len)

        x = K.permute_dimensions(inputs, (0, 2, 1))

        # x.shape = (batch_size, seq_len, time_steps)

        # general

        a = K.softmax(K.tanh(K.dot(x, self.W)))

        a = K.permute_dimensions(a, (0, 2, 1))

        outputs = a * inputs

        outputs = K.sum(outputs, axis=1)

        return outputs



    def compute_output_shape(self, input_shape):

        return input_shape[0], input_shape[2]
sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sentence_input)

l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)

l_dense = TimeDistributed(Dense(200))(l_lstm)

l_att = AttentionLayer()(l_dense)

sentEncoder = Model(sentence_input, l_att)



review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')

review_encoder = TimeDistributed(sentEncoder)(review_input)

l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)

l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)

l_att_sent = AttentionLayer()(l_dense_sent)

preds = Dense(2, activation='softmax')(l_att_sent)

model = Model(review_input, preds)

model.summary()



model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['acc'])



print("model fitting - Hierachical attention network")

model.fit(x_train, y_train, validation_data=(x_val, y_val),

          epochs=10, batch_size=50)
data_test = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip', sep='\t')

print(data_test.shape)



from nltk import tokenize

from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical



reviews = []

texts = []



for idx in range(data_test.review.shape[0]):

    text = data_test.review[idx]

    texts.append(text)

    sentences = tokenize.sent_tokenize(text)

    reviews.append(sentences)

    

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(texts)



data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')



for i, sentences in enumerate(reviews):

    for j, sent in enumerate(sentences):

        if j< MAX_SENTS:

            wordTokens = text_to_word_sequence(sent)

            k=0

            for _, word in enumerate(wordTokens):

                if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:

                    data[i,j,k] = tokenizer.word_index[word]

                    k=k+1                    

                    

word_index = tokenizer.word_index

print('Total %s unique tokens.' % len(word_index))

print('Shape of data tensor:', data.shape)



indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]
pre=model.predict(data)

value=np.argmax(pre, axis=1)

df = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip', sep='\t')

df.review =value

df.to_csv('submit.csv',header=True, index=False, sep=',')