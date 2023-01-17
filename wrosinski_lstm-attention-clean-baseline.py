import csv

import gc

import glob

import json

import math

import operator

import os

import pprint

import re

import string

import time

from collections import OrderedDict

from math import floor



import joblib

import keras

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from gensim.models import KeyedVectors

from keras import backend as K

from keras import constraints, initializers, layers, optimizers, regularizers

from keras.callbacks import *

from keras.engine.topology import Layer

from keras.layers import *

from keras.layers import (GRU, LSTM, Activation, Add, Bidirectional,

                          Concatenate, Conv1D, CuDNNGRU, CuDNNLSTM, Dense,

                          Dropout, Embedding, Flatten, GlobalMaxPool1D, Input)

from keras.models import Model

from keras.optimizers import *

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from sklearn.metrics import log_loss

from tqdm import tqdm



%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 9)

plt.style.use('ggplot')
files = sorted(glob.glob('../input/gendered-pronoun-resolution/*.tsv'))

pprint.pprint(files)



submission = pd.read_csv('../input/gendered-pronoun-resolution/sample_submission_stage_1.csv')

print('submission head:')

print(submission.head())



# Download data from GAP Google dataset

# From there, -test and -validation will serve as training set

gap_test = pd.read_csv('../input/kaggle-gap/gap-test.tsv', sep='\t')

gap_valid = pd.read_csv('../input/kaggle-gap/gap-validation.tsv', sep='\t')

train = pd.concat([gap_test, gap_valid], ignore_index=True, sort=False)



# gap-development is public test set on Kaggle

test = pd.read_csv('../input/kaggle-gap/gap-development.tsv', sep='\t')



print('\ntrain shape: {}'.format(train.shape))

print('test shape: {}'.format(test.shape))
train_samples = np.random.choice(train.Text.values, 15)

test_samples = np.random.choice(test.Text.values, 15)



print('\ttrain samples inspection:\n')

for i in range(len(train_samples)):

    print('sample {}:\n{}\n'.format(i, train_samples[i]))



print('\n\ttest samples inspection:\n')

for i in range(len(test_samples)):

    print('sample {}:\n{}\n'.format(i, test_samples[i]))
train_lens = train.Text.apply(lambda x: len(x.split(' ')))

test_lens = test.Text.apply(lambda x: len(x.split(' ')))



plt.hist(train_lens.values, bins=25, color='r')

plt.hist(test_lens.values, bins=25, color='b')

plt.title('train & test length distribution:')

plt.show()
def build_vocab(sentences, verbose=True):

    """

    :param sentences: list of list of words

    :return: dictionary of words and their count

    """

    vocab = {}

    for sentence in tqdm(sentences, disable=(not verbose)):

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab





def check_coverage(vocab, embeddings_index):

    a = {}

    oov = {}

    k = 0

    i = 0

    for word in tqdm(vocab):

        try:

            a[word] = embeddings_index[word]

            k += vocab[word]

        except:



            oov[word] = vocab[word]

            i += vocab[word]

            pass



    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))

    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))

    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]



    return sorted_x





# Functions to clean the text:

def clean_text(x):



    x = str(x)

    for punct in "/-'":

        x = x.replace(punct, ' ')

    for punct in '&':

        x = x.replace(punct, f' {punct} ')

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':

        x = x.replace(punct, '')

    return x





def clean_numbers(x):



    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    return x





def add_lower(embedding, vocab):

    count = 0

    for word in vocab:

        if word in embedding and word.lower() not in embedding:  

            embedding[word.lower()] = embedding[word]

            count += 1

    print(f"Added {count} words to embedding")
# First step is to concatenate train and test data to act on whole set during processing:

df_full = pd.concat([train, test], ignore_index=True, sort=False)
# Load GoogleNews embedding

news_path = '../input/nlpword2vecembeddingspretrained/GoogleNews-vectors-negative300.bin'

embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)
# Initial trial:

vocab = build_vocab(df_full.Text.apply(lambda x: x.split(' ')).tolist())

oov_words = check_coverage(vocab, embeddings_index)



# When applied on raw text, we see that embeddings for only 48% of words are available.

# This is too low, let's try to improve upon this!
# Lowercase:

vocab = build_vocab(df_full.Text.apply(lambda x: x.lower().split(' ')).tolist())

oov_words = check_coverage(vocab, embeddings_index)



# After lowering the words, number of embeddings found has dropped.

# For this embedding, words should be kept in their original case.
# Clean punctuation:

vocab = build_vocab(df_full.Text.apply(lambda x: clean_text(x).split(' ')).tolist())

oov_words = check_coverage(vocab, embeddings_index)



# After cleaning punctuation, a significant gain is achieved, nice!
# Clean punctuation & numbers:

vocab = build_vocab(df_full.Text.apply(lambda x: clean_numbers(clean_text(x)).split(' ')).tolist())

oov_words = check_coverage(vocab, embeddings_index)



# When cleaning numbers, there is an additional subtle gain of almost 3%.

# Not very much but still useful! 

# Now we have almost 93% of words covered, that's very good fraction.
# Clean punctuation & numbers, in addition replace all non-alphanumeric characters:

regex = re.compile('[^a-zA-Z]')



sentences_clean = df_full.Text.apply(lambda x: clean_numbers(clean_text(x)).split(' ')).tolist()

sentences_clean = list(map(lambda y: [regex.sub('', x) for x in y], sentences_clean))



vocab = build_vocab(sentences_clean)

oov_words = check_coverage(vocab, embeddings_index)



# Removing all non-alphanumeric characters does not bring an improvement.

# Let's stick with the previous version, where punctuation and numbers are cleaned.
# Final vocabulary:

final_vocab = build_vocab(df_full.Text.apply(lambda x: clean_numbers(clean_text(x)).split(' ')).tolist())

# oov_words = check_coverage(final_vocab, embeddings_index)





del embeddings_index

gc.collect()
# From https://www.kaggle.com/shujian/single-rnn-with-4-folds-clr

def load_glove(word_index):



    all_embs = np.stack(embeddings_index.values())

    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

            

    return embedding_matrix





max_features = 40000
# Process embedding matrix once and save it.

# Afterwards, is the file exists, matrix will be loaded instead of processed again.

# Saves a lot of time when performing experiments, because embedding loading takes quite a while!



if os.path.isfile('glove_embedding.joblib'):

    embedding_matrix = joblib.load('glove_embedding.joblib')

    print('embedding matrix loaded.')

else:

    # Load embedding:

    EMBEDDING_FILE = '../input/glove-embeddings/glove.840B.300d.txt'

    def get_coefs(word,*arr): 

        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    

    embedding_matrix = load_glove(final_vocab)

    joblib.dump(embedding_matrix, 'glove_embedding.joblib')

    print('embedding matrix processed.')

    del embeddings_index

    gc.collect()





# We set max_features to number equal to all words from vocabulary,

# because the number isn't very high - less than 37k.

max_features = embedding_matrix.shape[0]

embed_size = embedding_matrix.shape[1]

print('embedding matrix shape: {}'.format(embedding_matrix.shape))

print('max features: {}'.format(max_features))
# https://www.kaggle.com/keyit92/coref-by-mlp-cnn-coattention

# Function to create labels out of original df columns:

def _row_to_y(row):

    if row.loc['A-coref']:

        return 0

    if row.loc['B-coref']:

        return 1

    return 2





# Process data to tokenized form:

maxlen = 128





X = df_full.Text.apply(lambda x: clean_numbers(clean_text(x)).split(' '))



# max_features = 40000

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(X))



X = tokenizer.texts_to_sequences(X)



X_train = X[:train.shape[0]]

X_valid = X[train.shape[0]:]



X_train = pad_sequences(X_train, maxlen=maxlen)

X_valid = pad_sequences(X_valid, maxlen=maxlen)



y_train = train.apply(_row_to_y, axis=1)

y_valid = test.apply(_row_to_y, axis=1)



y_train = to_categorical(y_train)

y_valid = to_categorical(y_valid)



print(X_train.shape, y_train.shape)

print(X_valid.shape, y_valid.shape)
# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb



class Attention(Layer):

    def __init__(self, step_dim,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True, **kwargs):

        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')



        self.W_regularizer = regularizers.get(W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)



        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)



        self.bias = bias

        self.step_dim = step_dim

        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3



        self.W = self.add_weight((input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]



        if self.bias:

            self.b = self.add_weight((input_shape[1],),

                                     initializer='zero',

                                     name='{}_b'.format(self.name),

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)

        else:

            self.b = None



        self.built = True



    def compute_mask(self, input, input_mask=None):

        return None



    def call(self, x, mask=None):

        features_dim = self.features_dim

        step_dim = self.step_dim



        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),

                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))



        if self.bias:

            eij += self.b



        eij = K.tanh(eij)



        a = K.exp(eij)



        if mask is not None:

            a *= K.cast(mask, K.floatx())



        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



        a = K.expand_dims(a)

        weighted_input = x * a

        return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        return input_shape[0],  self.features_dim





def LstmBasic(embedding_matrix, dropout=0.0):



    inp = Input(shape=(maxlen,))

    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)

    # x = Embedding(max_features, embed_size)(inp)

    x = SpatialDropout1D(0.1)(x)



    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

    # x = PReLU()(x)

    identity = x

    x = Dropout(dropout)(x)

    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

    # x = PReLU()(x)

    x = Add()([x, identity])

    identity = x

    x = Dropout(dropout)(x)



    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

    # x = PReLU()(x)

    x = Add()([x, identity])

    x = Attention(maxlen)(x)

    x = Dropout(dropout)(x)



    x = Dense(128)(x)

    x = PReLU()(x)

    x = Dense(3, activation="softmax")(x)



    model = Model(inputs=inp, outputs=x)

    model.compile(

        loss='categorical_crossentropy',

        optimizer=optimizers.Adam(lr=1e-4),

        metrics=['accuracy'])



    return model
# Run parameters

N_BAGS = 3

N_EPOCHS = 5

TO_MONITOR = 'val_loss'

RUN_NAME = 'LstmAttention'

monitor_mode = 'min'





# Create checkpoints dir

if not os.path.isdir('./checkpoints/'):

    os.mkdir('./checkpoints/')





val_preds = []

    

# Train bagged model

for b in range(N_BAGS):

    

    print('\trunning bag: {}'.format(b))

    

    # Initialize model:

    model = LstmBasic(embedding_matrix)



    # Set of training callbacks

    ckpt_name = 'checkpoints/{0}_bag_{1}.h5'.format(RUN_NAME, b)

    logger = CSVLogger('./checkpoints/{0}_bag_{1}.log'.format(RUN_NAME, b))



    ckpt = ModelCheckpoint(ckpt_name, 

                           save_best_only=True,

                           save_weights_only=True, 

                           verbose=1, 

                           monitor=TO_MONITOR, 

                           mode=monitor_mode)

    reduce_lr = ReduceLROnPlateau(monitor=TO_MONITOR,

                                  mode=monitor_mode, 

                                  patience=10)

    early_stop = EarlyStopping(monitor=TO_MONITOR,

                               mode=monitor_mode, 

                               patience=20)



    history = model.fit(X_train, y_train, batch_size=256, epochs=N_EPOCHS, 

            validation_data=(X_valid, y_valid), verbose=0,

            callbacks=[ckpt, reduce_lr, early_stop, logger])



    val_loss_min = min(history.history['val_loss'])

    print('best model val loss: {:.4f}'.format(val_loss_min))

    

    print('loading best weights')

    model.load_weights(ckpt_name)

    print('predict validation set with best weights')

    val_pred = model.predict(X_valid, batch_size=256)

    val_preds.append(val_pred)

    



# Average over bags:

val_pred = np.mean(val_preds, axis=0)

val_logloss = log_loss(y_valid, val_pred)

print('averaged logloss: {:.4f}'.format(val_logloss))
plt.hist(val_pred[:, 0], bins=50, color='r', label='A')

plt.hist(val_pred[:, 1], bins=50, color='b', label='B')

plt.hist(val_pred[:, 2], bins=50, color='g', label='Neither')

plt.legend()

plt.title('validation prediction distribution:')

plt.show()
submission_ = submission.copy()

submission_.iloc[:, 1:] = val_pred

submission_.to_csv('lstm_baseline_loss_{:.4f}.csv'.format(val_logloss), index=False)