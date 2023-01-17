import gc
import re
import sys
import csv
import codecs
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
%matplotlib inline

import gensim.models.keyedvectors as word2vec

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, Bidirectional, GlobalMaxPool1D,Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from tqdm.notebook import tqdm
from tqdm import tqdm_notebook

import warnings
warnings.simplefilter('ignore')
# TPU Config
# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
def get_data():
    train = pd.read_csv("../input/steam-game-reviews/train.csv")
    test = pd.read_csv("../input/steam-game-reviews/test.csv")
    sub = pd.read_csv("../input/steam-game-reviews/sample_submission.csv")
    
    print("Train Shape : \t{}\nTest Shape : \t{}\n".format(train.shape, test.shape))

    return train, test, sub
train, test, sub = get_data()
list_sentences_train = train["user_review"]
list_sentences_test = test["user_review"]
%%time

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
max_len = 500
X_t = pad_sequences(list_tokenized_train, maxlen=max_len)
X_te = pad_sequences(list_tokenized_test, maxlen=max_len)
def load_embedding_matrix():
    word2vec_dict = word2vec.KeyedVectors.load_word2vec_format("../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin", binary=True)
    embed_size = 300
    embedding_index = dict()
    for word in tqdm_notebook(word2vec_dict.wv.vocab):
        embedding_index[word] = word2vec_dict.word_vec(word)
    print("Loaded {} word vectors.".format(len(embedding_index)))
    gc.collect()
    
    # We get the mean, and std of the embedding weights so that we could maintain the same statistics for the rest of our random generated weights.
    all_embs = np.stack(list(embedding_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    
    nb_words = len(tokenizer.word_index)
    
    # We are going to set the embedding size to the pretrained dimension as we are replicating it.
    # The size would be Number of Words X Embedding Size
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    gc.collect()
    # Now we have generated a random matrix lets fill it out with our own dictionary and the one with pretrained embeddings.
    embedded_count = 0
    for word, i in tqdm_notebook(tokenizer.word_index.items()):
        i -= 1
        # Then we see whether the word is in Word2Vec dictionary, if yes get the pretrained weights.
        embedding_vector = embedding_index.get(word)
        # And store that in our embedding_matrix that we will use to train ML model.
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            embedded_count += 1
    print("Total Embedded : {} common wrds".format(embedded_count))
        
    del embedding_index
    gc.collect()
    
    # Finally return the embedding matrix
    return embedding_matrix
%%time

embedding_matrix = load_embedding_matrix()
embedding_matrix.shape
embed_size = embedding_matrix.shape[1]
max_features = len(tokenizer.word_index)

print(max_features, embed_size)

with strategy.scope():    
    inp = Input(shape=(max_len, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(LSTM(60, return_sequences=True, name='lstm_layer', dropout=0.1, recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])
model.summary()
batch_size = 32
epochs = 3

hist = model.fit(X_t, train['user_suggestion'], batch_size=batch_size, epochs=epochs, validation_split=0.1)