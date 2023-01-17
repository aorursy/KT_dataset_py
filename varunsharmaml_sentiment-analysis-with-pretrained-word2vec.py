# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras.models import Model
from keras.layers import Dense, Input, LSTM, GRU, Conv1D, MaxPooling1D, Concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform

from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/train.tsv", sep="\t")
test = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/test.tsv", sep="\t")
train.shape, test.shape
train.head()
test.head()
import string
tr = str.maketrans(string.punctuation, " "*32)
def modify_phrase(ph, tr):
    ph = ph.lower()
    return ph.translate(tr).strip()
modify_phrase("Hey there! I am here, using whatsapp!!...", tr)
import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    vocab_size, dim = map(int, fin.readline().split())
    word_to_vec_map = {}
    words = set()
    for line in fin:
        tokens = line.rstrip().split(' ')
        words.add(tokens[0])
        word_to_vec_map[tokens[0]] = np.array(tokens[1:], dtype=np.float64)
    i = 1
    words_to_index = {}
    index_to_words = {}
    for w in sorted(words):
        words_to_index[w] = i
        index_to_words[i] = w
        i = i + 1
    return word_to_vec_map, words_to_index, index_to_words, vocab_size, dim
word_to_vec_map, word_to_index, index_to_words, vocab_size, dim= load_vectors('../input/fasttext-wikinews/wiki-news-300d-1M.vec')
train['Phrase'] = train.apply(lambda row: modify_phrase(row.Phrase, tr), axis=1)
test['Phrase'] = test.apply(lambda row: modify_phrase(row.Phrase, tr), axis=1)
train.head()
X = np.array(train.Phrase)
Y = np.array(train.Sentiment)
X_test = np.array(test.Phrase)
X.shape, Y.shape
encode = OneHotEncoder(sparse=False)
Y = encode.fit_transform(np.reshape(Y, (Y.shape[0], 1)))
Y.shape
maxLen = 60          #maximum length of sentences
def sentences_to_indices(X, word_to_index, maxLen):
    m = X.shape[0]                                   # number of training examples
    
    X_indices = np.zeros((m, maxLen))
    
    for i in range(m):
        sentence_words = X[i].lower().strip().split()
        j = 0
        for w in sentence_words:
            if w not in word_to_index:
                w = "person"        #mostly names are not present in vocabulary
            X_indices[i, j] = word_to_index[w]
            j = j + 1
    
    return X_indices
X_vec = sentences_to_indices(X, word_to_index, maxLen)
X_test_vec = sentences_to_indices(X_test, word_to_index, maxLen)
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map["cucumber"].shape[0]
    
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    embedding_layer = Embedding(input_dim = vocab_len, output_dim = emb_dim, trainable = False) 

    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer
def movie_review_analysis(input_shape, word_to_vec_map, word_to_index):
    sentence_indices = Input(shape=input_shape, dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    embeddings = embedding_layer(sentence_indices)   
    
    X1 = Conv1D(128, 3)(embeddings)
    X2 = Conv1D(128, 3)(embeddings)
    X1 = MaxPooling1D(pool_size=4)(X1)
    X2 = MaxPooling1D(pool_size=5)(X2)
    X = Concatenate(axis=1)([X1, X2])
    
    X = GRU(units=128, dropout=0.4, return_sequences=True)(X)
    X = LSTM(units=128, dropout=0.3)(X)
    
    X = Dense(units = 32, activation="relu")(X)
    X = Dense(units=5, activation='softmax')(X)
    
    model = Model(inputs=sentence_indices, outputs=X)
    return model
model = movie_review_analysis((maxLen,), word_to_vec_map, word_to_index)
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
model.summary()
track = model.fit(X_vec, Y, batch_size=128, epochs=9)
plt.plot(track.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
plt.plot(track.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
preds = model.predict(X_test_vec, batch_size=128, verbose=1)
preds = preds.argmax(axis=-1)
submit = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv")
submit["Sentiment"] = preds
submit.to_csv("submitNow.csv", index=False)
