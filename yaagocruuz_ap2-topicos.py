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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Bidirectional
import matplotlib.pyplot as plt
from keras.layers import LSTM
df = pd.read_csv('../input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv')
df = df.dropna(how='any',axis=0)
df = df[['Review Text','Rating']]
df.head()
X = df.iloc[:,0].values
y = df.iloc[:,1].values
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
maxlen = 200 # Cuts off reviews after 200 words
max_words = 1000 # Considers only the top 1000 words in the dataset
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', y.shape)
X_total, X_test, y_total, y_test = train_test_split(data, y, 
                                                    test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_total, y_total, 
                                                    test_size=0.1, random_state=42)
X_train.shape, X_test.shape
X_val.shape, y_val.shape
glove_dir = '../input/glove-global-vectors-for-word-representation/'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))
embedding_dim = 50
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector #Words not found in the embedding index will be all zeros. 
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(LSTM(32, dropout=0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop',
loss='sparse_categorical_crossentropy',
metrics=['acc'])

history = model.fit(X_train, y_train,
epochs=20,
batch_size=128,
validation_data=(X_val, y_val))
model.save_weights('pre_trained_glove_model.h5')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
