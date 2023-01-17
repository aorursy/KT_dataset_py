#Ignore the warnings

import warnings

warnings.filterwarnings('ignore')

warnings.filterwarnings('always')



#Data visualization and manipulation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns



#configure

%matplotlib inline

sns.set(style = 'whitegrid', color_codes = True)



import nltk



#importing stop-words

from nltk.corpus import stopwords

stop_words = set(nltk.corpus.stopwords.words('english'))



#Tokenization

from nltk import word_tokenize, sent_tokenize



#Keras

import keras

from keras.preprocessing.text import one_hot, Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Flatten, Embedding, Input

from keras.models import Model

from keras.optimizers import Adam

import tensorflow as tf
sample_text_1="bitty bought a bit of butter"

sample_text_2="but the bit of butter was a bit bitter"

sample_text_3="so she bought some better butter to make the bitter butter better"



corp = [sample_text_1, sample_text_2, sample_text_3]

no_docs = len(corp)

print(no_docs)
VOCAB_SIZE = 50

encod_corp = []

for i, doc in enumerate(corp):

    encod_corp.append(one_hot(doc, VOCAB_SIZE))

    print('The encoding for document', i+1, 'is : ', one_hot(doc, VOCAB_SIZE))
encod_corp #list of lists
#Finding max_len

MAX_LEN = -1

for doc in corp:

    tokens = nltk.word_tokenize(doc)

    if(len(tokens) > MAX_LEN):

        MAX_LEN = len(tokens)

print('The maximum number of unique words in any document is : ', MAX_LEN)
#How nltk word tokenizes a text

nltk.word_tokenize(corp[0])
#Actual padding

pad_corp = pad_sequences(encod_corp, maxlen=MAX_LEN, padding='post', value=0)

pad_corp
#Specifying the input shape

input = Input(shape=(no_docs, MAX_LEN), dtype='float64')

input
word_input = Input(shape = (MAX_LEN,), dtype = 'float64')
#Creating the embedding

word_embedding = Embedding(input_dim = VOCAB_SIZE, output_dim = 8, input_length= MAX_LEN)(word_input)
#Flattening the embedding

word_vec = Flatten()(word_embedding)

word_vec
#combining all into a Keras Model

embed_model = Model([word_input], word_vec)
#Training the model

embed_model.compile(optimizer = Adam(lr = 1e-3), loss='binary_crossentropy', metrics = ['acc'])
#Model summary

print(embed_model.summary())
#Getting the embeddings

embeddings = embed_model.predict(pad_corp)
print('Shape of embeddings : ', embeddings.shape)

print(embeddings)
#Reshaping embeddings

embeddings = embeddings.reshape(-1, MAX_LEN, 8)

print('Shape of embeddings : ', embeddings.shape)

print(embeddings)
for i, doc in enumerate(embeddings):

    for j, word in enumerate(doc):

        print('The encoding for Word', j+1, 'in Document', i+1, ' : ', word)