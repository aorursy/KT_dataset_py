%matplotlib inline

import os

import time

import math

import operator 

import re

import gc

import numpy as np

import pandas as pd

import keras

import tensorflow as tf

import keras.backend as K

import matplotlib

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn import metrics

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Input, Embedding, Dropout, Dense, Flatten, Activation, CuDNNGRU, CuDNNLSTM, Bidirectional, Average, SpatialDropout1D, Average

from keras.models import Model

from keras.optimizers import Adam

from keras import initializers, regularizers, constraints, optimizers, layers

tqdm.pandas()

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
# Adapted from Viel (2018)

glove_path = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

def load_embed(file):

    def get_coefs(word,*arr): 

        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))    

    return embeddings_index

print('Loading Glove...')

embed_glove = load_embed(glove_path)

print('Glove Loaded')
# Diccionarios. Viel (2018)

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi'}

mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization', 'pokémon': 'pokemon'}

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

# Funciones Auxiliares

def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text



def unknown_punct(embed, punct):

    unknown = ''

    for p in punct:

        if p not in embed:

            unknown += p

            unknown += ' '

    return unknown



def clean_special_chars(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])

    

    for p in punct:

        text = text.replace(p, f' {p} ')

    

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last

    for s in specials:

        text = text.replace(s, specials[s])

    

    return text



def correct_spelling(x, dic):

    for word in dic.keys():

        x = x.replace(word, dic[word])

    return x
# Pasar a Minusculas los datos

train['treated_question'] = train['question_text'].apply(lambda x: x.lower())

test['treated_question'] = test['question_text'].apply(lambda x: x.lower())

# Eliminar las Contracciones

train['treated_question'] = train['treated_question'].apply(lambda x: clean_contractions(x, contraction_mapping))

test['treated_question'] = test['treated_question'].apply(lambda x: clean_contractions(x, contraction_mapping))

# Eliminar los Caracteres Especiales

train['treated_question'] = train['treated_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

test['treated_question'] = test['treated_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

# Eliminar los Errores Ortograficos

test['treated_question'] = test['treated_question'].apply(lambda x: correct_spelling(x, mispell_dict))
train.head(10)
VOCABULARY=100000

tokenizer = Tokenizer(num_words=VOCABULARY, char_level=False, oov_token='<OOV>')

tokenizer.fit_on_texts(list(train['treated_question']))

X = tokenizer.texts_to_sequences(train['treated_question'])

X = pad_sequences(X, maxlen=50)

Y = train['target'].values
X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=0.05, random_state=27014)
# Funcion de Viel (2018)

def make_embed_matrix(embeddings_index, word_index, len_voc):

    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]

    word_index = word_index

    embedding_matrix = np.random.normal(emb_mean, emb_std, (len_voc, embed_size))

    

    for word, i in word_index.items():

        if i >= len_voc:

            continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: 

            embedding_matrix[i] = embedding_vector

    

    return embedding_matrix
embedding = make_embed_matrix(embed_glove, tokenizer.word_index, VOCABULARY)



gc.collect()
class Attention(keras.layers.Layer):

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

        return input_shape[0], self.features_dim

def model(embedding_matrix, EMBED_SIZE=300, MAX_LEN=50, VOCABULARY=100000):

    inp = Input(shape=(MAX_LEN, ), name='input')

    X = Embedding(VOCABULARY, EMBED_SIZE, trainable=True, weights=[embedding_matrix], name='Embedding')(inp)

    X = Bidirectional(CuDNNLSTM(128, return_sequences=True, name='LSTM'), name='bid1')(X)

    A_LSTM = Attention(MAX_LEN)(X)

    X = SpatialDropout1D(0.35)(X)

    X = Bidirectional(CuDNNGRU(128, return_sequences=True, name='GRU'), name='bid2')(X)

    A_GRU = Attention(MAX_LEN)(X)

    A = Average()([A_LSTM, A_GRU, X])

    A = Flatten()(A)

    A = Dense(256, activation='relu', name='fc1')(A)

    A = Dropout(0.2)(A)

    A = Dense(128, activation='relu', name='fc2')(A)

    O = Dense(1, activation='sigmoid', name='output')(A)

    model = Model(inputs=inp, outputs=O)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

    return model
model = model(embedding)

model.summary()
t_start = time.time()

history = model.fit(x=X_train, y=Y_train, batch_size=512, epochs=5,validation_data=(X_dev, Y_dev))

print(t_start - time.time())
def graf_model(train_history):

    f = plt.figure(figsize=(15,10))

    ax = f.add_subplot(121)

    ax2 = f.add_subplot(122)

    # summarize history for accuracy

    ax.plot(train_history.history['binary_accuracy'])

    ax.plot(train_history.history['val_binary_accuracy'])

    ax.set_title('model accuracy')

    ax.set_ylabel('accuracy')

    ax.set_xlabel('epoch')

    ax.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    ax2.plot(train_history.history['loss'])

    ax2.plot(train_history.history['val_loss'])

    ax2.set_title('model loss')

    ax2.set_ylabel('loss')

    ax2.set_xlabel('epoch')

    ax2.legend(['train', 'test'], loc='upper left')

    plt.show()

graf_model(history)    
X_valid = tokenizer.texts_to_sequences(test['treated_question'])

X_valid = pad_sequences(X_valid, maxlen=50)

Y_hat = model.predict(X_valid)
Y_hat = Y_hat.ravel()

for i in range(len(Y_hat)):

    if Y_hat[i] < 0.5: 

        Y_hat[i] = 0

    else:

        Y_hat[i] = 1

sub = pd.DataFrame({ 'qid': test['qid'].values, 'prediction': Y_hat })

sub['prediction'] = sub['prediction'].astype('int32')

sub.to_csv('submission.csv', index=False)