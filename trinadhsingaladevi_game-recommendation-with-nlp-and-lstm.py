import pandas as pd

import re

import gc

import sys

#from utils import write_status

from nltk.stem.porter import PorterStemmer

from keras.models import Model, Sequential

from keras.layers import Dense, Embedding, Input, Conv1D, GlobalMaxPool1D, Dropout, concatenate, Layer, InputSpec

from keras.preprocessing import text, sequence

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K

from keras import activations, initializers, regularizers, constraints

from keras.utils.conv_utils import conv_output_length

from keras.regularizers import l2

from keras.constraints import maxnorm

from keras.layers import LSTM
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df1 = pd.read_csv('../input/janata-data/train.csv')

df2 = pd.read_csv('../input/janata-data/game_overview.csv')
print(df1.shape)

print(df1.isnull().sum())

df1.head()
print(df2.shape)

print(df2.isnull().sum())

df2.head()
df = pd.merge(df1,df2,on='title')
print(df.shape)

df.head()
df.isnull().sum()
train_labels = df['user_suggestion']
df[df['user_review'].str.find('♥')>0]['user_review']

#replace url

def preprocess_word(word):

    # Remove punctuation

    word = word.strip('\'"?!,.():;')

    # Convert more than 2 letter repetitions to 2 letter

    # funnnnny --> funny

    word = re.sub(r'(.)\1+', r'\1\1', word)

    word = word.strip('@')

    # Remove - & '

    word = re.sub(r'(-|\')', '', word)

    return word

def handle_emojis(col):

    # Smile -- :), : ), :-), (:, ( :, (-:, :')

    col = col.replace(to_replace = '(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', value =  ' EMO_POS ', regex = True)

    #tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)

    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D

    col = col.replace(to_replace = '(:\s?D|:-D|x-?D|X-?D)', value = ' EMO_POS ',  regex = True)

    # Love -- <3, :*

    col = col.replace(to_replace = '(<3|:\*)', value = ' EMO_POS ', regex = True)

    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;

    col = col.replace(to_replace = '(;-?\)|;-?D|\(-?;)',value =  ' EMO_POS ', regex = True)

    # Sad -- :-(, : (, :(, ):, )-:

    col = col.replace(to_replace = '(:\s?\(|:-\(|\)\s?:|\)-:)',value =  ' EMO_NEG ', regex = True)

    # Cry -- :,(, :'(, :"(

    col = col.replace(to_replace = '(:,\(|:\'\(|:"\()', value = ' EMO_NEG ', regex = True)

    

    return col
#convert to lower case

df['user_review'] = df['user_review'].str.lower()

#Remove Urls

df['user_review'] = df['user_review'].str.replace('http\S+|www.\S+', '', case=False)





#remove #,@

df['user_review'] = df['user_review'].str.replace('[@#]', '')



#replace & with and

df['user_review'] = df['user_review'].str.replace('&', 'and')

#handle emojis

df['user_review'] = handle_emojis(df['user_review'])

#remove multiple spaces with single space

df['user_review'] = df['user_review'].str.replace(r'\s+', ' ')

#Remove punchutions

df['user_review'] = df['user_review'].str.replace('[\'"?!,.():;]','')

df['user_review'] = df['user_review'].replace(to_replace = r'(.)\1+',value = r'\1\1',regex = True)

df['user_review'] = df['user_review'].replace(to_replace = r'♥♥',value = r' love ',regex = True)

df['user_review'][34]
df['user_review'][77] #♥


max_features = 20000

maxlen = 100
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['user_review'])
tokenized_train = tokenizer.texts_to_sequences(df['user_review'])

X_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
X_train[0]
EMBEDDING_FILE = r'../input/glovetwitter27b100dtxt/glove.twitter.27B.100d.txt'

import numpy as np

def get_coefs(word, *arr): 

    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

#change below line if computing normal stats is too slow

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size)) #embedding_matrix = np.zeros((nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
del  tokenized_train, tokenizer, word_index, embeddings_index, all_embs, nb_words

gc.collect()
batch_size = 2048

epochs = 7

embed_size = 100
def lstm_model(conv_layers = 2, max_dilation_rate = 3):

    inp = Input(shape=(maxlen, ))

    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)

    x = Dropout(0.25)(x)

    x = Conv1D(2*embed_size, kernel_size = 3)(x)

    prefilt = Conv1D(2*embed_size, kernel_size = 3)(x)

    x = prefilt

    for strides in [1, 1, 2]:

        x = Conv1D(128*2**(strides), strides = strides, kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6), kernel_size=3, kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10))(x)

    x_f = LSTM(512, kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6), kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10))(x)  

    x_b = LSTM(512, kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6), kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10))(x)

    x = concatenate([x_f, x_b])

    x = Dropout(0.5)(x)

    x = Dense(64, activation="relu")(x)

    x = Dropout(0.1)(x)

    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['binary_accuracy'])



    return model



lstm_model = lstm_model()

lstm_model.summary()
weight_path="early_weights.hdf5"

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)

callbacks = [checkpoint, early_stopping]
lstm_model.fit(X_train, train_labels, batch_size=batch_size, epochs=epochs, shuffle = True, validation_split=0.20, callbacks=callbacks)