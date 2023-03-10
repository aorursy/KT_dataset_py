# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Sequential

from keras.layers.recurrent import GRU,SimpleRNN

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.embeddings import Embedding

from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D

from keras.preprocessing import sequence, text

from keras.callbacks import EarlyStopping





import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from plotly import graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff
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
#??????train,valid???test?????????

train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')

validation = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
validation.drop(['lang'],axis=1,inplace=True)
validation
#???train?????????????????????drop???

train.drop(['severe_toxic','obscene','threat','insult','identity_hate'],axis=1,inplace=True)
#?????????train?????????

train
#???25000???????????????????????????

train = train.loc[:19999,:]

train.shape
train['comment_text'].apply(lambda x:len(str(x).split())).max()
#AUC function,???AUC????????????????????????

def roc_auc(predictions,target):

    '''

    This methods returns the AUC Score when given the Predictions

    and Labels

    '''

    

    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)

    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc
#??????train???valid????????????

xtrain=train.comment_text.values

ytrain=train.toxic.values

xvalid=validation.comment_text.values

yvalid=validation.toxic.values

xtest=test.content.values
# using keras tokenizer here

token = text.Tokenizer(num_words=None)

max_len = 128 #???????????????????????????128



token.fit_on_texts(list(xtrain) ) #+ list(xvalid)+list(test)

xtrain_seq = token.texts_to_sequences(xtrain)

xvalid_seq = token.texts_to_sequences(xvalid)

xtest_seq = token.texts_to_sequences(xtest)



#zero pad the sequences

xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)

xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

xtest_pad = sequence.pad_sequences(xtest_seq, maxlen=max_len)





word_index = token.word_index

print(xtrain_pad.shape,xvalid_pad.shape,xtest_pad.shape)
#?????????train??????token????????????

xvalid_seq[:1]
# load the GloVe vectors in a dictionary:

#??????Word Embedding??????GloVe?????????????????????

embeddings_index = {}

f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt','r',encoding='utf-8')

for line in tqdm(f):

    values = line.split(' ')

    word = values[0]

    coefs = np.asarray([float(val) for val in values[1:]])

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
# create an embedding matrix for the words we have in the dataset

#???????????????????????????300????????????

embedding_matrix = np.zeros((len(word_index) + 1, 300))

for word, i in tqdm(word_index.items()):

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
#GRU??????

%time

with strategy.scope():

    # GRU with glove embeddings and two dense layers

     model = Sequential()

     model.add(Embedding(len(word_index) + 1,

                     300,

                     weights=[embedding_matrix],

                     input_length=max_len,

                     trainable=False))

     model.add(SpatialDropout1D(0.3))

     model.add(GRU(300,return_sequences=True))

     model.add(Dropout(0.2))

     model.add(GRU(300,return_sequences=False))

     model.add(Dropout(0.3))

     model.add(Dense(1, activation='sigmoid'))



     model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])   

    

model.summary()
#GRU????????????

model.fit(xtrain_pad, ytrain, nb_epoch=1, batch_size=64*strategy.num_replicas_in_sync)
#??????valid???????????????AUC??????

scores = model.predict(xvalid_pad)

print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))
sub = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")

sub['toxic'] = model.predict(xtest_pad, verbose=1)

sub.to_csv('submission.csv', index=False)

print("finish")