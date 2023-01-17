%matplotlib inline



from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Sequential

from keras.layers.recurrent import GRU,SimpleRNN,LSTM

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.embeddings import Embedding

from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D

from keras.preprocessing import sequence, text

from keras.callbacks import EarlyStopping



import sys

import os

import numpy as np

import pandas as pd

import IPython

import matplotlib.pyplot as plt

import seaborn as sns

from plotly import graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff
valid = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation-processed-seqlen128.csv")

train = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train-processed-seqlen128.csv")

test = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test-processed-seqlen128.csv")

submit = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")

train = train[['id', 'comment_text', 'input_word_ids', 'input_mask','all_segment_id', 'toxic']].iloc[:20000]
train.info()
train_distribution = train["toxic"].value_counts().values

valid_distribution = valid["toxic"].value_counts().values



non_toxic = [train_distribution[0] / sum(train_distribution) * 100, valid_distribution[0] / sum(valid_distribution) * 100]

toxic = [train_distribution[1] / sum(train_distribution) * 100, valid_distribution[1] / sum(valid_distribution) * 100]



plt.figure(figsize=(9,6))

plt.bar([0, 1], non_toxic, alpha=.4, color="r", width=0.35, label="non-toxic")

plt.bar([0.4, 1.4], toxic, alpha=.4, width=0.35, label="toxic")

plt.xlabel("Dataset")

plt.ylabel("Percentage")

plt.xticks([0.2, 1.2], ["train", "valid"])

plt.legend(loc="upper right")



plt.show()
print(f"train: \nnon-toxic rate: {train_distribution[0] / sum(train_distribution) * 100: .2f} %\ntoxic rate: {train_distribution[1] / sum(train_distribution) * 100: .2f} %")

print(f"valid: \nnon-toxic rate: {valid_distribution[0] / sum(valid_distribution) * 100: .2f} %\ntoxic rate: {valid_distribution[1] / sum(valid_distribution) * 100: .2f} %")
lang = valid["lang"].value_counts()



plt.figure(figsize=(9, 6))

plt.xlabel("Lang")

plt.ylabel("Num")

plt.xticks([0.2, 0.6, 1], ["tr", "es", "it"])

plt.bar([0.2, 0.6, 1], lang, color="purple", width=0.28, alpha=.4)

plt.show()
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
train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')

validation = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')



train.drop(['severe_toxic','obscene','threat','insult','identity_hate'], axis=1, inplace=True)
train = train.loc[:25000-1,:]

print(f"訓練資料總數：{train.shape[0]}\n句子最大長度：{train['comment_text'].apply(lambda x:len(str(x).split())).max()}")
def roc_auc(predictions,target):

  

    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)

    roc_auc = metrics.auc(fpr, tpr)

    

    return roc_auc
#取出train和valid所需資料

#將train中不必要的欄位drop掉

xtrain, xvalid, ytrain, yvalid = train_test_split(train.comment_text.values, train.toxic.values, 

                                                  stratify=train.toxic.values, 

                                                  random_state=42, 

                                                  test_size=0.2, shuffle=True)
token = text.Tokenizer(num_words=None)

max_len = 128



token.fit_on_texts(list(xtrain)+ list(xvalid)) #+ list(xvalid)+list(test)

xtrain_seq = token.texts_to_sequences(xtrain)

xvalid_seq = token.texts_to_sequences(xvalid)



xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)

xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)



word_index = token.word_index

print(xtrain_pad.shape, xvalid_pad.shape)

embeddings_index = {}

f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt','r',encoding='utf-8')

for line in tqdm(f):

    values = line.split(' ')

    word = values[0]

    coefs = np.asarray([float(val) for val in values[1:]])

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((len(word_index) + 1, 300))

for word, i in tqdm(word_index.items()):

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
%%time

with strategy.scope():

    

    model = Sequential()

    model.add(Embedding(len(word_index) + 1,

                     300,

                     weights=[embedding_matrix],

                     input_length=max_len,

                     trainable=False))



    model.add(LSTM(300, dropout=0.3, recurrent_dropout=0.3,return_sequences=True))

    model.add(LSTM(300, dropout=0.3, recurrent_dropout=0.3,return_sequences=True))

    model.add(LSTM(300, dropout=0.3, recurrent_dropout=0.3))

    model.add(Dense(1, activation='sigmoid'))

    

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    

model.summary()
%%time



model.fit(xtrain_pad, ytrain, epochs=5, batch_size=48*strategy.num_replicas_in_sync)
scores = model.predict(xvalid_pad)

print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))
%%time

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

     model.add(Dropout(0.3))

     model.add(GRU(300,return_sequences=True))

     model.add(Dropout(0.3))

     model.add(GRU(300,return_sequences=False))

     model.add(Dropout(0.3))

     model.add(Dense(1, activation='sigmoid'))



     model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])   

    

model.summary()
#GRU模型訓練

model.fit(xtrain_pad, ytrain, epochs=5, batch_size=64*strategy.num_replicas_in_sync)
#使用valid測試資料的AUC結果

scores = model.predict(xvalid_pad)

print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))