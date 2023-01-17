# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

print(keras.__version__)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from math import nan

from keras.callbacks import ModelCheckpoint



!pip install git+https://www.github.com/keras-team/keras-contrib.git

from keras_contrib.layers import CRF



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dframe = pd.read_csv("../input/entity-annotated-corpus/ner.csv", encoding = "ISO-8859-1", error_bad_lines=False)
dframe
dframe.columns
dataset=dframe.drop(['Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos',

       'next-next-shape', 'next-next-word', 'next-pos', 'next-shape',

       'next-word', 'prev-iob', 'prev-lemma', 'prev-pos',

       'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 'prev-prev-shape',

       'prev-prev-word', 'prev-shape', 'prev-word',"pos"],axis=1)
dataset.info()
dataset.head()
dataset=dataset.drop(['shape'],axis=1)
dataset.head()
class SentenceGetter(object):

    

    def __init__(self, dataset):

        self.n_sent = 1

        self.dataset = dataset

        self.empty = False

        agg_func = lambda s: [(w, t) for w,t in zip(s["word"].values.tolist(),

                                                        s["tag"].values.tolist())]

        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)

        self.sentences = [s for s in self.grouped]

    

    def get_next(self):

        try:

            s = self.grouped["Sentence: {}".format(self.n_sent)]

            self.n_sent += 1

            return s

        except:

            return None
getter = SentenceGetter(dataset)
sentences = getter.sentences
print(sentences[5])
maxlen = max([len(s) for s in sentences])

print ('Maximum sequence length:', maxlen)
# Check how long sentences are so that we can pad them

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use("ggplot")
plt.hist([len(s) for s in sentences], bins=50)

plt.show()
words = list(set(dataset["word"].values))

words.append("ENDPAD")
n_words = len(words); n_words
tags = []

for tag in set(dataset["tag"].values):

    if tag is nan or isinstance(tag, float):

        tags.append('unk')

    else:

        tags.append(tag)

print(tags)
n_tags = len(tags); n_tags
from future.utils import iteritems

word2idx = {w: i for i, w in enumerate(words)}

tag2idx = {t: i for i, t in enumerate(tags)}

idx2tag = {v: k for k, v in iteritems(tag2idx)}
word2idx['Obama']
tag2idx["O"]
tag2idx
idx2tag[5]
idx2tag
from keras.preprocessing.sequence import pad_sequences

X = [[word2idx[w[0]] for w in s] for s in sentences]
np.array(X).shape
X = pad_sequences(maxlen=140, sequences=X, padding="post",value=n_words - 1)
y_idx = [[tag2idx[w[1]] for w in s] for s in sentences]

print(sentences[100])

print(y_idx[100])
y = pad_sequences(maxlen=140, sequences=y_idx, padding="post", value=tag2idx["O"])

print(y_idx[100])
from keras.utils import to_categorical

y = [to_categorical(i, num_classes=n_tags) for i in y]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from keras.models import Model, Input

from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

import keras as k
print(k.__version__)
input = Input(shape=(140,))

word_embedding_size = 300

model = Embedding(input_dim=n_words, output_dim=word_embedding_size, input_length=140)(input)

model = Bidirectional(LSTM(units=word_embedding_size, 

                           return_sequences=True, 

                           dropout=0.5, 

                           recurrent_dropout=0.5, 

                           kernel_initializer=k.initializers.he_normal()))(model)

model = LSTM(units=word_embedding_size * 2, 

             return_sequences=True, 

             dropout=0.5, 

             recurrent_dropout=0.5, 

             kernel_initializer=k.initializers.he_normal())(model)

model = TimeDistributed(Dense(n_tags, activation="relu"))(model)  # previously softmax output layer



crf = CRF(n_tags)  # CRF layer

out = crf(model)  # output
model = Model(input, out)
adam = k.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)

#model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

model.compile(optimizer=adam, loss=crf.loss_function, metrics=[crf.accuracy, 'accuracy'])
model.summary()
# Saving the best only

filepath="ner-bi-lstm-td-model-{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]
history = model.fit(X_train, np.array(y_train), batch_size=256, epochs=20, validation_split=0.2, verbose=1, callbacks=callbacks_list)
TP = {}

TN = {}

FP = {}

FN = {}

for tag in tag2idx.keys():

    TP[tag] = 0

    TN[tag] = 0    

    FP[tag] = 0    

    FN[tag] = 0    



def accumulate_score_by_tag(gt, pred):

    """

    For each tag keep stats

    """

    if gt == pred:

        TP[gt] += 1

    elif gt != 'O' and pred == 'O':

        FN[gt] +=1

    elif gt == 'O' and pred != 'O':

        FP[gt] += 1

    else:

        TN[gt] += 1

i = 357

p = model.predict(np.array([X_test[i]]))

p = np.argmax(p, axis=-1)

gt = np.argmax(y_test[i], axis=-1)

print(gt)

print("{:14}: ({:5}): {}".format("Word", "True", "Pred"))

for idx, (w,pred) in enumerate(zip(X_test[i],p[0])):

    #

    print("{:14}: ({:5}): {}".format(words[w],idx2tag[gt[idx]],tags[pred]))
p = model.predict(np.array(X_test))  
p.shape
from sklearn.metrics import classification_report
np.argmax(p, axis=2)[0]
print(classification_report(np.argmax(y_test, 2).ravel(), np.argmax(p, axis=2).ravel(),labels=list(idx2tag.keys()), target_names=list(idx2tag.values())))
for i, sentence in enumerate(X_test):

    y_hat = np.argmax(p[i], axis=-1)

    gt = np.argmax(y_test[i], axis=-1)

    for idx, (w,pred) in enumerate(zip(sentence,y_hat)):

        accumulate_score_by_tag(idx2tag[gt[idx]],tags[pred])
for tag in tag2idx.keys():

    print(f'tag:{tag}')    

    print('\t TN:{:10}\tFP:{:10}'.format(TN[tag],FP[tag]))

    print('\t FN:{:10}\tTP:{:10}'.format(FN[tag],TP[tag]))    