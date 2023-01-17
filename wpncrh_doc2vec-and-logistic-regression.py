

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gensim

import os

import collections

import smart_open

import random



def read_corpus(fname, tokens_only=False):

    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:

        for i, line in enumerate(f):

            if tokens_only:

                yield gensim.utils.simple_preprocess(line)

            else:

                # For training data, add tags

                if i==0:

                    pass

                else:

                    yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line)[1:], 

                                                                                 gensim.utils.simple_preprocess(line)[0]) # tag is first item, the rest is text

                                                           

                                                           
all_data = list(read_corpus('../input/mbti_1.csv'))

total_num_obs = len(all_data)
# create train and test by just doing a 75/25% split

from math import floor, ceil

train_corpus = all_data[0:floor(3*total_num_obs/4)]

test_corpus = all_data[floor(3*total_num_obs/4):]
model = gensim.models.doc2vec.Doc2Vec(size=100, min_count=2, iter=55)
model.build_vocab(train_corpus) # remove infj, entp... to DO!
%time model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter) # 20 mins?
# play

model.infer_vector(['I', 'feel', 'sad'])
train_targets, train_regressors = zip(*[(doc.words, doc.tags[0]) for doc in train_corpus])

test_targets, test_regressors = zip(*[(doc.words, doc.tags[0]) for doc in test_corpus])
X = []

for i in range(len(train_targets)):

    X.append(model.infer_vector(train_targets[i]))

train_x = np.asarray(X)
train_x.shape
Y = np.asarray(train_regressors)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(Y)

train_y = le.transform(Y)

np.mean(train_y)
unique, counts = np.unique(Y, return_counts=True)



print(np.asarray((unique, counts)).T)
from sklearn import linear_model

logreg = linear_model.LogisticRegression()

logreg.fit(train_x, train_y)
test_list = []

for i in range(len(test_targets)):

    test_list.append(model.infer_vector(test_targets[i]))

test_x = np.asarray(test_list)
test_Y = np.asarray(test_regressors)

test_y = le.transform(test_Y)
preds = logreg.predict(test_x)
np.mean(test_y)
sum(preds == test_y) / len(test_y)