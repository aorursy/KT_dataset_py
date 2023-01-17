import re

import os

from collections import Counter

import logging

import time

import itertools



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.manifold import TSNE

from sklearn.feature_extraction.text import TfidfVectorizer



import nltk

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer
# parameters

FILTER_STEM = True

TRAIN_PORTION = 0.8

RANDOM_STATE = 7
dataset_path = os.path.join("..","input" ,os.listdir("../input")[0])

df = pd.read_csv(dataset_path, encoding="ISO-8859-1", names=["target", "ids", "date", "flag", "user", "text"])
%%time

decode_map = {0: -1, 2: 0, 4: 1}

df.target = df.target.apply(lambda x: decode_map[x])
df.target.value_counts()
%%time

stop_words = stopwords.words("english")

stemmer = SnowballStemmer("english")

def filter_stopwords(text):

    text = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ', str(text).lower()).strip()

    if FILTER_STEM:

        return " ".join([stemmer.stem(token) for token in text.split() if token not in stop_words])

    else:

        return " ".join([token for token in text.split() if token not in stop_words])

df.text = df.text.apply(filter_stopwords)
%%time

vectorizer = TfidfVectorizer()

word_frequency = vectorizer.fit_transform(df.text)
# for not stem

len(vectorizer.get_feature_names())
len(vectorizer.get_feature_names())
sample_index = np.random.random(df.shape[0])

X_train, X_test = word_frequency[sample_index <= TRAIN_PORTION, :], word_frequency[sample_index > TRAIN_PORTION, :]

Y_train, Y_test = df.target[sample_index <= TRAIN_PORTION], df.target[sample_index > TRAIN_PORTION]

print(f"shape of training set: X={X_train.shape}, Y={Y_train.shape}")

print(f"shape of test set: X={X_test.shape}, Y={Y_test.shape}")
%%time

clf = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial').fit(X_train, Y_train)
Y_predit = clf.predict(X_test)

sum(Y_predit == Y_test) / len(Y_test)