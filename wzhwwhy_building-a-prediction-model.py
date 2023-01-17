%matplotlib inline



import sqlite3

import pandas as pd

import numpy as np

import nltk

import string

import matplotlib.pyplot as plt

import numpy as np



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cross_validation import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from nltk.stem.porter import PorterStemmer



con = sqlite3.connect('../input/database.sqlite')



messages = pd.read_sql_query("""

SELECT Score, Summary

FROM Reviews

WHERE Score != 3

""", con)



def partition(x):

    if x < 3:

        return 'negative'

    return 'positive'



Score = messages['Score']

Score = Score.map(partition)

Summary = messages['Summary']

X_train, X_test, y_train, y_test = train_test_split(Summary, Score, test_size=0.2)

len(X_train)
print(messages.head(20))
tmp = messages

tmp['Score'] = tmp['Score'].map(partition)

print(tmp.head(20))
stemmer = PorterStemmer()

from nltk.corpus import stopwords



def stem_tokens(tokens, stemmer):

    stemmed = []

    for item in tokens:

        stemmed.append(stemmer.stem(item))

    return stemmed



def tokenize(text):

    tokens = nltk.word_tokenize(text)

    tokens = [word for word in tokens if word not in stopwords.words('english')]

    stems = stem_tokens(tokens, stemmer)

    return ' '.join(stems)



intab = string.punctuation

outtab = "                                "

trantab = str.maketrans(intab, outtab)



#--- Training set



corpus = []

for text in X_train:

    text = text.lower()

    text = text.translate(trantab)

    text=tokenize(text)

    corpus.append(text)

        

tfid_vect = TfidfVectorizer()

X_train_tfidf = tfid_vect.fit_transform(corpus)        



#--- Test set



test_set = []

for text in X_test:

    text = text.lower()

    text = text.translate(trantab)

    text=tokenize(text)

    test_set.append(text)



X_test_tfidf = tfid_vect.transform(test_set)



from pandas import *

df = DataFrame({'Before': X_train, 'After': corpus})

print(df.head(20))



prediction = dict()
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB().fit(X_train_tfidf, y_train)

prediction['Multinomial'] = model.predict(X_test_tfidf)
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB().fit(X_train_tfidf, y_train)

prediction['Bernoulli'] = model.predict(X_test_tfidf)
from sklearn import linear_model

logreg = linear_model.LogisticRegression(C=1e5)

logreg.fit(X_train_tfidf, y_train)

prediction['Logistic'] = logreg.predict(X_test_tfidf)

print(metrics.classification_report(y_test, prediction['Logistic'], target_names = ["positive", "negative"]))
