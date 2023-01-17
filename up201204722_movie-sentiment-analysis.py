# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Plotting labelled data
from nltk.corpus import stopwords # dealing with stop words
from textblob import TextBlob # dealing with spelling correction
from textblob import Word # dealing with lemmatization
from sklearn.feature_extraction.text import TfidfVectorizer # leading with term frequency

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.tsv', sep="\t")
test = pd.read_csv('../input/test.tsv', sep="\t")
submission = pd.read_csv('../input/sampleSubmission.csv', sep="\t")
train.shape
test.shape
train['Phrase'] = train['Phrase'].apply(lambda x: " ".join(x.lower() for x in x.split()))
train['Phrase'] = train['Phrase'].str.replace('[^\w\s]','')
stop = stopwords.words('english')
train['Phrase'] = train['Phrase'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
TextBlob(train['Phrase'][1]).words
TextBlob(test['Phrase'][1]).words
train['Phrase'] = train['Phrase'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
test['Phrase'] = test['Phrase'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
tfidf = TfidfVectorizer(ngram_range=(1,2))
train_vect = tfidf.fit_transform(train['Phrase'])
test_vect = tfidf.transform(test['Phrase'])
train_vect.shape
test_vect.shape
#train_features = hstack([train_vect])
#test_features = hstack([test_vect])
le=LabelEncoder()
y=le.fit_transform(train.Sentiment.values)
#y = pd.get_dummies(train.Sentiment)
#y.head()
lsv = LinearSVC()
nb = MultinomialNB()
lr = LogisticRegression(random_state=0)
X_train_vect, X_test_vect, y_train_vect, y_test_vect = train_test_split(train_vect, y, train_size=0.75)
nb.fit(X_train_vect, y_train_vect)
predictions_nb = nb.predict(X_test_vect)
accuracy = accuracy_score(y_test_vect, predictions_nb)
print(accuracy)
lr.fit(X_train_vect, y_train_vect)
predictions_lr = lr.predict(X_test_vect)
accuracy = accuracy_score(y_test_vect, predictions_lr)
print(accuracy)
lsv.fit(X_train_vect, y_train_vect)
predictions_lsv = lsv.predict(X_test_vect)
accuracy = accuracy_score(y_test_vect, predictions_lsv)
print(accuracy)
lsv.fit(train_vect, y)
predictions_lsv = lsv.predict(test_vect)
test['Sentiment'] = predictions_lsv
submission = test[["PhraseId","Sentiment"]]
submission.to_csv("Finalsubmission.csv", index = False)
submission.head()