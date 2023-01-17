# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from pylab import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv('../input/spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data.head()

# Any results you write to the current directory are saved as output.
classification = {"ham":0,"spam":1}
data['label']=data['label'].map(classification)
data.head()
X_train, X_test, y_train, y_test = train_test_split(data['message'],data['label'],test_size=0.2)
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(X_train)
classifier = MultinomialNB()
classifier.fit(counts,y_train)
counts_test = vectorizer.transform(X_test)
prediction = classifier.predict(counts_test)
print('Accuracy score: {}'.format(accuracy_score(y_test, prediction)))
print('Precision score: {}'.format(precision_score(y_test, prediction)))
print('Recall score: {}'.format(recall_score(y_test, prediction)))
print('F1 score: {}'.format(f1_score(y_test, prediction)))