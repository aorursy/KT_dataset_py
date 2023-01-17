# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importing the dataset

training_queries_labels = pd.read_csv('../input/training_queries_labels.csv')

training_queries = pd.read_csv('../input/training_queries.csv')

y = training_queries_labels.iloc[:, 1].values
#test file

from itertools import groupby

import csv



data = []



with open('../input/test.csv', newline='') as f_input:

    csv_input = csv.reader(f_input)

    header = next(csv_input)

    for row in csv_input:

        data.append(row[:2] )

        test = pd.DataFrame(data,columns=header[0:2]) 

test_index=test['index']
#combining train and find file

queries = pd.concat([training_queries,test],ignore_index=True)
 # Cleaning the texts

import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0,2233):

    review = re.sub('[^a-zA-Z]', ' ', queries ['query'][i])

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus.append(review)



# Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 5000)

X_combined = cv.fit_transform(corpus).toarray()
X=X_combined[:1808,:]

X_test=X_combined[1808:,:]
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Random forest to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators =200, criterion = 'gini', random_state = 0)



classifier.fit(X_train, y_train)



# Predicting the validation set results

y_pred = classifier.predict(X_valid)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_valid, y_pred)
from sklearn.metrics import accuracy_score 

#accuracy_score

score=accuracy_score(y_valid,y_pred)

print("score=%.4g"%score)
    

submission = pd.DataFrame()

submission['index'] = test_index

submission['tag'] = classifier.predict(X_test)

submission.to_csv('submission_rf.csv',index=False)