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




from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,AdaBoostClassifier

import seaborn as sns

from matplotlib import rcParams
train = pd.read_csv("/kaggle/input/fake-news/train.csv")

test = pd.read_csv("/kaggle/input/fake-news/test.csv")

submit = pd.read_csv("/kaggle/input/fake-news/submit.csv")
train.head()
train["label"].value_counts()
rcParams["figure.figsize"] = 10,8

sns.countplot(x = train["label"])
test=test.fillna(' ')

train=train.fillna(' ')

test['total']=test['title']+' '+test['author']+test['text']

train['total']=train['title']+' '+train['author']+train['text']
transformer = TfidfTransformer(smooth_idf=False)

count_vectorizer = CountVectorizer(ngram_range=(1, 2))

counts = count_vectorizer.fit_transform(train['total'].values)

tfidf = transformer.fit_transform(counts)
targets = train['label'].values

test_counts = count_vectorizer.transform(test['total'].values)

test_tfidf = transformer.fit_transform(test_counts)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(tfidf, targets, random_state=0)

Extr = ExtraTreesClassifier(n_estimators=5,n_jobs=4)

Extr.fit(X_train, y_train)

print('Accuracy of ExtrTrees classifier on training set: {:.2f}'

     .format(Extr.score(X_train, y_train)))

print('Accuracy of Extratrees classifier on test set: {:.2f}'

     .format(Extr.score(X_test, y_test)))
from sklearn.tree import DecisionTreeClassifier



Adab= AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=5)

Adab.fit(X_train, y_train)

print('Accuracy of Adaboost classifier on training set: {:.2f}'

     .format(Adab.score(X_train, y_train)))

print('Accuracy of Adaboost classifier on test set: {:.2f}'

     .format(Adab.score(X_test, y_test)))





Rando= RandomForestClassifier(n_estimators=5)



Rando.fit(X_train, y_train)

print('Accuracy of randomforest classifier on training set: {:.2f}'

     .format(Rando.score(X_train, y_train)))

print('Accuracy of randomforest classifier on test set: {:.2f}'

     .format(Rando.score(X_test, y_test)))



from sklearn.naive_bayes import MultinomialNB



NB = MultinomialNB()

NB.fit(X_train, y_train)

print('Accuracy of NB  classifier on training set: {:.2f}'

     .format(NB.score(X_train, y_train)))

print('Accuracy of NB classifier on test set: {:.2f}'

     .format(NB.score(X_test, y_test)))




from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=1e5)

logreg.fit(X_train, y_train)

print('Accuracy of Lasso classifier on training set: {:.2f}'

     .format(logreg.score(X_train, y_train)))

print('Accuracy of Lasso classifier on test set: {:.2f}'

     .format(logreg.score(X_test, y_test)))







targets = train['label'].values

logreg = LogisticRegression()

logreg.fit(counts, targets)



example_counts = count_vectorizer.transform(test['total'].values)

predictions = logreg.predict(example_counts)

pred=pd.DataFrame(predictions,columns=['label'])

pred['id']=test['id']

pred.groupby('label').count()



pred.to_csv('countvect5.csv', index=False)