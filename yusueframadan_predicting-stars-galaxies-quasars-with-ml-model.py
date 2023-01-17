# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from tensorflow import keras

%matplotlib inline
data = pd.read_csv("/kaggle/input/sloan-digital-sky-survey/Skyserver_SQL2_27_2018 6_51_39 PM.csv")
data.head()
data.shape




# drop the object id columns, they are of no use in the analysis

data.drop(['objid','specobjid'], axis=1, inplace=True)



data.head(20)
data.shape
data.describe()
data.info()
le = LabelEncoder().fit(data['class'])

data['class'] = le.transform(data['class'])
data.head(20)
data.info()




X = data.drop('class', axis=1)

y = data['class']



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(copy=True , with_mean= True , with_std = True)

X= scaler.fit_transform(X)
#Show data

X[:20]




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=128)



sns.countplot(x=data['class'])




sns.pairplot(data[['u','g','r','i','class']])



# Decision Tree Classifier

dtClassifer = DecisionTreeClassifier(max_leaf_nodes=15,max_depth=3)

#------------------------------------------------------------------

#Linear Classifiers:

# 1- Logistic Regression

LRClassifer = LogisticRegression()

# # 2-Naive Bayes Classifier

# NBClassifer = MultinomialNB()

#-------------------------------------------------------------------

#Nearest Neighbor Classifier

NeNeClassifier = KNeighborsClassifier(n_neighbors=3)

#-------------------------------------------------------------------

#Support Vector Machines Classifer

SVCModel = SVC()





dtClassifer.fit(X_train, y_train)

LRClassifer.fit(X_train, y_train)

#NBClassifer.fit(X_train, y_train)

NeNeClassifier.fit(X_train, y_train)

SVCModel.fit(X_train, y_train)
y_preds = dtClassifer.predict(X_test)

y_predsLR = LRClassifer.predict(X_test)

#y_predsNB = NBClassifer.predict(X_test)

y_predsNeNe = NeNeClassifier.predict(X_test)

y_predsSVC = SVCModel.predict(X_test)
print(y_preds[:10],'\n',y_test[:10])

print("*******************************************************")

print(y_predsLR[:10],'\n',y_test[:10])

print("*******************************************************")

#print(y_predsNB[:10],'\n',y_test[:10])

#print("*******************************************************")

print(y_predsNeNe[:10],'\n',y_test[:10])

print("*******************************************************")

print(y_predsSVC[:10],'\n',y_test[:10])
print('accuracy_score by Decision Tree Classifier:',accuracy_score(y_true=y_test, y_pred=y_preds))

print('accuracy_score by LR Classifier:',accuracy_score(y_true=y_test, y_pred=y_predsLR))

#print('accuracy_score by Naive Bayes Classifier:',accuracy_score(y_true=y_test, y_pred=y_predsNB))

print('accuracy_score by Nearest Neighbor Classifier:',accuracy_score(y_true=y_test, y_pred=y_predsNeNe))

print('accuracy_score by SVM Classifier:',accuracy_score(y_true=y_test, y_pred=y_predsSVC))