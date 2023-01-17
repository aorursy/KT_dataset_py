# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import recall_score,precision_score,f1_score

from sklearn.neural_network import MLPClassifier

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train
train.isnull().sum()
train["Age"] = train["Age"].fillna(train["Age"].median())

train["Embarked"] = train["Embarked"].fillna("S")
train
train1 = train.drop(['PassengerId','Survived','Name','Ticket','Cabin'], axis=1)

col1 = [n for n in train1.columns if 

                                train1[n].nunique() < 10 and

                                train1[n].dtype == "object"]

col2 = [n for n in train1.columns if 

                                train1[n].dtype in ['int64', 'float64']]

cols = col1 + col2

train_pre = train1[cols]



dummy = pd.get_dummies(train1)
y_target = train["Survived"].values

x_features_one = dummy.values

FM = []
def treeDecision():

    print("Tree Decision")

    x_train, x_validation, y_train, y_validation = train_test_split(x_features_one,y_target,test_size=.20,random_state=1)

    tree1 = tree.DecisionTreeClassifier()

    tree1 = tree1.fit(x_features_one, y_target)

    predictions = tree1.predict(x_validation)

    recall = recall_score(y_validation,predictions)

    precision = precision_score(y_validation,predictions)

    f_measure = f1_score(y_validation,predictions)

    print("Recall",recall)

    print("Precision",precision)

    print("F Measure",f_measure)

    FM.append(f_measure)
def naive():

    print("Naive")

    x_train, x_validation, y_train, y_validation = train_test_split(x_features_one,y_target,test_size=.20,random_state=1)

    gnb = GaussianNB()

    gnb = gnb.fit(x_features_one, y_target)

    predictions = gnb.predict(x_validation)

    recall = recall_score(y_validation,predictions)

    precision = precision_score(y_validation,predictions)

    f_measure = f1_score(y_validation,predictions)

    print("Recall",recall)

    print("Precision",precision)

    print("F Measure",f_measure)

    FM.append(f_measure)
def neural():

    print("Neural")

    x_train, x_validation, y_train, y_validation = train_test_split(x_features_one,y_target,test_size=.20,random_state=1)

    clf = MLPClassifier()

    clf = clf.fit(x_features_one, y_target)

    predictions = clf.predict(x_validation)

    recall = recall_score(y_validation,predictions)

    precision = precision_score(y_validation,predictions)

    f_measure = f1_score(y_validation,predictions)

    print("Recall",recall)

    print("Precision",precision)

    print("F Measure",f_measure)

    FM.append(f_measure)
treeDecision()

naive()

neural()

print('AVERAGE F Measure ',sum(FM)/len(FM))