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
traindf = pd.read_csv('../input/train.csv',index_col = 0)
traindf.head()
traindf.info()
traindf['Age'] = traindf['Age'].fillna(traindf['Age'].mean())
traindf.info()
traindf = traindf.drop('Cabin',axis=1)
traindf.head()
male = pd.get_dummies(traindf['Sex'],drop_first = True)
embarked = pd.get_dummies(traindf['Embarked'],drop_first = True)
traindf['Male'] = male
traindf.head()
traindf = traindf.drop(['Sex','Embarked','Ticket','Name'],axis=1)
traindf.head()
traindf.info()
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(traindf.drop('Survived',axis=1),traindf['Survived'])
testdf = pd.read_csv('../input/test.csv',index_col = 0)
testdf.info()
testdf['Age'] = testdf['Age'].fillna(testdf['Age'].mean())
testdf.info()
testdf['Fare'] = testdf['Fare'].fillna(testdf['Fare'].mean())
testdf.info()
testdf = testdf.drop('Cabin',axis=1)
male = pd.get_dummies(testdf['Sex'],drop_first = True)
embarked = pd.get_dummies(testdf['Embarked'],drop_first = True)
testdf['Male'] = male
testdf.head()
testdf = testdf.drop(['Sex','Embarked','Ticket','Name'],axis=1)
testdf.head()
pred = logreg.predict(testdf)
res = pd.read_csv('../input/gender_submission.csv',index_col = 0)
res.head()
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
accuracy_score(res,pred)

confusion_matrix(res,pred)
res1 = res['Survived'].tolist()
accuracy_score(res1,pred)
print(res1)
print(pred)
