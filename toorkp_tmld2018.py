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
# import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
x = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/gender_submission.csv')
x.head()
# looking for null values in training set
x.isnull().sum()
avgAge = x.Age.mean()
x.Age = x.Age.fillna(value = avgAge)
x.Age.isnull().sum()
x.Embarked.isnull().sum()
x.dropna(inplace = True)
x.isnull().sum()
# dropping the columns
#x_drop = x.drop(['PassengerId','Name','Ticket','Cabin'],1)
#x_drop.head()
x = pd.get_dummies(data = x, columns = ['Sex','Pclass','Embarked'])
x.head()
#x = x.drop(['PassengerId','Name','Ticket','Cabin'])
#x.head()
x = x.drop(['PassengerId','Name','Ticket','Cabin'],axis =1)#ropna(inplace = True)
x.head()
X = x.iloc[:,1:].values
y = x.iloc[:,0].values
'''
rfc = RandomForestClassifier
model = rfc(n_estimators = 100)
model.fit(X,y)
'''
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.2, random_state = 22)
#lr = LogisticRegression()
rfc = RandomForestClassifier
model = rfc(n_estimators = 100)
model_rfc = model.fit(X_train,y_train)
#y_pred = lr.predict(X_val)
y_pred = model_rfc.predict(X_val)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_val, y_pred)
confusion_matrix

#print(classification_report(y_val, y_pred))
from sklearn import metrics 
from sklearn.metrics import classification_report
print(classification_report(y_val,y_pred))
Train_Accuracy = accuracy_score(y_val, model_rfc.predict(X_val))
Train_Accuracy
test.head()
test.isnull().sum()
avgAge_test = test.Age.mean()
test.Age = test.Age.fillna(value = avgAge_test)
avgFare = test.Fare.mean()
test.Fare = test.Fare.fillna(value = avgFare)
test.Age.isnull().sum()
test.Fare.isnull().sum()
test_drop = test.drop(['PassengerId','Name','Ticket','Cabin'],axis =1)
test_drop.head()
test_dummy = pd.get_dummies(data = test_drop, columns = ['Sex','Pclass','Embarked'])
test_dummy.head()
y_test = model_rfc.predict(test_dummy)
test_Accuracy = accuracy_score(y_test, model_rfc.predict(test_dummy))
test_Accuracy
#sub = pd.to_csv(test['PassengerId'],y_test)
submission = pd.read_csv('../input/gender_submission.csv')
submission.head()
final = pd.DataFrame()
#final = pd.DataFrame(['PassengerId','Survived' == y_test])
final['PassengerId'] = test.PassengerId
final['Survived'] = y_test
final.head()
final.to_csv('sub.csv', index = False)
sub = pd.read_csv('sub.csv')
sub
