import matplotlib.pyplot as plt

import pandas as pd

import pylab as pl

import numpy as np

%matplotlib inline

import math

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        

        
# Load our warnings libraries

import warnings

warnings.filterwarnings('ignore')
# Read train data

train = pd.read_csv('/kaggle/input/titanic/train.csv')



# Read test data

test = pd.read_csv('/kaggle/input/titanic/test.csv')

train.head()
print("no. of passengers:" +str(len(train.index)))

#prints the total n0. of passengers
sns.countplot(x ="Survived", data = train)

#no. of survivived passengers
sns.countplot(x ="Survived",hue ="Sex", data = train)

#no. of male survived vs female survived
sns.countplot(x ="Survived",hue = "Pclass", data = train)

#survival passenger class (Pclass) wise
train["Age"].plot.hist()
train["Fare"].plot.hist()
train.info()

#gives the info of the data
sns.countplot(x ="SibSp", data = train)
train.isnull()

#checking null values
train.isnull().sum()

#sum of null values in each column
sns.boxplot(x="Pclass", y ="Age",data = train)

#pclass vs age box plot
train.head()
train.drop("Cabin", axis =1, inplace=True)
train.head()
train.dropna(inplace=True)

#droping all nan values
train.isnull().sum()

#checking if there is any nan values
sex =pd.get_dummies(train['Sex'], drop_first = True)

sex.head()

#there should not be any string values in data so get dummy values using get_dummies
embarked =pd.get_dummies(train['Embarked'], drop_first = True)

embarked.head()
pcl =pd.get_dummies(train['Pclass'], drop_first = True)

pcl.head()
train = pd.concat([train,sex,embarked,pcl],axis = 1)

#concat all three
train.head()
train.drop(['Sex', 'Embarked', 'Name', 'Ticket','Pclass'], axis=1,inplace=True)
train.head()
from sklearn.model_selection import train_test_split





X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived','PassengerId'], axis=1), 

                                                    train['Survived'], test_size = 0.2, 

                                                    random_state = 0)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
classification_report(y_test,predictions)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)
from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(X_train, y_train)

pred = gbk.predict(X_test)

acc_gbk = round(accuracy_score(pred, y_test) * 100, 2)

print(acc_gbk)

test.head()
test.drop("Cabin", axis =1, inplace=True)

#drop cabin because of nan values
sex1 =pd.get_dummies(test['Sex'], drop_first = True)

sex1.head()



# do the same what we have done to train data set, get dummy values .
embarked1 =pd.get_dummies(test['Embarked'], drop_first = True)

embarked1.head()
pcl1 =pd.get_dummies(test['Pclass'], drop_first = True)

pcl1.head()
test = pd.concat([test,sex1,embarked1,pcl1],axis = 1)
test.drop(['Sex', 'Embarked', 'Name', 'Ticket','Pclass'], axis=1,inplace=True)
test.head()
test.fillna(test.mean(), inplace=True)

#fill nan values with mean values.to avoid errors
# submit your predictions in csv format

ids = test['PassengerId']

predictions = gbk.predict(test.drop('PassengerId', axis=1))





output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)