# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/train.csv")


#show the info of train data
train.info()
#show top5 samples of data
train.head()
#show total null values in data
train.isnull().sum()
#show the heatmap for null values in data
sns.heatmap(train.isnull(),cmap="YlGnBu",cbar=False,yticklabels=False)
#fill null values of age by median
age_median = train.Age.median()
print(age_median)
train["Age"]=train.Age.fillna(age_median)
train.Age.describe()


bins=[0,10,18,30,60,np.inf]
labels=['Child','Teenager','Young','Adult','Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

#too many null values, so drop it
train=train.drop("Cabin",axis=1)
train.head()
train.Embarked.value_counts()

#use the mostly value to fill the embarked
train=train.fillna({"Embarked":"S"})
train[train.Embarked.isnull()].Embarked
#show survived
sns.countplot(data=train,x="Survived")
#relationshio between Pclass and Fare.
sns.barplot(data=train,x="Pclass",y="Fare",ci=None)
#show the relationship between class and survived.The survived of 1st is the highest
sns.barplot(data=train,x="Pclass",y="Survived",ci=None)
#show the relationship between class and survived.The survived of female is the highest
sns.barplot(data=train,x="Sex",y="Survived",ci=None)
#show the relationship for class,sex and survived
sns.pointplot(data=train,x="Pclass",y="Survived",hue="Sex",ci=None)
#show the relationship for age and survived
plt.figure(figsize= (10 ,5))
sns.set_style("whitegrid")
sns.barplot(data=train,x="AgeGroup",y="Survived",ci=None)
plt.xticks(rotation=60)
plt.show()
#show the relationship for age,sex and survived
plt.figure(figsize= (10 ,5))
sns.set_style("whitegrid")
sns.pointplot(data=train,x="AgeGroup",y="Survived",hue="Sex",ci=None)
plt.xticks(rotation=60)
plt.show()
train.head()
age_mapping = {'Child':1,'Teenager':2,'Young':3,'Adult':4,'Senior':5}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
train = train.drop(['Age'], axis = 1)
train.head()
train=train.drop(['Name','Ticket'],axis=1)
train.head()
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
train.head()
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
train.head()
test=test.drop(['Name','Ticket','Cabin'],axis=1)

test.head()
#fill null values of age by median
age_median = test.Age.median()
print(age_median)
test["Age"]=test.Age.fillna(age_median)
test.Age.describe()
bins=[0,10,18,30,60,np.inf]
labels=['Child','Teenager','Young','Adult','Senior']
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)
age_mapping = {'Child':1,'Teenager':2,'Young':3,'Adult':4,'Senior':5}
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
test = test.drop(['Age'], axis = 1)
test.head()
sex_mapping = {"male": 0, "female": 1}
test['Sex'] = test['Sex'].map(sex_mapping)
test.head()
test=test.fillna({"Embarked":"S"})
test[test.Embarked.isnull()].Embarked
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
test['Embarked'] = test['Embarked'].map(embarked_mapping)
test.head()
#mechine learning
from sklearn.model_selection import train_test_split
features = train.drop(['Survived', 'PassengerId'], axis=1)
target_labels=train["Survived"]
X_train, X_val, Y_train, Y_val = train_test_split(features, target_labels, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,f1_score
LogisticReg =LogisticRegression()
LogisticReg.fit(X_train, Y_train)
y_pred = LogisticReg.predict(X_val)
print("ACC:",accuracy_score(Y_val,y_pred))
print("REC:",recall_score(Y_val,y_pred))
print("F1:",f1_score(Y_val,y_pred))