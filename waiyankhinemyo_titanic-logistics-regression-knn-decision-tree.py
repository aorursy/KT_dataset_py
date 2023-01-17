#import libraries 

#structures
import pandas as pd
import numpy as np

#visualization
import matplotlib.pyplot as plt
%matplotlib inline
import math
import seaborn as sns
sns.set()
from mpl_toolkits.mplot3d import Axes3D

#get model duration
import time
from datetime import date

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#load train dataset
train_data = '../input/titanic/train.csv'
train_dataset = pd.read_csv(train_data)
train_dataset.shape
print("Total number of passengers in the train dataset: " + str(len(train_dataset.index)))
test_data = '../input/titanic/test.csv'
test_dataset = pd.read_csv(test_data)
test_dataset.shape
print("Total number of passengers in the test dataset: " + str(len(test_dataset.index)))
train_dataset.dtypes
test_dataset.dtypes
train_dataset.describe()
test_dataset.describe()
train_dataset.head(10)
test_dataset.head(10)
sns.countplot(x="Survived", data=train_dataset)
sns.countplot(x="Survived", hue="Sex", data=train_dataset)
sns.countplot(x="Survived", hue="Pclass", data=train_dataset)
total_dataset = pd.concat([train_dataset, test_dataset])
total_dataset.shape
total_dataset.head()
total_dataset.tail()
train_dataset["Age"].plot.hist()
test_dataset["Age"].plot.hist()
total_dataset["Age"].plot.hist()
sns.boxplot(x="Survived", y="Age", data=train_dataset)
train_dataset["Pclass"].plot.hist()
test_dataset["Pclass"].plot.hist()
total_dataset["Pclass"].plot.hist()
sns.boxplot(x="Pclass", y="Age", data=train_dataset)
sns.boxplot(x="Pclass", y="Age", data=test_dataset)
sns.boxplot(x="Pclass", y="Age", data=total_dataset)
train_dataset["Fare"].plot.hist(figsize=(10,10))
test_dataset["Fare"].plot.hist(figsize=(10,10))
total_dataset["Fare"].plot.hist(figsize=(10,10))
train_dataset.info()
test_dataset.info()
total_dataset.info()
sns.countplot(x="SibSp", data=train_dataset)
sns.countplot(x="SibSp", data=test_dataset)
sns.countplot(x="SibSp", data=total_dataset)
sns.countplot(x="Parch", data=train_dataset)
sns.countplot(x="Parch", data=test_dataset)
sns.countplot(x="Parch", data=total_dataset)
total_dataset.isnull()
train_dataset.isnull().sum()
test_dataset.isnull().sum()
sns.heatmap(train_dataset.isnull(), yticklabels=False, cmap="viridis")
sns.heatmap(test_dataset.isnull(), yticklabels=False, cmap="viridis")
train_dataset.head()
test_dataset.head()
#dropping 'Cabin' feature
train_dataset.drop("Cabin", axis=1, inplace=True)
test_dataset.drop("Cabin", axis=1, inplace=True)
#check if the 'Cabin' feature is dropped
train_dataset.head()
test_dataset.head()
sns.heatmap(train_dataset.isnull(), yticklabels=False, cmap="viridis")
sns.heatmap(test_dataset.isnull(), yticklabels=False, cmap="viridis")
#dropping rows of data with 'Age' null
train_dataset.dropna(inplace=True)
test_dataset.describe()
#replacing null values with average values
test_dataset['Age'].fillna((test_dataset['Age'].mean()), inplace=True)
test_dataset['Fare'].fillna((test_dataset['Fare'].mean()), inplace=True)
sns.heatmap(train_dataset.isnull(), yticklabels=False, cmap="viridis")
sns.heatmap(test_dataset.isnull(), yticklabels=False, cmap="viridis")
#checking if any more null value in the dataset
train_dataset.isnull().sum()
test_dataset.isnull().sum()
train_dataset.shape
test_dataset.shape
train_dataset.head()
test_dataset.head()
train_dataset.Pclass.unique()
test_dataset.Pclass.unique()
train_dataset.Sex.unique()
test_dataset.Sex.unique()
train_dataset.Embarked.unique()
test_dataset.Embarked.unique()
Pcl_train=pd.get_dummies(train_dataset["Pclass"],drop_first=True)
Pcl_test=pd.get_dummies(test_dataset["Pclass"],drop_first=True)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X1 = train_dataset
a = train_dataset['Sex']

X1['Sex'] = le.fit_transform(X1['Sex'])

a = le.transform(a)
train_dataset = X1
X2 = test_dataset
b = test_dataset['Sex']

X2['Sex'] = le.fit_transform(X2['Sex'])

b = le.transform(b)
test_dataset = X2
embark_train=pd.get_dummies(train_dataset["Embarked"])
embark_test=pd.get_dummies(test_dataset["Embarked"])
train_dataset=pd.concat([train_dataset,embark_train,Pcl_train],axis=1)
train_dataset.head()
test_dataset=pd.concat([test_dataset,embark_test,Pcl_test],axis=1)
test_dataset.head()
train_dataset.drop(['Pclass', 'Name','Ticket','Embarked'],axis=1, inplace=True)
train_dataset.head()
test_dataset.drop(['Pclass', 'Name','Ticket','Embarked'],axis=1, inplace=True)
test_dataset.head()
#get correlation map
corr_mat=train_dataset.corr()
#visualise data
plt.figure(figsize=(13,5))
sns_plot=sns.heatmap(data=corr_mat, annot=True, cmap='GnBu')
plt.show()
#to run for model without scaling
dropped_passengerId = train_dataset.drop("PassengerId", axis=1)

X_train = dropped_passengerId.drop("Survived", axis=1)
y_train = train_dataset["Survived"]

X_test = test_dataset.drop("PassengerId", axis=1)
#to run for model with scaling
dropped_passengerId = train_dataset.drop("PassengerId", axis=1)

dropped_survived = dropped_passengerId.drop("Survived", axis=1)

dropped_survived.head()
test_dropped_passengerId = test_dataset.drop("PassengerId", axis=1)
test_dropped_passengerId.head()
X_train = dropped_survived.iloc[:,0:10]
y_train = train_dataset["Survived"]

X_test = test_dropped_passengerId.iloc[:,0:10]
from sklearn.preprocessing import StandardScaler
#stadardize data
X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)
#get feature names
X_train_columns = train_dataset.columns[:10]
X_test_columns = test_dataset.columns[:10]
from sklearn.preprocessing import MinMaxScaler
#stadardize data
X_train_scaled = MinMaxScaler().fit_transform(X_train)
X_test_scaled = MinMaxScaler().fit_transform(X_test)
#get feature names
X_train_columns = train_dataset.columns[:10]
X_test_columns = test_dataset.columns[:10]
from sklearn.preprocessing import RobustScaler
#stadardize data
X_train_scaled = RobustScaler().fit_transform(X_train)
X_test_scaled = RobustScaler().fit_transform(X_test)
#get feature names
X_train_columns = train_dataset.columns[:10]
X_test_columns = test_dataset.columns[:10]
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(predictions)
output = pd.DataFrame({'PassengerId': test_dataset.PassengerId, 'Survived': predictions})
output.to_csv('my_submission_noscaling.csv', index=False)
print("Your submission was successfully saved!")
import math
math.sqrt(len(X_test))
from sklearn.neighbors import KNeighborsClassifier
knnmodel = KNeighborsClassifier(n_neighbors=21, p=2, metric='euclidean') #p is 2 cuz we are looking for survived or not: 2 results
#Fit Model
knnmodel.fit(X_train_scaled, y_train)
#predict the test set results
predictions = knnmodel.predict(X_test_scaled)
print(predictions)
output = pd.DataFrame({'PassengerId': test_dataset.PassengerId, 'Survived': predictions})
output.to_csv('my_submission_standard.csv', index=False)
print("Your submission was successfully saved!")
from sklearn.tree import DecisionTreeRegressor
decisionmodel = DecisionTreeRegressor()
#Fit Model
decisionmodel.fit(X_train_scaled, y_train)
#predict the test set results
predictions = decisionmodel.predict(X_test_scaled)
print(predictions)
output = pd.DataFrame({'PassengerId': test_dataset.PassengerId, 'Survived': predictions})
output.to_csv('my_submission_robust.csv', index=False)
print("Your submission was successfully saved!")