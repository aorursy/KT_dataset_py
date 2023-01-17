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
#load dataset
data = '../input/titanicdataset-traincsv/train.csv'
dataset = pd.read_csv(data)
dataset.shape
dataset.dtypes
dataset.describe()
dataset.head(10)
print("Total number of passengers in the dataset: " + str(len(dataset.index)))
sns.countplot(x="Survived", data=dataset)
sns.countplot(x="Survived", hue="Sex", data=dataset)
sns.countplot(x="Survived", hue="Pclass", data=dataset)
dataset["Age"].plot.hist()
sns.boxplot(x="Survived", y="Age", data=dataset)
dataset["Pclass"].plot.hist()
sns.boxplot(x="Pclass", y="Age", data=dataset)
dataset["Fare"].plot.hist(figsize=(10,10))
dataset.info()
sns.countplot(x="SibSp", data=dataset)
sns.countplot(x="Parch", data=dataset)
dataset.isnull()
dataset.isnull().sum()
sns.heatmap(dataset.isnull(), yticklabels=False, cmap="viridis")
dataset.head()
#dropping 'Cabin' feature
dataset.drop("Cabin", axis=1, inplace=True)
#check if the 'Cabin' feature is dropped
dataset.head()
sns.heatmap(dataset.isnull(), yticklabels=False, cmap="viridis")
#dropping rows of data with 'Age' null
dataset.dropna(inplace=True)
sns.heatmap(dataset.isnull(), yticklabels=False, cmap="viridis")
#checking if any more null value in the dataset
dataset.isnull().sum()
dataset.shape
dataset.head()
dataset.Pclass.unique()
dataset.Sex.unique()
dataset.Embarked.unique()
Pcl=pd.get_dummies(dataset["Pclass"],drop_first=True)
Pcl.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X = dataset
a = dataset['Sex']

X['Sex'] = le.fit_transform(X['Sex'])

a = le.transform(a)
dataset = X
embark=pd.get_dummies(dataset["Embarked"])
embark.head()
dataset=pd.concat([dataset,embark,Pcl],axis=1)
dataset.head()
dataset.drop(['PassengerId','Pclass', 'Name','Ticket','Embarked'],axis=1, inplace=True)
dataset.head()
#get correlation map
corr_mat=dataset.corr()
#visualise data
plt.figure(figsize=(13,5))
sns_plot=sns.heatmap(data=corr_mat, annot=True, cmap='GnBu')
plt.show()
# Train
X = dataset.drop("Survived", axis=1)
y = dataset["Survived"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(predictions)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
from sklearn.metrics import accuracy_score 
accuracy_score(y_test,predictions) #(0+1)/(0+1+1+3) = 0.2