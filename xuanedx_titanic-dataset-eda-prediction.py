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
#import data sets
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
#EDA
import matplotlib as plt
import seaborn as sns
train_data.head()
train_data.info()
#Survival by Class
sns.countplot('Survived', hue='Pclass', data=train_data,palette='Set1')
#Survived by Age
train_data[train_data['Survived']==0]['Age'].hist(bins=30,color='red',label='Survived=0')
train_data[train_data['Survived']==1]['Age'].hist(bins=30,color='blue',label='Survived=1')
#Survived by Sex
sns.countplot('Survived', hue='Sex', data=train_data,palette='Set1')
#Machine Learning Prediction
from sklearn.ensemble import RandomForestClassifier

#Cleaning data
X_train = pd.get_dummies(train_data,columns=['Sex','Ticket','Cabin','Embarked'],drop_first=True)
X_test = pd.get_dummies(test_data,columns=['Sex','Ticket','Cabin','Embarked'],drop_first=True)
y_train = X_train['Survived']
X_train = X_train.drop('Survived', axis=1)

rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
prediction = rfc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
