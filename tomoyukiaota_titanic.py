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
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
%matplotlib inline
train_df = pd.read_csv('../input/train.csv')
train_df.head()
train_df.info()
sns.heatmap(train_df.isnull(), cbar=False)
train_df['Survived'].value_counts()
sns.countplot(x='Survived', data=train_df, hue='Sex')
sns.countplot(x='Survived', data=train_df, hue='Pclass')
sns.distplot(train_df['Age'].dropna(), kde=False, bins=30)
sns.countplot(x='SibSp', data=train_df)
train_df['SibSp'].value_counts() / train_df['SibSp'].count()
train_df['Fare'].hist(bins=40)
sns.heatmap(train_df.isnull(), cbar=False)
sns.boxplot(x='Pclass', y='Age', data=train_df)
train_df.groupby('Pclass').mean()
def impute_age(columns):
    Age = columns[0]
    Pclass = columns[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 38.233441    # Average age for Pclass == 1
        elif Pclass == 2:
            return 29.877630    # Average age for Pclass == 2
        else:
            return 25.140620    # Average age for Pclass == 3
    else:
        return Age
train_df['Age'] = train_df[['Age', 'Pclass']].apply(impute_age, axis=1)
sns.heatmap(train_df.isnull(), cbar=False)
train_df.drop('Cabin', axis=1, inplace=True)
train_df.info()
train_df.dropna(inplace=True)
train_df.info()
sex = pd.get_dummies(train_df['Sex'], drop_first=True)
sex.head()
embark = pd.get_dummies(train_df['Embarked'], drop_first=True)
embark.head()
train_df = pd.concat([train_df, sex, embark], axis=1)
train_df.head()
train_df.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train_df.head()
train_df.drop('PassengerId', axis=1, inplace=True)
train_df.head()
test_df = pd.read_csv('../input/test.csv')
test_df.head()
test_df.info()
test_df['Fare'].mean()
test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)
test_df['Age'] = test_df[['Age', 'Pclass']].apply(impute_age, axis=1)
test_df.drop('Cabin', axis=1, inplace=True)
sex_test = pd.get_dummies(test_df['Sex'], drop_first=True)
embark_test = pd.get_dummies(test_df['Embarked'], drop_first=True)
test_df = pd.concat([test_df, sex_test, embark_test], axis=1)
test_df.head()
test_df.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
test_df.drop('PassengerId', axis=1, inplace=True)
test_df.head()
train_df.head()
X_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(test_df)
submission_df = pd.read_csv('../input/test.csv')
submission_df.head()
submission_df['Survived'] = predictions
submission_df[["PassengerId","Survived"]].to_csv("submission.csv",index=False)