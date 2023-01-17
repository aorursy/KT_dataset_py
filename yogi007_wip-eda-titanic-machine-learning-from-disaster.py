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
# importing data
DATA_PATH = '../input/'
train_data = pd.read_csv(DATA_PATH + 'train.csv')
test_data = pd.read_csv(DATA_PATH + 'test.csv')
gender_submission_data = pd.read_csv(DATA_PATH + 'gender_submission.csv')
print("Train Shape", train_data.shape)
print("Test Shape", test_data.shape)
print("Submission Shape", gender_submission_data.shape)
train_data.head()
test_data.head()
gender_submission_data.head()
survival_by_gender = train_data.groupby('Sex')['Survived'].sum()
print(survival_by_gender)

# Here I want to keep our actual data as it is. Thus making a copy of it.
train_data_copy = train_data
print("%s : %d " % ('Ticket Count', train_data['Ticket'].unique().size))
print('Pclass : ', train_data['Pclass'].unique())
print("%s : %d " % ('Age Count',train_data['Age'].unique().size))
print('SibSp : ', train_data['SibSp'].unique())
print('Parch : ', train_data['Parch'].unique())
print("%s : %d " % ('Fares', train_data['Fare'].unique().size))
print("%s : %d " % ('Cabin', train_data['Cabin'].unique().size))
print('Embarked : ', train_data['Embarked'].unique())


train_data['Cabin'].isnull().sum()
train_data['Embarked'].isnull().sum()
train_data['Age'].isnull().sum()
train_data['Ticket'].isnull().sum()
train_data['Age'].median()


y_train = train_data['Survived'].values
X_train = train_data.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']]
mean_age = X_train['Age'].mean()
X_train.loc[:,'Age'] = X_train.loc[:,'Age'].fillna(mean_age)

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_train = LabelEncoder()
# X_train = labelencoder_train.fit_transform(X_train.loc[:, ['Pclass', 'Sex', 'SibSp', 'SibSp', 'Embarked']].values)
# labelencoder_train = LabelEncoder()
# X_train = labelencoder_train.fit_transform(X_train.loc[:, ['Pclass', 'Sex', 'SibSp', 'SibSp', 'Embarked']].values)
# onehotencoder = OneHotEncoder(categorical_features = ['Pclass', 'Sex', 'SibSp', 'SibSp', 'Embarked'])
# train_data = onehotencoder.fit_transform(X_train.loc[:, ['Pclass', 'Sex', 'SibSp', 'SibSp', 'Embarked']].values).toarray()