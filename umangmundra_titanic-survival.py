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
import pandas as pd

import numpy as np

import random as rnd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import roc_auc_score
train_df = pd.read_csv("../input/titanic/train.csv")

test_df = pd.read_csv("../input/titanic/test.csv")

ytest = pd.read_csv("../input/titanic/gender_submission.csv")
# Let's look at the dataset.

train_df.head()
train_df.info()

print("-"*50)

test_df.info()
# Create a new column by joining Parch and SibSp

train_df['Family'] =  train_df["Parch"] + train_df["SibSp"]

test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]
# Drop the irrelavent columns.

train_df.drop(['Cabin', 'Ticket', 'Name', 'Parch', 'SibSp'], axis=1, inplace=True)

test_df.drop(['Cabin', 'Ticket', 'Name', 'Parch', 'SibSp'], axis=1, inplace=True)

ytest.drop(['PassengerId'], axis=1, inplace=True)
# Filling missing values by Fillna method.

train_df['Embarked'] = train_df['Embarked'].fillna(value='S')
# Missing value treatment of train data

train_df.interpolate(limit_direction='both', inplace=True)
# Missing value treatment of test data

test_df.interpolate(limit_direction='both', inplace=True)
# Changing the dtype

test_df['Fare'] = test_df['Fare'].interpolate(limit_direction='both').astype(int)
train_df.info()
# Visualizing the Dataset
# Count of Family members.

sns.countplot(x='Family', data=train_df)
# People Survived according to Embark.

#grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid = sns.FacetGrid(train_df, row='Embarked', col='Survived')

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

#grid.map(sns.barplot, 'Sex', 'Fare')

grid.add_legend()
# Checking Male Female Surviving ratio

sns.barplot(train_df['Sex'], train_df['Survived'])
# Checking the people of which class has more chance of surviving

sns.barplot(train_df['Pclass'], train_df['Survived'])
train_df.groupby(['Embarked']).count()
train_df.groupby(['Embarked', 'Sex']).count()
sns.barplot(train_df['Embarked'], train_df['Survived'])
# Changing Age dtype in train and test data

train_df['Age'] = train_df['Age'].astype(int)

test_df['Age'] = test_df['Age'].astype(int)
# Encoding Categorical Columns

lenc = LabelEncoder()

train_df['Sex'] = lenc.fit_transform(train_df['Sex'])

train_df['Embarked'] = lenc.fit_transform(train_df['Embarked'])

test_df['Sex'] = lenc.fit_transform(test_df['Sex'])

test_df['Embarked'] = lenc.fit_transform(test_df['Embarked'])
train_df.info()
# Dividing xtrain and ytrain for fitting.

xtrain = train_df.drop(['Survived'], axis=1)

ytrain = train_df['Survived']

xtest = test_df
lr = LogisticRegression(solver='liblinear')
lr.fit(xtrain, ytrain)
lr.score(xtrain, ytrain)
acc_log = round(lr.score(xtrain, ytrain) * 100, 2)

acc_log
ypred = lr.predict(xtest)

ypred
test_df
acc = accuracy_score(ytest, ypred)

acc
confusion_matrix(ytest, ypred)
# Classification Report

print(classification_report(ytest, ypred))
ypred_prob = lr.predict_proba(xtest)
# AUC ROC Curve

lr_probs = ypred_prob[:, 1]

ns_probs = [0 for _ in range(len(ytest))]

ns_auc = roc_auc_score(ytest, ns_probs)

lr_auc = roc_auc_score(ytest, lr_probs)

print(lr_auc)

ns_fpr, ns_tpr, _ = roc_curve(ytest, ns_probs)

lr_fpr, lr_tpr, _ = roc_curve(ytest, lr_probs)

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')

plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend()

plt.show()
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':ypred})



#Visualize the first 5 rows

submission.head(10)
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'Titanic_Survival.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)