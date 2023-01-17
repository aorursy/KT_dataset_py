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
# for plotting

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Read the data

train_data_file = "../input/titanic/train.csv"

train_dt1 = pd.read_csv(train_data_file)

test_data_file = "../input/titanic/test.csv"

test_dt1 = pd.read_csv(test_data_file)
# Save the PassengerId to be used while submitting at the end

orig_test_dt = pd.DataFrame()

orig_test_dt['PassengerId'] = test_dt1['PassengerId']
train_dt1.head()
train_dt1.shape
train_dt1.describe(include = "all")
corr = train_dt1.describe(include="all").corr()

plt.figure(figsize=(12, 10))

sns.heatmap(corr, annot=True, cmap="summer")

plt.show()
# PassengerID, Name, Ticket number can't help in prediction, so removing them.
train_dt1.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
train_dt1.isnull().sum().sort_values(ascending = False)
# Cabin has many NaN values. Since it can't help in prediction, removing them.
train_dt1.drop("Cabin", axis=1, inplace=True)
# Encode the predictors Sex, Embarked, Pclass. Also fill NaN values with the right value.
train_dt1['Sex'] = np.where(train_dt1['Sex'] == "male", 1, 0)
train_dt1['Embarked'] = train_dt1['Embarked'].fillna(train_dt1['Embarked'].mode())

embarked = pd.get_dummies(train_dt1['Embarked'], prefix = 'Embarked_')

train_dt1.drop("Embarked", axis=1, inplace=True)

train_dt1 = train_dt1.join(embarked)
pclass = pd.get_dummies(train_dt1['Pclass'], prefix="Pclass_")

train_dt1.drop("Pclass", axis=1, inplace=True)

train_dt1 = train_dt1.join(pclass)
train_dt1['Age'] = train_dt1['Age'].fillna(train_dt1['Age'].median())
train_dt1['SibSp'] = train_dt1['SibSp'].fillna(0)
train_dt1['Parch'] = train_dt1['Parch'].fillna(0)
# Pclass is a better predictor than Fare as Pclass defines the social status of people and may help in predicting which Pclass people survived more.

# So, removing Fare predictor to avoid multi-collinearity problem.

train_dt1.drop('Fare', axis=1, inplace=True)
#downcasting some attributes' size to save some memory
train_dt1['Sex'] = pd.to_numeric(train_dt1['Sex'], downcast='unsigned')

train_dt1['Age'] = pd.to_numeric(train_dt1['Age'], downcast='float')

train_dt1['SibSp'] = pd.to_numeric(train_dt1['SibSp'], downcast='unsigned')

train_dt1['Parch'] = pd.to_numeric(train_dt1['Parch'], downcast='unsigned')
# Draw correlation plot with updated features
corr = train_dt1.describe(include="all").corr()

plt.figure(figsize=(12, 10))

sns.heatmap(corr, annot=True, cmap="summer")

plt.show()
from sklearn.linear_model import LogisticRegression
logit1 = LogisticRegression()
logit1.fit(train_dt1.drop('Survived', axis = 1), train_dt1['Survived'])
# do all the feature engineering on the test data
test_dt1.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_dt1['Sex'] = np.where(test_dt1['Sex'] == "male", 1, 0)
test_dt1['Embarked'] = test_dt1['Embarked'].fillna(test_dt1['Embarked'].mode())

embarked = pd.get_dummies(test_dt1['Embarked'], prefix = 'Embarked_')

test_dt1.drop("Embarked", axis=1, inplace=True)

test_dt1 = test_dt1.join(embarked)
pclass = pd.get_dummies(test_dt1['Pclass'], prefix="Pclass_")

test_dt1.drop("Pclass", axis=1, inplace=True)

test_dt1 = test_dt1.join(pclass)
test_dt1['Age'] = test_dt1['Age'].fillna(test_dt1['Age'].median())
test_dt1['SibSp'] = test_dt1['SibSp'].fillna(0)
test_dt1['Parch'] = test_dt1['Parch'].fillna(0)
test_dt1['Sex'] = pd.to_numeric(test_dt1['Sex'], downcast='unsigned')

test_dt1['Age'] = pd.to_numeric(test_dt1['Age'], downcast='float')

test_dt1['SibSp'] = pd.to_numeric(test_dt1['SibSp'], downcast='unsigned')

test_dt1['Parch'] = pd.to_numeric(test_dt1['Parch'], downcast='unsigned')
test_dt1.drop('Fare', axis=1, inplace=True)
orig_test_dt['Survived'] = logit1.predict(test_dt1)
submit = orig_test_dt[['PassengerId', 'Survived']]
submit.to_csv('submission.csv', index=False)