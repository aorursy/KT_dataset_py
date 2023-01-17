# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df.head()
df.describe()
df.info()
df.nunique()
df['Ticket'].values
df[df['Cabin'].isnull() == False]['Cabin'].values
titanic = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')

titanic.head()
titanic.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)
titanic.columns.values
sns.countplot(titanic['Survived'])
sns.countplot(titanic['SibSp'])
sns.countplot(titanic['Parch'])
sns.distplot(titanic['Fare'], bins=100, kde=False)
titanic['Fare'].value_counts().sort_index().head(60)
titanic['Fare'].value_counts().sort_index(ascending=False).head(10)
titanic['Pclass'].value_counts()
titanic['pclass_1'] = titanic['Pclass'].apply(lambda cl: 1 if cl == 1 else 0)

titanic['pclass_2'] = titanic['Pclass'].apply(lambda cl: 1 if cl == 2 else 0)

titanic.drop('Pclass', axis=1, inplace=True)
titanic['Male'] = titanic['Sex'].map({'male': 1, 'female':0})

titanic.drop('Sex', axis=1, inplace=True)
titanic.head()
titanic['Embarked'].value_counts(dropna=False)
titanic['Embarked_S'] = titanic['Embarked'].map({np.NaN: 1, 'S': 1, 'C': 0, 'Q': 0})

titanic['Embarked_S'].sum()
titanic['Embarked_Q'] = titanic['Embarked'].map({np.NaN: 0, 'S': 0, 'C': 0, 'Q': 1})

titanic['Embarked_Q'].sum()

titanic.drop('Embarked', axis=1, inplace=True)
titanic.head()
sns.distplot(titanic[titanic['Age'].isnull() == False]['Age'])
mean_age = titanic['Age'].mean()

mean_age
test = pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')

test['Age'].value_counts(dropna=False)
titanic['Age'].replace({np.NaN: mean_age}, inplace=True)

titanic['Age'].value_counts()
y_train = titanic['Survived']

X_train_none_standarized = titanic.drop('Survived', axis=1)

X_train_none_standarized.head()
sns.heatmap(pd.DataFrame(X_train_none_standarized).corr(), cmap='coolwarm')
X_train_none_standarized.columns
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_all_standarized = scaler.fit_transform(X_train_none_standarized)

pd.DataFrame(X_train_all_standarized).describe()

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(X_train_all_standarized, y_train)

log_reg.score(X_train_all_standarized, y_train)

log_reg.coef_[0]
X_train_none_standarized.columns
coefs = pd.DataFrame(log_reg.coef_[0], index=X_train_none_standarized.columns).sort_values(by=0)

coefs
test.columns.values
test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.info()
test['Age'].replace({np.NaN: mean_age}, inplace=True)
test[test['Fare'].isnull()]
mean_class_3_fare = titanic[(titanic['pclass_1'] == 0) & (titanic['pclass_2'] == 0)]['Fare'].mean()

mean_class_3_fare
test['Fare'].replace({np.NaN: mean_class_3_fare}, inplace=True)
test.info()
test.head()
test['Pclass'].value_counts()
test['pclass_1'] = test['Pclass'].apply(lambda cl: 1 if cl == 1 else 0)

test['pclass_2'] = test['Pclass'].apply(lambda cl: 1 if cl == 2 else 0)

test.drop('Pclass', axis=1, inplace=True)
test['pclass_1'].sum()
test['pclass_2'].sum()
test['Sex'].value_counts()
test['Male'] = test['Sex'].map({'male': 1, 'female':0})

test.drop('Sex', axis=1, inplace=True)
test['Male'].sum()
test['Embarked'].value_counts()
test['Embarked_S'] = test['Embarked'].map({np.NaN: 1, 'S': 1, 'C': 0, 'Q': 0})

test['Embarked_S'].sum()
test['Embarked_Q'] = test['Embarked'].map({np.NaN: 0, 'S': 0, 'C': 0, 'Q': 1})

test['Embarked_Q'].sum()
test.drop('Embarked', axis=1, inplace=True)
X_test_all_standarized = scaler.transform(test)
preds = log_reg.predict(X_test_all_standarized)

preds
pred_df = test.copy()

pred_df['Survived'] = preds
pred_df.columns
pred_df.drop(['Age', 'SibSp', 'Parch', 'Fare', 'pclass_1', 'pclass_2', 'Male','Embarked_S', 'Embarked_Q'], axis=1, inplace=True)

pred_df.head()
pred_df.to_csv('/kaggle/working/preds.csv')