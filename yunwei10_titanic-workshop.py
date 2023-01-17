# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
test.head()
train.info()
train.describe()
cols = list(train)



for i in cols:

    print(i, ' '*(20-len(i)), train[i].isnull().sum())
train['Ticket'].head()
print(train.shape)

col_to_drop = ['Ticket', 'Cabin', 'Name', 'PassengerId']

train = train.drop(col_to_drop, axis=1)

train.shape
train['Embarked'] = train['Embarked'].fillna('C')

train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'female' else 0)

train['Embarked'] = train['Embarked'].map( {'S': 0, 'Q': 1, 'C': 2}).astype(int)
def fill_ages(df):

    null_count = df['Age'].isnull().sum()

    avg = df['Age'].mean()

    std = df['Age'].std()

    random_ages = np.random.randint(avg - std, avg + std, null_count)

    df['Age'][np.isnan(df['Age'])] = random_ages

    df['Age'] = df['Age'].astype(int)

    return df



def group_ages(df):

    df.loc[df['Age'] <= 18, 'Age'] = 0

    df.loc[(df['Age'] > 18) & (df['Age'] <= 36), 'Age'] = 1

    df.loc[(df['Age'] > 36) & (df['Age'] <= 54), 'Age'] = 2

    df.loc[(df['Age'] > 54) & (df['Age'] <= 72), 'Age'] = 3

    df.loc[df['Age'] > 72, 'Age'] = 4

    return df



def group_fares(df):

    df.loc[df['Fare'] <= 7.91, 'Fare'] = 0

    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1

    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2

    df.loc[df['Fare'] > 31, 'Fare'] = 3

    return df



train = group_ages(fill_ages(train))

# train = group_fare(group_age(fill_ages(train)))
train.head()
plt.figure(figsize=(14, 12))

plt.title('Correlation of features')

colormap = plt.cm.RdBu_r

sns.heatmap(train.astype(float).corr(), annot=True, cmap=colormap)
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False).head(10)
train.head()
y = train['Survived']

x = train.drop('Survived', axis=1)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

print(x_train.head())

y_train.head()
lr = LogisticRegression(solver='lbfgs')

lr.fit(x_train, y_train)



score = round(lr.score(x_test, y_test) * 100, 2)

print(score)



y_pred = lr.predict(x_test)

y_pred
svc = SVC(gamma='auto')

svc.fit(x_train, y_train)



score = round(svc.score(x_train, y_train) * 100, 2)

print(score)



y_pred = svc.predict(x_test)

y_pred