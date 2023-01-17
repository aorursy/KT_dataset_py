# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df[:3].T
df.isnull().sum()
df.groupby('Sex')['Survived'].mean()
def kill(d):
    if d['Sex'] == 'male':
        return 1
    return 0
criteria = list(df.columns)
criteria
useless = ['Name', # 귀찮아
           'PassengerId', # 필요없어
           'Survived', # ??
           'Ticket', # 귀찮아
           'Sex', # 아까 죽였어
           'Cabin' # null이 너무 많아
          ]
for criterion in useless:
    criteria.remove(criterion)
for criterion in criteria:
    print(criterion)
    print(df.groupby(criterion)['Survived'].mean())
df['Age'].hist(bins=20)
df.groupby(df['Age'].isnull())['Survived'].mean()
# 음... Age가 Nan이라고 죽이는 게 맞을까요? 일단 해봅시다.
def kill2(d):
    if kill(d) or np.isnan(d['Age']):
        return 1
    return 0
df[df['Age'].isnull()][:3].apply(kill2, axis=1)
df[df['Age'].isnull()==False][:3]
df[df['Age'].isnull()==False][:3].apply(kill2, axis=1)
df['Age_bin'] = df['Age'].apply(lambda x:x//10)
df[:3]
df['Fare'].hist(bins=20, range=(0,200))
df['Fare_bin'] = df['Fare'].apply(lambda x:x//10)
criteria.remove('Fare')
criteria.remove('Age')
criteria += ['Fare_bin','Age_bin']
for criterion in criteria:
    print(criterion)
    print(df.groupby(criterion)['Survived'].mean())
def kill3(d):
    if kill2(d) or d['SibSp'] > 4 or d['Parch'] > 3 or d['Fare_bin'] < 1:
        return 1
    return 0
1 - df.apply(kill3, axis=1).mean()
df['Survived'].mean()
df['Predicted'] = df.apply(lambda x: 1-kill3(x), axis=1)
np.mean(df['Predicted'] == df['Survived'])
def kill4(d):
    if kill(d) or d['SibSp'] > 4 or d['Parch'] > 3 or d['Fare_bin'] < 1:
        return 1
    return 0
df['Predicted'] = df.apply(lambda x: 1-kill4(x), axis=1)
np.mean(df['Predicted'] == df['Survived'])
df_test = pd.read_csv('../input/test.csv')
df_test['Predicted'] = df_test.apply(kill4, axis=1)
def kill5(d):
    if kill(d) or d['SibSp'] > 4 or d['Parch'] > 3 or d['Fare'] < 5:
        return 1
    return 0
df_test['Survived'] = df_test.apply(kill5, axis=1)
df_test[:3]
df_test[['PassengerId','Survived']].to_csv('znmean.csv', index=False)
print(df_test[['PassengerId','Survived']].to_csv(index=False))
