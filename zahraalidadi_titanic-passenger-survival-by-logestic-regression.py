# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/titanic/train.csv'):
    for filename in filenames:
        print(os.path.join(kaggle, tatanic))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
survive_df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train_df = pd.concat([train_df,survive_df])
test_df =  pd.concat([test_df,survive_df])
train_df.head(5)
train_df.shape
test_df.head(5)
train_df.info()
test_df.info()
train_df = train_df.drop(columns = ['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)
test_df = test_df.drop(columns = ['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)
test_df.head(5)
train_df.info()
train_df.shape
train_df
train_df.info()
test_df.info()
train_df = train_df.dropna(axis=0)

train_df.info()
#test_df =  test_df.dropna(axis=0)
test_df.info()
x_train = train_df[train_df.columns[1:9]] 
y_train = train_df[train_df.columns[0]] 
x_train
y_train
x_test
y_test
train_df.corr()
import matplotlib.pyplot as plt
import seaborn as sns 
f,ax = plt.subplots(figsize=(7, 7))
sns.heatmap(train_df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
from sklearn.preprocessing import OneHotEncoder
enc= pd.get_dummies(train_df[['Sex', 'Embarked']])
train_d = train_df.join(enc)

train_d
train_d = train_d.drop(columns = ['Sex', 'Embarked'], axis=1)
train_d
x_train = train_d[train_d.columns[1:12]] 
y_train = train_d[train_d.columns[1]] 
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
score = logreg.score(x_test, y_test)
print(score)

