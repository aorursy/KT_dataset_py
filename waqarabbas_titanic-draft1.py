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
data_train = pd.read_csv('/kaggle/input/titanic/train.csv')

data_test = pd.read_csv('/kaggle/input/titanic/test.csv')



print(data_train.shape)

data_train.head()
print(data_test.shape)

data_test.head()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



def fateOf(feature):

    survived = data_train[data_train['Survived']==1][feature].value_counts()

    dead = data_train[data_train['Survived']==0][feature].value_counts()

    df = pd.DataFrame ([survived, dead])

    df.index = ['Survived','Dead']

    df.plot(kind = 'bar', stacked= True, figsize=(10,5))

    
data_train["Age"].fillna(data_train.Age.mean(), inplace = True)

data_test["Age"].fillna(data_train.Age.mean(), inplace = True)

data_test["Fare"].fillna(data_train.Fare.mean(), inplace = True)
a = data_train.pop('Survived')

a.head()
x = list(data_train.dtypes[data_train.dtypes != object].index)

y = list(data_test.dtypes[data_test.dtypes != object].index)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
model = KNeighborsClassifier(n_neighbors = 5)

model.fit(data_train[x],a)
print (accuracy_score(a, model.predict(data_train[x])))
target = data_test[y]

prediction = model.predict(target)
submission = pd.DataFrame({

        "PassengerId": data_test["PassengerId"],

        "Survived": prediction

    })



submission.to_csv('submission.csv', index=False)