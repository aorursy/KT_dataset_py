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
#testing

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')



data_train.sample(3)
s = data_train['Survived']

fig1, ax1 = plt.subplots()

ax1.pie([s[s==1].count(), s[s==0].count()], labels=['survived', 'dead'], autopct='%1.1f%%')

ax1.axis('equal')

plt.show()
data_train.Age.hist()
data_train.Age.isnull().any()
data_train.Age.size
f, ax = plt.subplots(figsize=(10, 8))

corr = data_train.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)
corr
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
clf = LogisticRegression(solver='lbfgs')
Y = data_train['Survived']
X = data_train['Pclass'].values.reshape(-1,1)
plt.figure(figsize=(50,10))

plt.scatter(data_train.index ,data_train['Pclass'])

plt.xlabel('Pclass')

plt.ylabel('Survived')

plt.title('Survived vs Pclass')
data_train['Survived'].shape
clf.fit(X, Y)
print(accuracy_score(clf.predict(X), Y))
Predictions = clf.predict(data_test['Pclass'].values.reshape(-1,1))
result  = pd.DataFrame({'PassengerId': data_test['PassengerId'], 'Survived': Predictions})
result.columns
result.to_csv("lr.csv", index=False)