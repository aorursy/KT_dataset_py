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
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn import metrics
%matplotlib inline
train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
print('train shape:',train.shape)
print('test.shape:',test.shape)
train['Sex']=train['Sex'].map({'male':0,'female':1})
test['Sex']=test['Sex'].map({'male':0,'female':1})

train.head()
test.head()
train.describe(include='all')
test.describe(include='all')
sns.countplot(x = 'Survived', hue = 'Sex', data = train)

train.isnull().sum()
sns.countplot(x='Survived',hue='Pclass',data=train)

train.head()
test.head()

sns.countplot(x='Survived',hue='SibSp',data=train)


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)
print (xtrain[0:10, :])
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=100000)
lr.fit(xtrain,ytrain)
ypred=lr.predict(xtest)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)
cm
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred))
train.isnull().sum()
train1=train.dropna()
train1.isnull().sum()

test1=test.dropna()
test1.isnull().sum()
x=train1.iloc[:,[2,4,6,9]].values
x
x1=test1.iloc[:,[1,3,5,8]].values
y=train1.iloc[:,[1]].values
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.20, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)
print (xtrain[0:10, :])
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=100000)
lr.fit(xtrain,ytrain)
ypred=lr.predict(xtest)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred))
