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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.describe()
train.info()
train['Survived'].value_counts()
train_C = train.drop('PassengerId',axis=1).corr()

train_C
train.groupby('Sex')['Sex','Survived'].mean().plot.bar()
train.groupby('Pclass')['Survived'].mean().plot.bar()
train.groupby('SibSp',)['Survived'].mean()
train.groupby('Parch')['Survived'].mean()
train['Age'].value_counts()
df = pd.DataFrame(train['Age'])

df
df.fillna(train['Age'].mean())
train.groupby('Embarked')['Survived'].mean().plot.bar()
test['Survived'] = 0

train_test = train.append(test)
train_test = pd.get_dummies(train_test,columns=['Pclass'])
train_test = pd.get_dummies(train_test,columns=['Sex'])
train_test['sibsp_parch'] = train_test['SibSp']+train_test['Parch']

train_test=pd.get_dummies(train_test,columns=['sibsp_parch','SibSp','Parch'])
train_test = pd.get_dummies(train_test,columns=['Embarked'])
train_test = pd.get_dummies(train_test,columns=['Age'])
train_test.drop('Cabin',axis=1,inplace=True)

train_test.drop('Name',axis=1,inplace=True)

train_test.drop('Fare',axis=1,inplace=True)

train_test.drop('Ticket',axis=1,inplace=True)
train_test.info()
train_data = train_test[:891]

test_data = train_test[891:]

train_data_X = train_data.drop(['Survived'],axis=1)

train_data_Y = train_data['Survived']

test_data_X = test_data.drop(['Survived'],axis=1)
from sklearn.preprocessing import StandardScaler

ss2 = StandardScaler()

ss2.fit(train_data_X)

train_data_X_sd = ss2.transform(train_data_X)

test_data_X_sd = ss2.transform(test_data_X)
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=150,min_samples_leaf=2,max_depth=6,oob_score=True)



model.fit(train_data_X,train_data_Y)

model.score(train_data_X,train_data_Y)