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

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
train.head()
train.dtypes
sns.countplot(x='Survived', data=train)
train.describe()
train.isnull().sum()
sns.heatmap(train.isnull())
train['Cabin'].fillna('no cabin', inplace=True)

train['Age'].fillna('30', inplace=True)

train['Embarked'].fillna('no data', inplace=True)
train.isnull().sum()
df0=train.drop(columns=['Name','Ticket'])

df0.head(2)
test.isnull().sum()
test['Age'].fillna('30',inplace=True)

test.isnull().sum()
df1 = pd.get_dummies(df0['Sex'])

df2 = pd.get_dummies(df0['Embarked'])

df3 = pd.concat([df0,df1,df2], axis=1).drop(columns=['Sex','Embarked','no data','Cabin'])

df3.head()
# Number of trees:

n_estimators = [int(x) for x in np.linspace(start= 200, stop=2000, num=10)]

# Number of features in every split

max_features = ['auto','sqrt']

# Maximum number of level in a tree

max_depth = [int(x) for x in np.linspace(start=10, stop=120, num=12)]

# Minimum number of samples required to split a node

min_samples_split = [2,5,7]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1,3,5]

# Split method

bootstrap = [True, False]



grid = {'n_estimators':n_estimators,

       'max_features':max_features,

       'max_depth':max_depth,

       'min_samples_split':min_samples_split,

       'min_samples_leaf':min_samples_leaf,

       'bootstrap':bootstrap}
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split



trn = df3.drop(columns='Survived')

tst = df3['Survived']



x_train,x_test,y_train,y_test = train_test_split(trn,tst,test_size=0.20,random_state=23)



rf = RandomForestRegressor()



rf_rand = RandomizedSearchCV(estimator=rf, param_distributions=grid, n_iter=100, cv=3, verbose = 2,

                             n_jobs=-1)

rf_rand.fit(x_train,y_train)
rf_rand.best_params_
rf1 = RandomForestRegressor(n_estimators=30,

                            criterion='mse',

                            min_samples_split=2,

                            min_samples_leaf=1,

                            max_features='auto',

                            max_depth=10,

                            bootstrap=True)

rf1_fit = rf1.fit(x_train,y_train)

y_pred=rf1.predict(x_test).astype(int)
from sklearn import metrics



print('Mean absolute error: ', metrics.mean_absolute_error(y_test,y_pred))

print('Mean squared error: ', metrics.mean_squared_error(y_test,y_pred))

print('Root mean absolute error: ', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))



print('Accuracy score: ', metrics.accuracy_score(y_test,y_pred.round()))
df3.head(2)
df_test1 = pd.get_dummies(test['Sex'])

df_test2 = pd.get_dummies(test['Embarked'])

df_test = pd.concat([test,df_test1,df_test2], axis=1).drop(columns=['Sex','Embarked','Name','Ticket','Cabin'])

df_test.head()
df_test.isnull().sum()
df_test['Fare'].fillna('36',inplace=True)
y_pred1=rf1.predict(df_test).astype(int)
submission= pd.DataFrame({ 

    'PassengerId': test['PassengerId'],

    'Survived': y_pred1 })

submission.to_csv("Submission.csv", index=False)
dm = pd.read_csv('Submission.csv')

dm.head()