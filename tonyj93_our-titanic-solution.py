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
submission_df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')



submission_df
x = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



x
x.isna().sum()
x.isna().sum() / x.isna().count() * 100
x['Pclass'].unique()
x[['Sex','PassengerId','Survived']].groupby(['Sex']).count()
x[['Sex','PassengerId','Survived']].groupby(['Sex']).sum()
x[['Sex','PassengerId','Survived']].groupby(['Sex']).mean()
x[['Pclass','Survived']].groupby('Pclass').count()
x[['Pclass','Survived']].groupby('Pclass').sum()
x[['Pclass','Survived']].groupby('Pclass').sum() / x[['Pclass','Survived']].groupby('Pclass').count()
col_to_extract = ['Pclass','Sex']

target = 'Survived'





x_train = x[col_to_extract]

x_test = test[col_to_extract]

y_train = x[target]

# y_test = test[target]
x_train.loc[:, 'Gender'] = 0

x_train.loc[x_train['Sex'] == 'female', 'Gender'] = 1

x_train = x_train.drop('Sex',axis=1)

x_train
x_test.loc[:, 'Gender'] = 0

x_test.loc[x_test['Sex'] == 'female', 'Gender'] = 1

x_test = x_test.drop('Sex',axis=1)

x_test
from sklearn.linear_model import LogisticRegression
estimator = LogisticRegression()
estimator.fit(x_train,y_train)
y_pred = estimator.predict(x_test)
test
y_pred
test['Survived'] = y_pred
test
test[['PassengerId','Survived']].set_index('PassengerId').to_csv('submissions2.csv')