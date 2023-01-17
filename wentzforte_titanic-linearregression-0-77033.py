# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from sklearn.preprocessing import LabelEncoder



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import numpy as np

import pandas as pd

import statsmodels.api as sm

import scipy.stats as st

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn.metrics import confusion_matrix

import matplotlib.mlab as mlab

%matplotlib inline



from sklearn import datasets, linear_model

from  sklearn.model_selection  import  train_test_split

from sklearn.metrics import mean_squared_error, r2_score

import warnings

warnings.filterwarnings('ignore')        
train = pd.read_csv(os.path.join(dirname, 'train.csv'))

test = pd.read_csv(os.path.join(dirname, 'test.csv'))

submission = pd.read_csv(os.path.join(dirname, 'gender_submission.csv'))
train.head()
test.head()
submission.head()
train.head()
train['Embarked'].unique()
le = LabelEncoder()
train['Cabin'].value_counts()
train['Embarked'] = train['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})

train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})

train.fillna(0, inplace=True)



test['Embarked'] = test['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})

test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

test.fillna(0, inplace=True)
train.head()
train = train.drop(columns=['Name', 'Ticket', 'Cabin'])

test = test.drop(columns=['Name', 'Ticket', 'Cabin'])
train.describe().T
fig = plt.subplots(figsize = (10,10))

sb.set(font_scale=1.5)

sb.heatmap(train.corr(),square = True,cbar=True,annot=True,annot_kws={'size': 10})

plt.show()
sb.countplot(x='Survived',data=train)
X_train, X_test, y_train, y_test = train_test_split(train.drop(columns=['Survived']), pd.DataFrame(train.Survived))
y_test.describe().T
X_train.describe().T
train.head()
lm = linear_model.LinearRegression()

lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
print('Coefficients: \n', lm.coef_)

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

print('Variance score: %.2f' % r2_score(y_test, y_pred))

print('Total True: %d' % lm.predict(X_test).round().sum())
y_test['Survived'] = lm.predict(X_test).round()

y_test.classe = y_test['Survived'].astype(int)
sb.countplot(x='Survived',data= y_test)
submission.head()
submission['Survived'] = lm.predict(test).round().astype(int)
submission.to_csv('Submission.csv', index=False)