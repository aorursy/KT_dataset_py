# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.float_format = '{:,.2f}'.format

import matplotlib.pyplot as plt
import seaborn as sns

# Figures inline and set visualization style
%matplotlib inline
sns.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
df_train.head()
df_test = pd.read_csv('../input/test.csv')
df_test.head()
col_num = df_train.select_dtypes(include=['int64','float64']).columns
col_text = df_train.select_dtypes(include=['object']).columns

print('size of training data:',df_train.shape)
print('Descriptive statistics of numeric columns:')
df_train[col_num].describe()
print('Descriptive statistics of text columns:')
df_train[col_text].describe()
print('Missing value: ')
df_train.isna().sum()
df_train.Age.hist()
df_train.Age.fillna(df_train.Age.median(),inplace=True)
print('The number of N/A values: ',df_train.Age.isna().sum())
sns.countplot(x='Embarked',data=df_train)
df_train.Embarked.fillna(df_train.Embarked.mode()[0],inplace=True)
print('The number of N/A values: ',df_train.Embarked.isna().sum())
for s in [0,1]:
    # Subset to the airline
    subset = df_train[df_train['Survived'] == s]
    plt.title('Ditribution of fare separated by survival rate')
    # Draw the density plot
    sns.distplot(subset['Fare'], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = s)
for s in [0,1]:
    # Subset to the airline
    subset = df_train[df_train['Survived'] == s]
    plt.title('Ditribution of age separated by survival rate')
    # Draw the density plot
    sns.distplot(subset['Age'], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = s)
sns.factorplot(x='Survived', col='Pclass', kind='count', data=df_train)

col_sel = ['Age','Fare','SibSp']

#X_train = df_train.drop("Survived", axis=1)
X_train = df_train[col_sel]
Y_train = df_train["Survived"]
X_test = df_test[col_sel]
#X_test  = df_train.drop("PassengerId", axis=1).copy()

X_test.Age.fillna(X_test.Age.median(),inplace=True)
X_test.Fare.fillna(X_test.Fare.median(),inplace=True)
X_test.describe()
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(round(acc_log,2,), "%")
submission = pd.DataFrame({
    'PassengerID':df_test.PassengerId,
    'Survived':Y_pred
})

submission.to_csv('submission.csv', index=False)