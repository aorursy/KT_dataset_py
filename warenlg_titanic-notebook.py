# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/train.csv')
df.shape
df_train = df.iloc[:712, :]

df_test = df.iloc[712:, :]
df_train.tail(10)
df_train = df_train.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df_train.info()
df_train = df_train.dropna()
df_train['Sex'].unique()
df_train['Sex'] = df_train['Sex'].map({'female':0, 'male':1})
df_train['Embarked'].unique()
df_train['Embarked'] = df_train['Embarked'].map({'C':1, 'S':2, 'Q':3})
df_train.head(10)
X_train = df_train.iloc[:, 2:].values

y_train = df_train['Survived']
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=0)
model = model.fit(X_train, y_train)
df_test.head(10)
df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)



df_test = df_test.dropna()



df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male':1})

df_test['Embarked'] = df_test['Embarked'].map({'C':1, 'S':2, 'Q':3})



X_test = df_test.iloc[:, 2:]

y_test = df_test['Survived']
y_prediction = model.predict(X_test)
np.sum(y_prediction == y_test)
df_test.info()
float(len(y_test))
np.sum(y_prediction == y_test) / float(len(y_test))
np.sum(y_test) / float(len(y_test))
import pandas as pd

import numpy as np



df = pd.read_csv('../input/train.csv')

df_train = df.iloc[:712, :]

df_test = df.iloc[712:, :]
df_train = df_train.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df_train.info()
age_mean = df_train['Age'].mean()

df_train['Age'] = df_train['Age'].fillna(age_mean)
from collections import Counter

Counter(df_train['Embarked'])
df_train['Embarked'] = df_train['Embarked'].fillna('S')
df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})
df_train.info()
pd.get_dummies(df_train['Embarked'], prefix='Embarked').head(10)
df_train = pd.concat([df_train, pd.get_dummies(df_train['Embarked'], prefix='Embarked')], axis=1)
df_train = df_train.drop(['Embarked'], axis=1)