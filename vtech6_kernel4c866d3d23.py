# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

import keras

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv(r'/kaggle/input/titanic/train.csv')

test = pd.read_csv(r'/kaggle/input/titanic/test.csv')
train[:10]
train['Embarked'].unique()
train['Embarked']=train['Embarked'].fillna('NaN')

test['Embarked']=test['Embarked'].fillna('NaN')

sum(train['Embarked'].isna()==True)
LE = LabelEncoder()



train['Embarked'] = LE.fit_transform(train['Embarked'])

test['Embarked'] = LE.transform(test['Embarked'])
train['Sex'] = LE.fit_transform(train['Sex'])

test['Sex'] = LE.transform(test['Sex'])
irrelevant = ['Name', 'Cabin', 'Ticket', 'Fare', 'PassengerId']

X_train = train.drop(irrelevant, axis=1)

X_train = X_train.drop('Survived', axis=1)

X_test = test.drop(irrelevant, axis=1)

y_train = train['Survived']
X_train = X_train.ffill()

X_test = X_test.ffill()
corr_matrix=X_train.corr()

plt.figure(figsize=(12, 10))

sns.heatmap(corr_matrix, lw=.5, cmap='coolwarm', annot=True)
SS = StandardScaler()

X_numpy = np.array(X_train)

X_numpy = SS.fit_transform(X_numpy)

test_numpy = np.array(X_test)

test_numpy = SS.transform(test_numpy)
y_numpy = np.array(y_train)
y_numpy
nn = keras.models.Sequential()

nn.add(keras.layers.Dense(512, activation='relu'))

nn.add(keras.layers.Dense(256, activation='relu'))

nn.add(keras.layers.Dense(128, activation='relu'))

nn.add(keras.layers.Dense(32, activation='relu'))

nn.add(keras.layers.Dense(1, activation='sigmoid'))

nn.compile(optimizer='adam',

          loss='binary_crossentropy',

          metrics=['accuracy'])

nn.fit(X_numpy, y_numpy, epochs=20)
from sklearn.metrics import accuracy_score

preds = nn.predict_classes(test_numpy)
submission = pd.DataFrame(data=preds, columns=['Survived'])

submission['PassengerId'] = test['PassengerId']
submission.to_csv('submission.csv', index=False)