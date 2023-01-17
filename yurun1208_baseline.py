# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score





df_train = pd.read_csv('/kaggle/input/data-mining-kaggle-competition/train.csv')

X = df_train[['Pclass', 'Sex', 'SibSp', 'Parch']]

X['Sex'] = pd.get_dummies(X['Sex'])

Y = df_train['Survived']



X_train, X_test, Y_train, Y_test = train_test_split(X,Y,stratify=Y)



clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=100,

                     solver='sgd', verbose=1,  random_state=42)



clf.fit(X_train,Y_train)



print('Accuracy on training---')

y_pred_train=clf.predict(X_train)

print(accuracy_score(Y_train,y_pred_train))



print('Accuracy on test---')

y_pred_test=clf.predict(X_test)

print(accuracy_score(Y_test,y_pred_test))
df_test = pd.read_csv('/kaggle/input/data-mining-kaggle-competition/test.csv')

X_test = df_test[['Pclass', 'Sex', 'SibSp', 'Parch']]

X_test['Sex'] = pd.get_dummies(X_test['Sex'])

y_pred_test = clf.predict(X_test)

df_test['Survived'] = y_pred_test

df_test[['PassengerId', 'Survived']].to_csv('baseline.csv', index=False)