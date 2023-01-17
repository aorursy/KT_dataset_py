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
data_train = pd.read_csv('/kaggle/input/titanic/train.csv')

data_train.head()
data_test = pd.read_csv('/kaggle/input/titanic/test.csv')

data_test.head()
y = data_train['Survived']

y
X_no_cols_train = data_train.loc[:, data_train.columns != "Survived"].drop(columns=['Name', 'Ticket', 'PassengerId'])

X_no_cols_train
X_no_cols_test = data_test.drop(columns=['Name', 'Ticket', 'PassengerId'])

X_no_cols_test
X_no_cols_train.info()
X_no_cols_test.info()
import seaborn as sns

# Visualize nulls

sns.heatmap(X_no_cols_train.isnull(), cbar=False)
# Visualize nulls

sns.heatmap(X_no_cols_test.isnull(), cbar=False)
# Fill Age with mean

X_no_cols_train['Age'] = X_no_cols_train['Age'].fillna(X_no_cols_train['Age'].mean())

sns.heatmap(X_no_cols_train.isnull(), cbar=False)
X_no_cols_test['Age'] = X_no_cols_test['Age'].fillna(X_no_cols_test['Age'].mean())

sns.heatmap(X_no_cols_test.isnull(), cbar=False)
X_no_cols_test['Fare'] = X_no_cols_test['Fare'].fillna(X_no_cols_test['Fare'].mean())

sns.heatmap(X_no_cols_test.isnull(), cbar=False)
# Fill null cabins with a constant

X_no_cols_train['Cabin'] = X_no_cols_train['Cabin'].fillna('NOCABIN')

sns.heatmap(X_no_cols_train.isnull(), cbar=False)
X_no_cols_train['Embarked'] = X_no_cols_train['Embarked'].fillna('UNKNOWN')

sns.heatmap(X_no_cols_train.isnull(), cbar=False)
# Fill null cabins with a constant

X_no_cols_test['Cabin'] = X_no_cols_test['Cabin'].fillna('NOCABIN')

sns.heatmap(X_no_cols_test.isnull(), cbar=False)
X_no_nulls_train = X_no_cols_train

X_no_nulls_train.head()
X_no_nulls_test = X_no_cols_test

X_no_nulls_test.head()
X_no_nulls_train.info()
from sklearn.preprocessing import OrdinalEncoder



for col in ['Sex', 'Cabin', 'Embarked']:

    encoder = OrdinalEncoder()

    encoder.fit(X_no_nulls_train[[col]])

    X_no_nulls_train[[col]] = encoder.transform(X_no_nulls_train[[col]])



X_train = X_no_nulls_train  

X_train.head()
for col in ['Sex', 'Cabin', 'Embarked']:

    encoder = OrdinalEncoder()

    encoder.fit(X_no_nulls_test[[col]])

    X_no_nulls_test[[col]] = encoder.transform(X_no_nulls_test[[col]])



X_test = X_no_nulls_test 

X_test.head()
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression



logreg_model = LogisticRegression(max_iter=500)

logreg_scores = cross_val_score(logreg_model, X_train, y, cv=5)

print("Attempt 1: ", logreg_scores)

print("Mean: ", logreg_scores.mean())
logreg_model.fit(X_train, y)
predictions = logreg_model.predict(X_test)
submission = pd.DataFrame({'PassengerId': data_test['PassengerId'], 'Survived': predictions})

submission.head()
submission.to_csv('gender_submission.csv', index=False)