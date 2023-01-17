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
# making Training Data X, y and Test Data X_test

from sklearn.model_selection import train_test_split

data = pd.read_csv('/kaggle/input/titanic/train.csv')

X_test = pd.read_csv('/kaggle/input/titanic/test.csv')

X = data.drop('Survived', axis= 1)

y = data.Survived
X.info()
X_test.info()
X['Age'].fillna(X['Age'].mean(), inplace= True)

X_test['Age'].fillna(X_test['Age'].mean(), inplace= True)
X['Embarked'].fillna('S', inplace= True)
X_test['Fare'].fillna(X_test['Fare'].mean(), inplace= True)
X['Sex'].replace({'male':0, 'female':1}, inplace= True)

X_test['Sex'].replace({'male':0, 'female':1}, inplace= True)

X['Embarked'].replace({'S':0, 'C':1, 'Q':2}, inplace= True)

X_test['Embarked'].replace({'S':0, 'C':1, 'Q':2}, inplace= True)
col_not_used = ['PassengerId', 'Name', 'Ticket', 'Cabin']

X_clean = X.drop(col_not_used, axis= 1)

X_test_clean = X_test.drop(col_not_used, axis= 1)
X_clean.head()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



# Splitting the clean data  into Training Data and Validation Data

X_train, X_valid, y_train, y_valid = train_test_split(X_clean, y, random_state= 0)





# Making scores dictionary with different parameters

model = RandomForestClassifier(n_estimators= 6)

model.fit(X_train, y_train)



y_train_pred = classification_report(y_train, model.predict(X_train))

y_valid_pred = classification_report(y_valid, model.predict(X_valid))

print(y_train_pred, y_valid_pred)

pred = model.predict(X_test_clean)

output = pd.DataFrame({'PassengerId': X_test.PassengerId, 'Survived':pred})

output.to_csv('my_submission.csv', index= False)

print('Submission was saved')
