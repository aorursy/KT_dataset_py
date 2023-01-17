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
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
data = pd.concat([train, test], sort=False)
data['Sex'].replace(['male', 'female'], [2, 1], inplace=True)
data['Embarked'].fillna(('S'), inplace=True)
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

age_avg = data['Age'].mean()
age_std = data['Age'].std()
data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)

delete_columns = ['Name', 'PassengerId', 'Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)

train = data[:len(train)]
test = data[len(train):]

y_train = train['Survived']
X_train = train.drop('Survived', axis=1)
X_test = test.drop('Survived', axis=1)
import seaborn as sns
data['sex&Pclass'] = data['Sex']+data['Pclass']
train['sex&Pclass'] = data['sex&Pclass'][:len(train)]
test['sex&Pclass'] = data['sex&Pclass'][:len(test)]

data['Famsize'] = data['Parch']+data['SibSp']+1
train['Famsize'] = data['sex&Pclass'][:len(train)]
test['Famsize'] = data['sex&Pclass'][:len(test)]

data['Pclass&Age'] = data['Pclass']*data['Age']
train['Pclass&Age'] = data['Pclass&Age'][:len(train)]
test['Pclass&Age'] = data['Pclass&Age'][:len(test)]
import pandas_profiling
train.profile_report()
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100,max_depth=4, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
sub = pd.read_csv('../input/titanic/gender_submission.csv')
sub['Survived'] = list(map(int, y_pred))
sub.to_csv('submission.csv', index=False)