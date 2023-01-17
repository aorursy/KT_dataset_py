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
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train = train.drop(['Cabin', 'Name'], axis = 1)
train_new = train.dropna()
train_new = train_new.drop(['Ticket'], axis = 1)
train_last = pd.get_dummies(train_new, columns= train_new.select_dtypes(include='object').columns, drop_first=True)
train_last = train_last.drop(['PassengerId'], axis = 1)
test_new = test.drop(['Cabin'], axis = 1)
test_new['Age'].fillna(test_new['Age'].mean(), inplace=True)
test_new['Fare'].fillna(test_new['Fare'].mean(), inplace=True)
test_new = test_new.drop(['PassengerId', 'Name', 'Ticket'], axis = 1)
test_last = pd.get_dummies(test_new, columns= train_new.select_dtypes(include='object').columns, drop_first=True)
y_train = pd.DataFrame(train_last['Survived'])

train_last = train_last.drop(['Survived'], axis = 1)
from xgboost import XGBClassifier
classifier2 = XGBClassifier()

classifier2.fit(train_last, y_train)

predictions2 = classifier2.predict(test_last)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions2})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")