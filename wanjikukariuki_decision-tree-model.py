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
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.head()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.head()
train_data.columns[train_data.isna().any()].to_list()
train_data = train_data.fillna(0)

train_data.tail()
test_data.columns[test_data.isna().any()].to_list()
test_data = test_data.fillna(0)

test_data.head()
features = ['Sex','SibSp', 'Parch', 'Pclass']

features
test_data['Sex'] = pd.get_dummies(test_data['Sex'])

train_data['Sex'] = pd.get_dummies(train_data['Sex'])
from sklearn.tree import DecisionTreeClassifier
X_train = train_data[features]

X_test = test_data[features]

Y_train = train_data['Survived']
titanic_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

titanic_tree.fit(X_train, Y_train)

predicted = titanic_tree.predict(X_test)

predicted
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predicted})

output.to_csv('Titanic_submissionsDT.csv', index = False)