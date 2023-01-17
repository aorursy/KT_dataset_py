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
titanic_df = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
titanic_df.head(10)
titanic_df.dtypes
titanic_df = titanic_df.drop(columns=['Name','Fare','Ticket'])
test = test.drop(columns=['Name','Fare','Ticket'])
titanic_df['Sex'] = pd.get_dummies(titanic_df['Sex'])
test['Sex'] = pd.get_dummies(test['Sex'])
titanic_df['Age'].isnull().sum()
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())
titanic_df['Embarked'].value_counts()
titanic_df['Embarked'].isnull().sum()
titanic_df['Embarked'] = titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0])
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])
titanic_df['Embarked'] = pd.Categorical(titanic_df['Embarked']).codes
test['Embarked'] = pd.Categorical(test['Embarked']).codes
titanic_df['Cabin'].isnull().sum()
titanic_df.shape
titanic_df['Cabin'] = titanic_df['Cabin'].fillna(titanic_df['Cabin'].mode()[0])
test['Cabin'] = test['Cabin'].fillna(test['Cabin'].mode()[0])
titanic_df['Cabin'] = pd.Categorical(titanic_df['Cabin']).codes
test['Cabin'] = pd.Categorical(test['Cabin']).codes
titanic_df.head()
titanic_df_attr = titanic_df.iloc[:,1:8]
test_attr = test.iloc[:,1:8]
titanic_df_attr.head()
test_attr.head()
X_train = np.array(titanic_df_attr.iloc[:,1:6])
y_train = np.array(titanic_df_attr['Survived'])
X_test = np.array(test.iloc[:,1:6])
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 1)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)
classifier.score(X_train , y_train)
from sklearn.linear_model import LogisticRegression
clf_Logistic = LogisticRegression()
clf_Logistic.fit(X_train , y_train)
pred_logistic = clf_Logistic.predict(X_test)
clf_Logistic.score(X_train , y_train)
from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier(n_estimators = 100)
clf_RF.fit(X_train , y_train)
pred_RF = clf_RF.predict(X_test)
clf_RF.score(X_train , y_train)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred_RF})

output.to_csv('My_Submission.csv', index=False)
print("Submission Done!!")