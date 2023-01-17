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
import numpy as np

import pandas as pd 

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error

from sklearn.neighbors import KNeighborsClassifier
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
real_data = pd.read_csv("/kaggle/input/titanic/test.csv")

# real_data.head()
print("Unique values in Parch ", train_data['Parch'].unique())

train_data['Parch'].value_counts();
print(train_data.columns)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Survived']

features_test = ['PassengerId','Pclass', 'Sex', 'Age', 'SibSp']
train_data.isnull().sum() # only age with 177

real_data.isnull().sum(); # only age with 86
# print(train_data.shape) # 891

# print(real_data.shape) # 418

train_data.dropna(subset=['Age'], inplace=True) #891-177 = 714

real_data.dropna(subset=['Age'], inplace=True); #491-86 = 332

print(train_data.shape) # 714

print(real_data.shape) # 332
train_data = train_data[features]

real_data = real_data[features_test]
X = pd.get_dummies(train_data.drop(['Survived'], axis='columns'))

y = train_data["Survived"]

X_real = pd.get_dummies(real_data.drop(['PassengerId'], axis='columns'))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
# KNN Classifier with train_test_split()

knn = KNeighborsClassifier(n_neighbors=3, p=2, metric='euclidean')

knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print(classification_report(y_test, pred))
# model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)

# model.fit(X, y)

predictions = knn.predict(X_real)
output = pd.DataFrame({'PassengerId': real_data.PassengerId, 'Survived': predictions})

output
output = pd.DataFrame({'PassengerId': real_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")