# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
titanic = pd.read_csv("../input/train.csv")

titanic.head()
titanic.columns
y = titanic['Survived']

X = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state =1,test_size = 0.2)

X_train.head()
one_hot_X_train = pd.get_dummies(X_train)
one_hot_X_train.head()
one_hot_X_test = pd.get_dummies(X_test)

final_train, final_test = one_hot_X_train.align(one_hot_X_test, join='left', axis=1)

final_test.head()
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
final_train = my_imputer.fit_transform(final_train)
final_test = my_imputer.fit_transform(final_test)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

sc.fit(final_train)



final_train_std = sc.transform(final_train)

final_test_std = sc.transform(final_test)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=1)

lr.fit(final_train_std, y_train)
predictions = lr.predict(final_test_std)
predictions
from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions)
X_combined = np.vstack((final_train_std, final_test_std))

y_combined = np.hstack((y_train, y_test))

lr2 = LogisticRegression(random_state=1)

lr2.fit(X_combined, y_combined)

predictions2 = lr2.predict(X_combined)

accuracy_score(predictions2, y_combined)
submission_test = pd.read_csv('../input/test.csv')

submission_test.head()
sub_test = submission_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

sub_test = pd.get_dummies(sub_test)
sub_test = my_imputer.fit_transform(sub_test)
sub_test_std = sc.transform(sub_test)
sub_predictions = lr2.predict(sub_test_std)
sub_predictions
output = pd.DataFrame({

    'PassengerId': submission_test.PassengerId,

    'Survived': sub_predictions

})
output.to_csv('submssion.csv', index = False)