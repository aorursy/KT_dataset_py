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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

print(train_data.head())
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

print(test_data.head())
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)
from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer

y = train_data["Survived"]



features = ["Age", "Pclass", "Sex", "SibSp", "Parch", "Fare"]



my_imputer = SimpleImputer()



X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

X = my_imputer.fit_transform(X)

X_test = my_imputer.fit_transform(X_test)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_imputed.csv', index=False)

print("Your submission was successfully saved!", model.score(X,y))