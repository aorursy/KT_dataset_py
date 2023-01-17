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
data1 = pd.read_csv("/kaggle/input/titanic/train.csv")

data1.head()

women = data1.loc[data1.Sex == 'female']["Survived"]

rate1 = sum(women)/len(women)

print(rate1)

men = data1.loc[data1.Sex == 'male']["Survived"]

rate2 = sum(men)/len(men)

print(rate2)
from sklearn.ensemble import RandomForestClassifier

y = data1["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(data1[features])

X_test = pd.get_dummies(data1[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': data1.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print(output)

print("Your submission was successfully saved!")