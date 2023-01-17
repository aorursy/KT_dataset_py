import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../input/titanic/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("../input/titanic/test.csv")

test_data.head()
all_women = len(train_data.loc[train_data.Sex == "female"])

survived_women = len(train_data.loc[train_data.Sex == "female"][train_data.Survived == 1])



print("all women: ", all_women)

print("Survived women: ", survived_women)



print("Women survival rate: ", survived_women/all_women)



all_men = len(train_data.loc[train_data.Sex == "male"])

survived_men = len(train_data.loc[train_data.Sex == "male"][train_data.Survived == 1])



print("all men: ", all_men)

print("Survived men: ", survived_men)



print("men survival rate: ", survived_men/all_men)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]

x = pd.get_dummies(train_data[features])

x_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(x, y)

predictions = model.predict(x_test)

output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": predictions})

output.to_csv("../output/random_forest_submission_final.csv", index=False)