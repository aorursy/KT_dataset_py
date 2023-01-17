import numpy as np
import pandas as pd

titanic = pd.read_csv("../input/train.csv")
titanic_test = pd.read_csv("../input/test.csv")
titanic.describe()
titanic_test.describe()
titanic.head()
titanic.drop("PassengerId", axis = 1, inplace = True)
titanic.drop("Name", axis=1, inplace = True)
titanic.drop("Ticket", axis=1, inplace = True)
titanic.drop("Cabin", axis=1, inplace = True)
titanic.drop("Embarked", axis=1, inplace = True)

titanic_test.drop("PassengerId", axis = 1, inplace = True)
titanic_test.drop("Name", axis=1, inplace = True)
titanic_test.drop("Ticket", axis=1, inplace = True)
titanic_test.drop("Cabin", axis=1, inplace = True)
titanic_test.drop("Embarked", axis=1, inplace = True)
titanic.head()

column = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]
titanic = titanic.reindex(columns=column)
titanic.Age.fillna(titanic.Age.mean(), inplace = True)
titanic_test.Age.fillna(titanic_test.Age.mean(), inplace = True)
titanic_test.Fare.fillna(titanic_test.Fare.mean(), inplace = True)
titanic.describe()
titanic_test.describe()
titanic_test.head()
def f(s):
    if s == "male":
        return 0
    else:
        return 1
titanic["Sex"] =titanic.Sex.apply(f)       #apply rule/function f
titanic.head()
titanic_test["Sex"] =titanic_test.Sex.apply(f)  
titanic_test.head()
titanic.describe()
titanic_test.describe()
from sklearn import preprocessing
titanic_whole = pd.concat([titanic, titanic_test])
del titanic_whole['Survived']
titanic_whole.describe()

titanic_scaled = pd.DataFrame(preprocessing.scale(titanic_whole))
titanic_scaled.describe()
titanic_train_x = titanic_scaled.iloc[0:891,:]
titanic_test_x = titanic_scaled.iloc[891:1309,:]

titanic_train_y = titanic.iloc[:,6]

titanic_train_x = titanic_train_x.values
titanic_test_x = titanic_test_x.values
titanic_train_y = titanic_train_y.values
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(titanic_train_x, titanic_train_y)

clf.score(titanic_train_x, titanic_train_y)
output = clf.predict(titanic_test_x)

df = pd.DataFrame(output)
titanic_test_df = pd.read_csv("../input/test.csv")
titanic_test_df.head()
df["PassengerId"] = titanic_test_df["PassengerId"]
df.head()
df.columns = ["Survived", "PassengerId"]
df.head()
result = df.reindex(columns = ["PassengerId", "Survived"])
result.head()
result.to_csv("out.csv", header=True, index=False,  )