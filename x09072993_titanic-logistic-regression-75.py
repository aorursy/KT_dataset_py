import numpy as np

import pandas as pd

from sklearn.linear_model import LogisticRegression
titanic_train = pd.read_csv('../input/train.csv')

titanic_test = pd.read_csv('../input/test.csv')
titanic_train['Age'] = titanic_train['Age'].fillna(titanic_train['Age'].median())

titanic_train.loc[titanic_train['Sex'] == 'male', 'Sex'] = 0

titanic_train.loc[titanic_train['Sex'] == 'female', 'Sex'] = 1

titanic_train['Embarked'] = titanic_train['Embarked'].fillna('S')

titanic_train.loc[titanic_train['Embarked'] == 'S', 'Embarked'] = 0

titanic_train.loc[titanic_train['Embarked'] == 'C', 'Embarked'] = 1

titanic_train.loc[titanic_train['Embarked'] == 'Q', 'Embarked'] = 2

titanic_train['Fare'] = titanic_train['Fare'].fillna(titanic_train['Fare'].median())



titanic_test['Age'] = titanic_test['Age'].fillna(titanic_train['Age'].median())

titanic_test.loc[titanic_test['Sex'] == 'male', 'Sex'] = 0

titanic_test.loc[titanic_test['Sex'] == 'female', 'Sex'] = 1

titanic_test['Embarked'] = titanic_test['Embarked'].fillna('S')

titanic_test.loc[titanic_test['Embarked'] == 'S', 'Embarked'] = 0

titanic_test.loc[titanic_test['Embarked'] == 'C', 'Embarked'] = 1

titanic_test.loc[titanic_test['Embarked'] == 'Q', 'Embarked'] = 2

titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].median())
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
alg = LogisticRegression(random_state=1)

alg.fit(titanic_train[predictors], titanic_train["Survived"])

predictions = alg.predict(titanic_test[predictors])
submission = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": predictions

    })

submission.to_csv("kaggle.csv", index=False)