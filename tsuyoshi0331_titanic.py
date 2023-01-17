import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression



train = pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv("../input/titanic/test.csv")



def kesson_table(df): 

        null_val = df.isnull().sum()

        percent = 100 * df.isnull().sum()/len(df)

        kesson_table = pd.concat([null_val, percent], axis=1)

        kesson_table_ren_columns = kesson_table.rename(

        columns = {0 : '欠損数', 1 : '%'})

        return kesson_table_ren_columns

    

train["Age"] = train["Age"].fillna(train["Age"].median())

train["Embarked"] = train["Embarked"].fillna("S")

    

train["Sex"][train["Sex"]=="male"]=0

train["Sex"][train["Sex"] == "female"] = 1

train["Embarked"][train["Embarked"] == "S" ] = 0

train["Embarked"][train["Embarked"] == "C" ] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2





test["Age"] = test["Age"].fillna(test["Age"].median())

test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"] == "female"] = 1

test["Embarked"][test["Embarked"] == "S"] = 0

test["Embarked"][test["Embarked"] == "C"] = 1

test["Embarked"][test["Embarked"] == "Q"] = 2

test.Fare[152] = test.Fare.median()



#「train」の目的変数と説明変数の値を取得

train_y=train["Survived"]

train_x=train[["Pclass","Sex","Age","Fare","SibSp","Parch","Embarked"]]

lr=LogisticRegression()

lr.fit(train_x,train_y)

test_x=test[["Pclass","Sex","Age","Fare","SibSp","Parch","Embarked"]]



pred = lr.predict(test_x)

pred.shape



PassengerId=np.array(test["PassengerId"]).astype(int)

my_solution=pd.DataFrame(pred,PassengerId,columns=["Survived"])

my_solution.to_csv("solution.csv",index_label=["PassengerId"])
