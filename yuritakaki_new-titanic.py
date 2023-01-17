import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
train.head(10)
train.isnull().sum()
test.isnull().sum()
train["Age"] = train["Age"].fillna(train["Age"].mean())
test["Age"] = test["Age"].fillna(test["Age"].mean())
train["Embarked"] = train["Embarked"].fillna("S")
test["Fare"] = test["Fare"].fillna(test["Fare"].mean())
train = train.replace("male",0).replace("female",1)
test = test.replace("male",0).replace("female",1)
train = train.replace("S",0).replace("Q",1).replace("C",2)
test = test.replace("S",0).replace("Q",1).replace("C",2)
train.isnull().sum()
X = train[["Pclass","Sex","SibSp","Fare","Parch","Embarked"]].values
y = train[["Survived"]].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 1)
model =  LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy_score(y_pred,y_test)
features = test[["Pclass","Sex","SibSp","Fare","Parch","Embarked"]].values
my_prediction = model.predict(features)
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
my_solution.to_csv("my_tree_one.csv", index_label = ["PassengerId"])