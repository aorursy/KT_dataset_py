import pandas as pd

data = pd.read_csv("../input/titanic/train.csv")

data.head()
data=data.fillna("")



data["Sex"]=data["Sex"].map({"male":1,"female":2, "":""})

data["Embarked"]=data["Embarked"].map({"C":1,"Q":2,"S":3, "":""})



for i in range(len(data)):

    if data.loc[i, "Pclass"]=="":

        data.loc[i, "Pclass"]=2

    if data.loc[i, "Age"]=="":

        data.loc[i, "Age"]=30

    if data.loc[i, "SibSp"]=="":

        data.loc[i, "SibSp"]=0

    if data.loc[i, "Fare"]=="":

        data.loc[i, "Fare"]=30
data.shape
features=data.loc[:,["Pclass", "Sex", "Age", "SibSp", "Fare"]]

features
label= data["Survived"]

label
from sklearn import tree

import numpy as np

clf=tree.DecisionTreeClassifier(max_leaf_nodes=20)

X=np.array(features)

y=np.array(label)

clf=clf.fit(X, y)
test=pd.read_csv("../input/titanic/test.csv")

test
test=test.fillna("")



test["Sex"]=test["Sex"].map({"male":1,"female":2, "":""})

test["Embarked"]=test["Embarked"].map({"C":1,"Q":2,"S":3, "":""})



for i in range(len(test)):

    if test.loc[i, "Pclass"]=="":

        test.loc[i, "Pclass"]=2

    if test.loc[i, "Age"]=="":

        test.loc[i, "Age"]=30

    if test.loc[i, "SibSp"]=="":

        test.loc[i, "SibSp"]=0

    if test.loc[i, "Fare"]=="":

        test.loc[i, "Fare"]=30
test
testFile=test.loc[:,["Pclass", "Sex", "Age", "SibSp", "Fare"]]

ids=test["PassengerId"]
result=list(clf.predict(testFile))
testresult=pd.DataFrame({"PassengerId": ids, "Survived":result})
testresult
testresult.to_csv("/kaggle/working/titanic_result.csv", index=False)