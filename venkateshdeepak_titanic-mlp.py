# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from sklearn.neural_network import MLPClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
train[train["Cabin"].isin(["B58 B60"])]
def cabin(t):

    if isinstance(t,str):

        return t[0]

    return "other"
train["Cabin"] = train["Cabin"].apply(cabin)
train["Cabin"].value_counts()
feature = ["Pclass","Age","SibSp","Parch"]

label = "Survived"
X_train = train[feature].copy()

Y = train[label]
enc = preprocessing.OneHotEncoder()
enc.fit(train["Sex"].values.reshape(-1,1))
enc.transform(train["Sex"].values.reshape(-1,1)).toarray()
ens = preprocessing.OneHotEncoder()
ens.fit(train["Embarked"].fillna("S").values.reshape(-1,1))
enb = preprocessing.OneHotEncoder()
enb.fit(train["Cabin"].fillna("others").values.reshape(-1,1))
enb.categories_[0]


for cab in enb.categories_[0]:

    print(cab)

    X_train[cab] = 0
enc.categories_
X_train["male"] =0

X_train["female"]=0
X_train[["male","female"]] = enc.transform(train["Sex"].values.reshape(-1,1)).toarray()
X_train["S"] = 0

X_train["C"] = 0

X_train["Q"] = 0
X_train[["S","C","Q"]]= ens.transform(train["Embarked"].fillna("S").values.reshape(-1,1)).toarray()
X_train[enb.categories_[0]]=enb.transform(train["Cabin"].fillna("others").values.reshape(-1,1)).toarray()
X_train.isnull().sum()
X_train["Age"]=X_train["Age"].fillna(0)
X_train.isnull().sum()
Y.isnull().sum()
mlp = MLPClassifier(hidden_layer_sizes=(25,),max_iter=500)
mlp.fit(X_train,Y)
mlp.score(X_train,Y)
X_test = test[feature].copy()
enc.transform(test["Sex"].values.reshape(-1,1)).toarray()
ens.transform(test["Embarked"].fillna("S").values.reshape(-1,1)).toarray()
test["Cabin"] = test["Cabin"].apply(cabin)
enb.transform(test["Cabin"].fillna("others").values.reshape(-1,1)).toarray()
X_test["male"] = 0

X_test["female"] = 0

X_test[["male","female"]] = enc.transform(test["Sex"].values.reshape(-1,1)).toarray()
X_test["Q"] = 0

X_test["S"] = 0

X_test["C"] = 0
X_test[["Q","S","C"]] = ens.transform(test["Embarked"].fillna("S").values.reshape(-1,1)).toarray()
for cab in enb.categories_[0]:

    print(cab)

    X_test[cab] = 0
X_test.head()
X_test.isnull().sum()
X_test["Age"] = X_test["Age"].fillna(0)
mlp.predict(X_test)
test["Survived"] = mlp.predict(X_test)
test.columns
test[["PassengerId","Survived"]].to_csv("MLP titanic.csv",index=False)