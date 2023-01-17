# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/train.csv")
print(df.head)
df.columns
df=df.drop("PassengerId",axis=1)
df.ix[df["Sex"]=="male","Sex"]=0

df.ix[df["Sex"]=="female","Sex"]=1
df["Age"].isnull().any()
sum(df["Age"].isnull())
df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"].unique()
sum(df["Embarked"].isnull())
df["Embarked"]=df["Embarked"].fillna("S")
df["Name_Len"]=(len(x) for x in df["Name"])
arr1=np.array(df["Name"])
arr1=[len(x) for x in arr1]
df["Name_Len"]=arr1
df["Name"]
%matplotlib inline
import matplotlib.pyplot as plt
plt.hist(df["Survived"])

plt.hist(df["Name_Len"])
plt.scatter(df["Name_Len"],df["Survived"])
df=df.drop(["Name","Name_Len"],axis=1)
df
df=df.drop("Ticket",axis=1)
df
df.ix[df["Embarked"]=="S","Embarked"]=0

df.ix[df["Embarked"]=="C","Embarked"]=1

df.ix[df["Embarked"]=="Q","Embarked"]=2
df["Embarked"].unique()
df
df.isnull().any()
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=2)
lis=df.columns

lis=lis[1:]
lis
knn.fit(df[lis],df["Survived"])
test=pd.read_csv("../input/test.csv")
test
test.columns
test=test.drop(["PassengerId","Name","Ticket","Cabin"],axis=1)
test.head()
test.ix[test["Sex"]=="male","Sex"]=0

test.ix[test["Sex"]=="female","Sex"]=1

test.ix[test["Embarked"]=="S","Embarked"]=0

test.ix[test["Embarked"]=="Q","Embarked"]=1

test.ix[test["Embarked"]=="C","Embarked"]=2
test["Embarked"]=test["Embarked"].fillna(0)
test.isnull().any()
test["Fare"]=test["Fare"].fillna(test["Fare"].median())
test["Age"]=test["Age"].fillna(test["Age"].median())
test["Survived"]=knn.predict(test[lis])
test
test1=pd.read_csv("../input/test.csv")
test3=pd.DataFrame()

test3
test["PassengerId"]=test1["PassengerId"]
test
sd=pd.DataFrame(test["PassengerId"])
sd["Survived"]=test["Survived"]
sd.to_csv("../input/testres.csv",header=True)