import numpy as np
import pandas as pd
df = pd.read_csv("../input/titanic/train.csv")
df.head()
df1 = pd.read_csv("../input/titanic/test.csv")
df1.head()
import sklearn
from sklearn.tree import DecisionTreeClassifier
df1.head()
#testデータの欠損値を調べる
df1.isnull().sum()
#Ageの欠損値を修正する
df1 = df1.fillna(df1.mean())
df1.isnull().sum()
#trainデータの欠損値を修正する
df.isnull().sum()
#Ageデータの欠損値を修正する
df = df.fillna(df1.mean())
df.isnull().sum()
df["Sex"] = df["Sex"].map({"male":0,"female":1})
df1["Sex"] = df1["Sex"].map({"male":0,"female":1})
#Embarkedの欠損値を修正する
df["Embarked"] = df["Embarked"].fillna("S")
df.isnull().sum()
#決定木の予測モデルを利用し、生存者を推定する
from sklearn import tree
target = df["Survived"].values
features_one = df[["Pclass","Sex","Age","Fare"]].values
my_tree_one = tree.DecisionTreeClassifier()
from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(df,test_size = 0.2,random_state = 1)
clf1 = DecisionTreeClassifier(max_depth = 4)
clf1 = clf1.fit(features_one,target)
test_features = df1[["Pclass","Sex","Age","Fare"]].values
my_prediction = clf1.predict(test_features)
my_prediction.shape
PassengerId = np.array(df1["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction,PassengerId,columns=["Survived"])
my_solution.to_csv("my_tree_one.csv",index_label = ["PassengerId"])
import os
for dirname, _, filenames in os.walk('./'):

    for filename in filenames:

        print(os.path.join(dirname, filename))