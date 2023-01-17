import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
train.info()
train.head()
plt.figure(figsize = (10,8))

sns.heatmap(train.corr(), cmap= sns.color_palette(palette="RdBu"),linewidths=.5,annot=True)

plt.title("Correlation Matrix of Train DF")

plt.yticks(rotation = "0")
train.corr()["Survived"] # Fare, Pclass
sns.countplot(data = train, x = "Sex", hue = "Survived")

plt.title("Count of Sexes by Survival")
sns.boxplot(data = train, y = "Fare", x = "Survived")

plt.title("Boxplot of Fare by Survival")
sns.boxplot(data = train, y = "Fare", x = "Sex", hue = "Survived")

plt.title("Boxplot of Fare by Sex, and Survival")
sns.boxplot(data = train, x = "Sex", y = "Age", hue = "Survived")

plt.title("Age vs Sex grouped by Survived")
sns.boxplot(data = train, x = "Survived", y = "Age", hue = "Pclass",)

plt.title("Age vs Sex grouped by Survived")
train["bool_cabin"] = train["Cabin"].isna() == False

train["bool_cabin"].head()
sns.countplot(data = train, x = "bool_cabin", hue = "Survived")

plt.title("Count of Known/Unknown Passenger's Cabins grouped by Survival")
x = train["bool_cabin"].values

y = train["Survived"].values

print(np.corrcoef(x,y)[1,0])
sns.countplot(data = train, x = "SibSp", hue = "Survived")
sns.countplot(data = train, x = "Parch", hue = "Survived")
train.isna().sum()
train[train["Embarked"].isna()]
train["Embarked"].value_counts()
train.groupby(["Sex","Embarked"]).count()["PassengerId"].rename({"PassengerId":"Count"})
train.groupby(["Sex","Pclass","Embarked"]).count()["PassengerId"].rename({"PassengerId":"Count"})
def fill_embarked(df,fill = "S"):

    copy = df.copy()

    copy.loc[copy.isna()["Embarked"],"Embarked"] = fill

    return copy
example = fill_embarked(train)

example.isna()[["Embarked"]].sum()
train.isna()[["Age"]].sum()
train[["Age"]].mean()
train[["Age"]].median()
temp = train.groupby(["Sex","Pclass"]).median()[["Age"]]

temp = temp.merge(train.groupby(["Sex","Pclass"]).mean()[["Age"]], left_index = True, right_index = True, suffixes= ("Median","Mean"))

age = temp.merge(train.groupby(["Sex","Pclass"]).count()[["Age"]], left_index= True, right_index = True, suffixes = ("Count","Count"))

age
def fill_age(df, method = np.median):

    global age

    copy = df.copy()

    copy.loc[copy["Age"].isna(),"Age"] = method(copy.dropna()["Age"])

    return copy    
temp = train.copy()

temp = fill_age(temp)

temp["Age"].isna().sum()
train.count().max(), train.isna()["Cabin"].sum()
train = pd.read_csv("../input/train.csv")
train = fill_age(train)

train = fill_embarked(train)

train["bool_cabin"] = train["Cabin"].isna() == False

train.drop("Cabin",axis=1,inplace = True)
train.isna().sum()
def one_hot_encoding(df):

    copy = df.copy()

    columns = copy.columns.tolist()

    

    first_column = copy[columns[0]]

    temp = pd.get_dummies(first_column, prefix = columns[0], drop_first = True)

    

    for i in columns[1:]:

        curr_column = copy[i]

        curr_df = pd.get_dummies(curr_column, prefix = i, drop_first = True)

        temp = temp.merge(curr_df,right_index = True, left_index = True)

    return temp
train = pd.read_csv("../input/train.csv")

train = fill_age(train)

train = fill_embarked(train)

train["bool_cabin"] = train["Cabin"].isna() == False

train.drop("Cabin",axis=1,inplace = True)

num = ["SibSp","Parch","Fare", "Age","Survived"]

cats = ["Pclass","Sex","Embarked","bool_cabin"]

train_cats = train[cats]

train_num = train[num]

train_cats = one_hot_encoding(train_cats)

train_x = train_num.merge(train_cats, right_index = True, left_index= True)

train_x.head()
x = train_x.copy()

x.drop("Survived", axis=1,inplace = True)

y = train_x["Survived"]
from sklearn.linear_model import LogisticRegressionCV
model = LogisticRegressionCV(cv = 5, fit_intercept= True)

model.fit(x,y)

model.score(x,y)
num = ["SibSp","Parch","Fare", "Age"]

cats = ["Pclass","Sex","Embarked","bool_cabin"]
test = pd.read_csv("../input/test.csv")

test = fill_age(test)

test = fill_embarked(test)

test["bool_cabin"] = test["Cabin"].isna() == False

test.drop("Cabin",axis=1,inplace = True)

PassengerId = test["PassengerId"]

test_cats = test[cats]

test_num = test[num]

test_cats = one_hot_encoding(test_cats)

test_x = test_num.merge(test_cats, right_index = True, left_index= True)

test_x.head()
test_x.isna().sum()
test_x.loc[test["Fare"].isna(),"Fare"] = np.mean(test["Fare"])
predictions = model.predict(test_x)
submission = pd.DataFrame({"PassengerId":PassengerId, "Survived":predictions})
submission.to_csv("submission.csv", index=False)