# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
titanic_train = pd.read_csv("../input/train.csv")

titanic_test = pd.read_csv("../input/test.csv")
titanic_train.info()
titanic_test.info()
#Let's set the survived to -1 and then we combine both datasets

titanic_test["Survived"] = -1

titanic = titanic_train.append(titanic_test)
#let's check it now

titanic.Survived.value_counts()
titanic.sample(3)
titanic[["Age","Fare"]].boxplot()
titanic[titanic.Fare > 500]
temp = titanic_train.groupby("Sex")["Survived"].value_counts(normalize = True).mul(100).reset_index(name = "percentage")

sns.barplot(x="Sex",y = "percentage",hue = "Survived",data = temp).set_title("GENDER _ SURVIVAL")
temp = titanic[(titanic.Survived!=-1)].groupby("SibSp")["Survived"].value_counts(normalize = True).mul(100).reset_index(name = "percentage")

sns.barplot(x="SibSp",y = "percentage",hue = "Survived",data = temp).set_title("SibSp - Survival rate")

temp = titanic[(titanic.Survived!=-1)].groupby("Parch")["Survived"].value_counts(normalize = True).mul(100).reset_index(name = "percentage")

sns.barplot(x="Parch",y = "percentage",hue = "Survived",data = temp).set_title("Parch vs Survival Rate")
titanic.sample()
titanic.drop(columns=["Ticket","PassengerId","Cabin"], inplace = True)
titanic.isna().sum()
titanic[titanic.Age.isna()].head(3)
"""

Storing the titles of passengers in title list and then adding it to titanic dataframe

"""

title = []

for item in titanic.Name:

    title.append(item.split(',')[1].split('.')[0].strip())

print (title[:3])

print (titanic.Name[:3])

titanic["title"] = title

titanic.title.value_counts()
using = dict(titanic.groupby("title").mean()["Age"])

sns.barplot(x = list(using.keys()), y = list(using.values()))

plt.xticks(rotation = 90)
final_age = []

for i in range(len(titanic)):

    age = titanic.iloc[i,0]

    if np.isnan(age):

        age = using[titanic.iloc[i,-1]]

    final_age.append(age)

titanic["Age"] = final_age
titanic.isna().sum()
sns.countplot(x="Embarked", data = titanic)
titanic.Embarked.fillna("S",inplace=True)
titanic.isna().sum()
sns.barplot(x="Embarked",y="Fare",hue = "Pclass",data = titanic)
titanic[titanic.Fare.isna()]
titanic.Fare.fillna(18,inplace=True)

titanic.isna().sum()
temp = titanic[(titanic.Survived!=-1)].groupby("Parch")["Survived"].value_counts(normalize = True).mul(100).reset_index(name = "percentage")

sns.barplot(x="Parch",y = "percentage",hue = "Survived",data = temp).set_title("Parch vs Survival Rate")
Parch = titanic.Parch.tolist()

is_par = [0 if item == 0 else 1 for item in Parch ]

titanic["is_par"] = is_par
temp = titanic[(titanic.Survived!=-1)].groupby("SibSp")["Survived"].value_counts(normalize = True).mul(100).reset_index(name = "percentage")

sns.barplot(x="SibSp",y = "percentage",hue = "Survived",data = temp).set_title("SibSp - Survival rate")
SibSp = titanic.SibSp.tolist()

has_sib = [0 if item == 0 else 1 for item in SibSp ]

titanic["has_sib"] = has_sib
sns.countplot(x = "Embarked", hue = "Survived",data = titanic[titanic.Survived != -1])
temp = titanic[(titanic.Survived!=-1)].groupby("Pclass")["Survived"].value_counts(normalize = True).mul(100).reset_index(name = "percentage")

sns.barplot(x="Pclass",y = "percentage",hue = "Survived",data = temp).set_title("SibSp - Survival rate")
titanic.sample(4)
titanic[titanic.Survived!=-1].groupby("Survived").mean()[["Age","Fare"]]
titanic.drop(columns=["Name","Parch","SibSp","title"], inplace=True)

titanic.sample()
titanic = pd.get_dummies(titanic, columns=["Embarked","Pclass"])

titanic.Sex = titanic.Sex.map({"male":1,"female":0})

titanic.sample()
titanic_training_y = titanic[titanic.Survived!=-1].Survived

titanic_training_x = titanic[titanic.Survived!=-1].drop(columns = ["Survived"])

from sklearn.model_selection import train_test_split

for random in range(15):

    train_x, test_x, train_y, test_y = train_test_split(titanic_training_x, titanic_training_y, test_size = 0.1)

    from xgboost import XGBClassifier

    from sklearn.metrics import accuracy_score

    scores = []

    for i in range(5,15):

        model = XGBClassifier(max_depth = i)

        model.fit(train_x, train_y)

        target = model.predict(test_x)

        score = accuracy_score(test_y, target)

        scores.append(score)

    print("best scores: ",max(scores), " at depth : ",scores.index(max(scores))+5)
titanic_training_y = titanic[titanic.Survived!=-1].Survived

titanic_training_x = titanic[titanic.Survived!=-1].drop(columns = ["Survived"])

test_x = titanic[titanic.Survived==-1].drop(columns = ["Survived"])

model = XGBClassifier(max_depth = i)

model.fit(titanic_training_x, titanic_training_y)

target = model.predict(test_x)

print (target[:4])

print (test_x[:4])
titanic_test = pd.read_csv("../input/test.csv")

titanic_test = pd.DataFrame(titanic_test["PassengerId"])

titanic_test["Survived"] = target

titanic_test.head()

titanic_test.to_csv("predictions.csv")