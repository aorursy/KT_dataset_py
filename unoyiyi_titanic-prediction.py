# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")

gendersub = pd.read_csv("../input/gender_submission.csv")
train.shape, test.shape, gendersub.shape
train.head(5)

test.head(5)
print('Train columns with null values:\n', train.isnull().sum())

print("-"*10)

      

print('Test columns with null values:\n', test.isnull().sum())

print("-"*10)

      

test.describe(include = "all")
#COMPLETING

test.count(),train.count()
train.groupby(['Pclass']).mean()

#drop values & replace missing values 

train = train.drop(['PassengerId','Name','Ticket'], axis = 1)

test = test.drop(['Name','Ticket'], axis=1)

test["Age"].fillna(test["Age"].mean(), inplace=True)

train["Age"].fillna(train["Age"].mean(), inplace=True)

test["Fare"].fillna(test["Fare"].mean(), inplace=True)

#because cabin miss too many values, so we drop it directly

train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis=1)

#because embarked needs a lot of attention, so we will drop it now (i am running out of time)

train = train.drop(['Embarked'], axis = 1)

test = test.drop(['Embarked'], axis=1)

#change sex to 0 and 1

test["Sex"] = test["Sex"].map({"male": 0, "female":1})

train["Sex"] = train["Sex"].map({"male": 0, "female":1})

train.head()
#correlation heatmap

corr = test.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
#model1 Logistic Regression I found this model on "https://www.kaggle.com/omarelgabry/a-journey-through-titanic")

# define training and testing sets



X_train = train.drop("Survived",axis=1)

Y_train = train["Survived"]

X_test  = test.drop("PassengerId",axis=1).copy()



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)
#Model2 Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train) 

## maybe overfit
#print submission

submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": Y_pred})

submission.to_csv("titanichw1.csv",index = False)