# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 
import pandas as pd 

train = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")
train.head()
test.head()
print(train.shape)

print(test.shape)
print(train.info())

print("\r")

print(test.info())
train = train.drop(["Cabin","Name", "Ticket"], axis=1)

test = test.drop(["Cabin","Name", "Ticket"], axis=1)

train.describe()
train["Fare"] = train["Fare"].replace(np.nan, 32)

test["Fare"] = test["Fare"].replace(np.nan, 32)

train["Age"] = train["Age"].replace(np.nan, 30)

test["Age"] = test["Age"].replace(np.nan, 30)

train["Embarked"] = train["Embarked"].replace(np.nan, "C")
print(train.info())

print("\r")

print(test.info())
train["Sex"].replace(["female","male"] , [0,1], inplace = True)

test["Sex"].replace(["female","male"] , [0,1], inplace = True)

train["Embarked"].replace(["Q","S","C"],[0,1,2],inplace=True)

test["Embarked"].replace(["Q","S","C"],[0,1,2],inplace=True)

train.head()
bins = [0,8,15,20,40,60,100]

names=(['Baby', 'Child', 'Teenager', 'Youngster', 'Adult', 'Senior Citizen'])



train["Age"] = pd.cut(train["Age"], bins, labels = names)

test["Age"] = pd.cut(test["Age"], bins, labels = names)

train.head()
train["Fare"] = pd.cut(train.Fare,[-1, 130, 260, 390, 520], labels=['1', '2', '3', '4'])

test["Fare"] = pd.cut(test.Fare,[-1, 130, 260, 390, 520], labels=['1', '2', '3', '4'])



train.head()
train.pivot_table(index = "Sex", values = "Survived")
sns.barplot(x="Sex", y="Survived", data=train)

plt.show()
train.pivot_table(index = "Pclass", values = "Survived")
sns.barplot(x="Pclass", y="Survived", data=train)

plt.show()
train.pivot_table(index = "Age", values = "Survived")


sns.barplot(x="Age", y="Survived", data=train)

plt.show()
train["Age"].replace(["Baby","Child","Teenager","Youngster","Adult","Senior Citizen"] , [1,2,3,4,5,6], inplace = True)

test["Age"].replace(["Baby","Child","Teenager","Youngster","Adult","Senior Citizen"] , [1,2,3,4,5,6], inplace = True)

train.head()
train.dtypes    #Checking if all data types are redable or not
#Declaring Model

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

columns = ['Pclass', 'Sex', 'SibSp','Embarked', 'Age', 'Fare']
from sklearn.model_selection import train_test_split

X = train[columns]

y = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=0)
# Checking Accuracy

from sklearn.metrics import accuracy_score

lr.fit(X_train,y_train)

predictions = lr.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(accuracy)
print("Intercept",lr.intercept_)

print("\r")

print("Coefficient",lr.coef_)   
lr.fit(X,y)

test_predictions = lr.predict(test[columns])
#Submission dataframe

test_ids = test["PassengerId"]

submission_df = {"PassengerId": test_ids,

                 "Survived": test_predictions}

submission = pd.DataFrame(submission_df)

submission.head(10)
submission.to_csv("submission.csv",index=False)





print(lr.score(X_test, y_test))
fig=plt.figure(figsize=(4,5))

sns.countplot(submission['Survived'])

plt.show()
print(submission["Survived"].value_counts())

print(submission.shape)