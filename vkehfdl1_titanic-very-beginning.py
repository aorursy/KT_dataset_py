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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings 
warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
test.describe(include="all")
train.head(7)
sns.barplot(x = "Sex", y= "Survived", data= train)
train.columns
sns.barplot(x = "Pclass", y= "Survived", data= train)
sns.barplot(x = "SibSp", y= "Survived", data= train)
sns.barplot(x = "Parch", y= "Survived", data= train)
sns.barplot(x="Embarked", y = "Survived", data=train)
train.Age = train.Age.fillna(-0.5)
test.Age = test.Age.fillna(-0.5)
train.Age.hist()
bins = [-1,0,5,12,18,24,35,60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teen', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train['Age'], bins, labels = labels)
test['AgeGroup'] = pd.cut(test['Age'], bins, labels = labels)
sns.barplot(x="AgeGroup", y = "Survived", data = train)
plt.show()
train['CabinBool'] = train.Cabin.notnull().astype('int')
test['CabinBool'] = test.Cabin.notnull().astype('int')
sns.barplot(x="CabinBool", y = "Survived", data=train)
train = train.drop(['Cabin', 'Ticket'], axis = 1)
test = test.drop(['Cabin', 'Ticket'], axis = 1)
train.Embarked.hist()
train = train.fillna({"Embarked" : "S"})
train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)
train.AgeGroup.unique()
age_group_mapping = {"Baby" : 1, "Child" : 2, "Teen" : 3, "Student" : 4, "Young Adult" : 5, "Audlt" : 6, "Senior" : 7}
train["AgeGroup"] = train["AgeGroup"].map(age_group_mapping)
test["AgeGroup"] = test["AgeGroup"].map(age_group_mapping)
sex_mapping = {"male" : 0, "female" : 1}
train["Sex"] = train["Sex"].map(sex_mapping)
test["Sex"] = test["Sex"].map(sex_mapping)
train.Embarked.unique()
Embarked_mapping = {'S' : 1, 'C' : 2, 'Q' : 3}
train["Embarked"] = train["Embarked"].map(Embarked_mapping)
test["Embarked"] = test["Embarked"].map(Embarked_mapping)
train.AgeGroup = train.AgeGroup.fillna(-1)
test.AgeGroup = test.AgeGroup.fillna(-1)
test.isnull().sum()
from sklearn.model_selection import train_test_split
predictors = train.drop(["Survived", "PassengerId"], axis = 1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val)*100, 2)
acc_gaussian
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val)*100, 2)
acc_logreg
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
svc_logreg = round(accuracy_score(y_pred, y_val)*100, 2)
print(svc_logreg)
from sklearn.ensemble import RandomForestClassifier

rd = RandomForestClassifier()
rd.fit(x_train, y_train)
y_pred = rd.predict(x_val)
rd_logreg = round(accuracy_score(y_pred, y_val)*100, 2)
print(rd_logreg)
test_predictors = train.drop('PassengerId', axis = 1)
ids = test["PassengerId"]
preds = rd.predict(test_predictors)
