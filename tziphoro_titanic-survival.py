import pandas as pd

import numpy as np

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
train_set = pd.read_csv("../input/titanic/train.csv")

test_set = pd.read_csv("../input/titanic/test.csv")
## training data

train_set.head()

train_set.isnull().sum()

train_set.dropna(axis=0, subset=["Embarked"], inplace=True)

train_set["Name"] = train_set["Name"].str.split('.').str[0]

train_set["Name"] = train_set["Name"].str.split(',').str[1]



y = train_set["Survived"]

X = train_set.drop(["Cabin", "Ticket", "PassengerId", "Survived"], axis=1)

X["Sex"] = (X["Sex"]  == "female").astype(int)

X = pd.get_dummies(X)



# impute missing train values

isnull_Age = X["Age"].isnull().astype(int)

imputer = IterativeImputer()

age = np.array(pd.DataFrame(imputer.fit_transform(X))[2])

X["Age"] = age

X["isnull_Age"] = isnull_Age

X.isnull().sum()

X.drop(['Name_ Capt', 'Name_ Don', 'Name_ Jonkheer', 'Name_ Lady', 'Name_ Major', \

        'Name_ Mlle', 'Name_ Mme', 'Name_ Sir', 'Name_ the Countess'], axis = 1, inplace = True)



## test data

test_set.head()

test_set.isnull().sum()

test_set["Name"] = test_set["Name"].str.split('.').str[0]

test_set["Name"] = test_set["Name"].str.split(',').str[1]



Xtest = test_set.drop(["Cabin", "Ticket", "PassengerId"], axis=1)

Xtest["Sex"] = (Xtest["Sex"]  == "female").astype(int)

Xtest = pd.get_dummies(Xtest)



# impute missing test values

isnull_Age = Xtest["Age"].isnull().astype(int)

imputer = IterativeImputer()

age = np.array(pd.DataFrame(imputer.fit_transform(Xtest))[2])

fare = np.array(pd.DataFrame(imputer.fit_transform(Xtest))[5])

Xtest["Age"] = age

Xtest["Fare"] = fare

Xtest["isnull_Age"] = isnull_Age

Xtest.isnull().sum()

Xtest.drop('Name_ Dona', axis = 1, inplace=True)
# Random Forest Model

rf = RandomForestClassifier(n_estimators = 500, max_depth = 7, oob_score = True)

rf.fit(X, y)

print('OOB Score: ', rf.oob_score_)

rf.score(X, y)

yhat_in_sample = rf.predict(X)

print('Confusion Matrix (In Sample): \n', confusion_matrix(y, yhat_in_sample))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

rf.fit(X_train, y_train)

print('OOB Score: ', rf.oob_score_)

yhat = rf.predict(X_test)

np.mean((y_test - yhat)**2)

print('Confusion Matrix: \n', confusion_matrix(y_test, yhat))
# predict test data

rf.fit(X, y)

yhat_test = rf.predict(Xtest)

print(yhat_test)



predictions = {"PassengerId" : test_set["PassengerId"], "Survived" : yhat_test}

predictions = pd.DataFrame(predictions)

predictions.to_csv('survived_submission.csv', index = False)