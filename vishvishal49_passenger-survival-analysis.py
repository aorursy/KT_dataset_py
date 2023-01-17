import pandas as pd
import numpy as np
from sklearn import linear_model
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
train = pd.read_csv("../input/train.csv")
train.columns
train.describe()
train.dtypes
train.drop("Cabin", axis=1, inplace=True)
train.drop("Name", axis=1, inplace=True)
train.drop("Ticket", axis=1, inplace=True)

train['Age'] = train['Age'].fillna(train['Age'].mean())
#train["Cabin"] = train["Cabin"].fillna(value ='ABC')
train["Embarked"] = train["Embarked"].fillna(value ='S')
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
#train["Name"] = lb_make.fit_transform(train["Name"])
train["Sex"] = lb_make.fit_transform(train["Sex"])
#train["Ticket"] = lb_make.fit_transform(train["Ticket"])
#train["Cabin"] = lb_make.fit_transform(train["Cabin"])
train["Embarked"] = lb_make.fit_transform(train["Embarked"])
y_da =train.filter(regex = 'Survived')[:].values
y = y_da.ravel()
X = train.drop('Survived',axis=1)[:]
#Split the test data to test and validation data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33,random_state=42)
regressor = LogisticRegression()
regressor.fit(X_train,y_train)
# Predicting the Test set results
y_pred = regressor.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm
test = pd.read_csv("../input/test.csv")
test.dtypes
test.drop("Cabin", axis=1, inplace=True)
test.drop("Name", axis=1, inplace=True)
test.drop("Ticket", axis=1, inplace=True)

test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
#test["Cabin"] = test["Cabin"].fillna(value ='ABC')
test["Embarked"] = test["Embarked"].fillna(value ='S')
from sklearn.preprocessing import LabelEncoder
lb_make1 = LabelEncoder()
#test["Name"] = lb_make1.fit_transform(test["Name"])
test["Sex"] = lb_make1.fit_transform(test["Sex"])
#test["Ticket"] = lb_make1.fit_transform(test["Ticket"])
#test["Cabin"] = lb_make1.fit_transform(test["Cabin"])
test["Embarked"] = lb_make1.fit_transform(test["Embarked"])
test.columns
# Predicting the Test set results
test_pred = regressor.predict(test)
dat1 = pd.DataFrame({'Survived': test_pred})
test = test.join(dat1)
test.columns
predicted_report = test[['PassengerId','Survived']]
predicted_report.to_csv('predicted.csv')