import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
train_data = pd.read_csv("../input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("../input/titanic/test.csv")

test_data
print(train_data.shape[0]," are rows and", train_data.shape[1]," are columns of train data" )

print(test_data.shape[0]," are rows and", test_data.shape[1]," are columns of test data" )
null_train_columns = [columns for columns in train_data.columns if train_data[columns].isnull().sum()>1]

for columns in null_train_columns:

    print(columns,train_data[columns].isnull().mean()*100,"% missing values")
sns.heatmap(train_data.isnull(),cbar = False)
null_test_columns = [columns for columns in test_data.columns if test_data[columns].isnull().sum()>1]

for columns in null_test_columns:

    print(columns,test_data[columns].isnull().mean()*100,"% missing values")
sns.heatmap(test_data.isnull(),cbar = False)
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

print("Embarked is ",train_data['Embarked'].isnull().mean()*100,"% missing")
print("Number of training sample are ",train_data.shape[0])

print("Every passenger Id is unique as its size is", train_data['PassengerId'].unique().shape)

print("Unique Passenger names size", train_data["Name"].unique().shape)

print("Ticket size ",train_data['Ticket'].unique().shape)
train_data = train_data.drop(["PassengerId", "Name"], axis=1)

test_data = test_data.drop(["PassengerId","Name"], axis =1)
train_data = train_data.drop(["Cabin"], axis=1)

test_data = test_data.drop(["Cabin"], axis =1)
train_data.head()
#removing Ticket column

train_data = train_data.drop(columns = ['Ticket'])

test_data = test_data.drop(columns = ['Ticket'])
sns.heatmap(train_data.isnull(),cbar = False)
sns.heatmap(test_data.isnull(),cbar = False)
train_data.head()
print('Train Sex Unique',train_data["Sex"].unique())

print('Train Embarked Unique',train_data["Embarked"].unique())

print('Test Sex Unique',test_data["Sex"].unique())

print('Test Embarked Unique',test_data["Embarked"].unique())
sex={'male':0,'female':1}

train_data.Sex=train_data.Sex.map(sex)

test_data.Sex=test_data.Sex.map(sex)

embarked = {'S':1,'Q':2,'C':3}

train_data.Embarked = train_data.Embarked.map(embarked)

test_data.Embarked = test_data.Embarked.map(embarked)
data = train_data.drop(columns=['Survived'])

data = data.append(test_data)
data.corr()
sns.catplot(x="Pclass", y="Age",kind='violin',hue='Sex', data=data,split=True)
#mean Age when Pclass is 1,2,3

data[["Pclass", "Age"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Age', ascending=False)
age = np.array(data['Age'])

pclass = np.array(data['Pclass'])

for i in range(len(age)):

    if(np.isnan(age[i])):

        if(pclass[i] == 1):

            age[i] = 40.0

        elif(pclass[i] == 2):

            age[i] = 30.0

        else:

            age[i] = 25.0

data['Age'] = age

data
sns.heatmap(data.isnull(),cbar=False)
data['Fare']=(data['Fare']-data['Fare'].min())/(data["Fare"].max()-data["Fare"].min())

data['Age']=(data['Age']-data['Age'].min())/(data["Age"].max()-data["Age"].min())

data.head()

#Checking again for any null values in Age

print("Age is ",np.round(data['Age'].isnull().mean()*100,2),"% missing in total.")
train = data[0:891]

train
train['Survived'] = train_data['Survived']

train
test = data[891:]

test
sns.catplot(x = 'Survived',y='Age',kind='swarm' ,hue='Sex',data=train_data)
sns.catplot(x = 'Survived',y='Age',kind='violin' ,hue='Sex',data=train_data,split=True)
sns.catplot(x = 'Pclass',y='Fare',hue='Survived',kind="violin" ,data=train_data,split=True)
sns.jointplot(x = train_data['Age'],y= train_data['Fare'])
sns.scatterplot(x='Age',y='Fare',hue='Survived',data=train_data)
x = np.array(train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']])

y = np.array(train['Survived'])

def data_split(x,n):

    return x[:n],x[n:]
x_train, x_valid = data_split(x,850)

y_train, y_valid = data_split(y,850)
from sklearn import svm

from sklearn.metrics import accuracy_score

clf = svm.SVC()

clf.fit(x_train, y_train)

y_pred = clf.predict(x_valid)

print("Validation Accuracy",accuracy_score(y_valid,y_pred))

y_pred = clf.predict(x_train)

print("Training Accuracy",accuracy_score(y_train,y_pred))
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_valid)

print("Validation Accuracy",accuracy_score(y_valid,y_pred))

y_pred = clf.predict(x_train)

print("Training Accuracy",accuracy_score(y_train,y_pred))
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_valid)

print("Validation Accuracy",accuracy_score(y_valid,y_pred))

y_pred = clf.predict(x_train)

print("Training Accuracy",accuracy_score(y_train,y_pred))
x_test = np.array(test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']])

clf = svm.SVC()

clf.fit(x_train, y_train)
test = test.fillna(test.mean())

x_test = test.values

y_test = clf.predict(x_test)
pid = [i+892 for i in range(len(test.values))]

submission = {'PassengerId':pid,'Survived':y_test}

submission_data = pd.DataFrame(submission)

submission_data
submission_data.to_csv('submission.csv', index=False)
from xgboost import XGBClassifier
xgb = XGBClassifier()

xgb.fit(x_train, y_train)
xgb.fit(x_train, y_train)

y_pred = xgb.predict(x_valid)

print("Validation Accuracy",accuracy_score(y_valid,y_pred))

y_pred = xgb.predict(x_train)

print("Training Accuracy",accuracy_score(y_train,y_pred))
y_test = xgb.predict(x_test)

submission_data['Survived'] = y_test

#submission_data

submission_data.to_csv('submission.csv', index=False)