import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
traindata = pd.read_csv('../input/train.csv')
traindata.head()

testdata = pd.read_csv('../input/test.csv')
testdata.head()
testdata.info()
traindata.info()
# Number of null value in our dataset for each variable
print(pd.isnull(traindata).sum())
# Distribution of survival vs not survival
surv = traindata[traindata['Survived']==1]
nosurv = traindata[traindata['Survived']==0]

print(f"Survived:",len(surv), round(len(surv)/len(traindata)*100.0, 2), 
      "perecnt;\n""Not Survived:",len(nosurv), round(len(nosurv)/len(traindata)*100.0, 2), "perecnt.\n"
     "Total:", len(traindata))
# Survival by gender
plt.subplot()
sns.barplot('Sex', 'Survived', data=traindata)
# Survival by social class
plt.subplot()
sns.barplot('Pclass', 'Survived', data=traindata)
# Survival by Embarked
plt.subplot()
sns.barplot('Embarked', 'Survived', data=traindata)
# Survival by number of sibling/spouse
plt.subplot()
sns.barplot('SibSp', 'Survived', data=traindata)
# Survival by parch
plt.subplot()
sns.barplot('Parch', 'Survived', data=traindata)
# Survival by Age
traindata["Age"] = traindata["Age"].fillna(-0.5)
testdata["Age"] = testdata["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
traindata['AgeGroup'] = pd.cut(traindata["Age"], bins, labels = labels)
testdata['AgeGroup'] = pd.cut(testdata["Age"], bins, labels = labels)

sns.barplot(x="AgeGroup", y="Survived", data=traindata)
plt.show()
# Survival by Cabin feature
traindata["CabinBool"] = (traindata["Cabin"].notnull().astype('int'))
testdata["CabinBool"] = (testdata["Cabin"].notnull().astype('int'))
plt.subplot()
sns.barplot(x="CabinBool", y="Survived", data=traindata)
plt.show()
print(f' Number of people embarking in Southampton:', len(traindata[traindata["Embarked"] == "S"]), ';\n',
     f'Number of people embarking in Cherbourg:', len(traindata[traindata["Embarked"] == "C"]), ';\n',
     f'Number of people embarking in Queenstown:', len(traindata[traindata["Embarked"] == "Q"]), ';')

# Since Southampton has most embarking number, we fill the null value with "S"
traindata = traindata.fillna({"Embarked": "S"})

print(f' Number of people embarking in Southampton:', len(traindata[traindata["Embarked"] == "S"]), ';\n',
     f'Number of people embarking in Cherbourg:', len(traindata[traindata["Embarked"] == "C"]), ';\n',
     f'Number of people embarking in Queenstown:', len(traindata[traindata["Embarked"] == "Q"]), ';')
# create a combined group of both datasets
combine = [traindata, testdata]

# extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(traindata['Title'], traindata['Sex'])
# replace various titles with more common names
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

traindata[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# map each of the title groups to a numerical value
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

traindata.head()
# fill missing age with mode age group for each title
mr_age = traindata[traindata["Title"] == 1]["AgeGroup"].mode() 
miss_age = traindata[traindata["Title"] == 2]["AgeGroup"].mode() 
mrs_age = traindata[traindata["Title"] == 3]["AgeGroup"].mode() 
master_age = traindata[traindata["Title"] == 4]["AgeGroup"].mode()
royal_age = traindata[traindata["Title"] == 5]["AgeGroup"].mode() 
rare_age = traindata[traindata["Title"] == 6]["AgeGroup"].mode() 

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}
traindata = traindata.fillna({"Age": traindata["Title"].map(age_title_mapping)})
testdata = testdata.fillna({"Age": testdata["Title"].map(age_title_mapping)})
for x in range(len(traindata["AgeGroup"])):
    if traindata["AgeGroup"][x] == "Unknown":
        traindata["AgeGroup"][x] = age_title_mapping[traindata["Title"][x]]
        
for x in range(len(testdata["AgeGroup"])):
    if testdata["AgeGroup"][x] == "Unknown":
        testdata["AgeGroup"][x] = age_title_mapping[testdata["Title"][x]]
# map each Age value to a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
traindata['AgeGroup'] = traindata['AgeGroup'].map(age_mapping)
testdata['AgeGroup'] = testdata['AgeGroup'].map(age_mapping)

traindata.head()
# map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
traindata['Sex'] = traindata['Sex'].map(sex_mapping)
testdata['Sex'] = testdata['Sex'].map(sex_mapping)

traindata.head()
# map each Embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
traindata['Embarked'] = traindata['Embarked'].map(embarked_mapping)
testdata['Embarked'] = testdata['Embarked'].map(embarked_mapping)

traindata.head()
# fill in missing Fare value in test based on mean fare for that Pclass 
for x in range(len(testdata["Fare"])):
    if pd.isnull(testdata["Fare"][x]):
        pclass = testdata["Pclass"][x] 
        testdata["Fare"][x] = round(traindata[traindata["Pclass"] == pclass]["Fare"].mean(), 4)
        
#map Fare values into groups of numerical values
traindata['FareBand'] = pd.qcut(traindata['Fare'], 4, labels = [1, 2, 3, 4])
testdata['FareBand'] = pd.qcut(testdata['Fare'], 4, labels = [1, 2, 3, 4])

traindata.head()
traindata = traindata.drop(['Cabin', 'Ticket', 'Name', 'Age', 'Fare'], axis = 1)
testdata = testdata.drop(['Cabin','Ticket', 'Name', 'Age', 'Fare'], axis = 1)

traindata.head()
from sklearn.model_selection import train_test_split

predictors = traindata.drop(['Survived', 'PassengerId'], axis=1)
target = traindata["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, random_state = 0)
# Testing different model

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)
# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)
#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)
# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)
# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)
# models comparision
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Decision Tree', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_decisiontree
              , acc_gbk]})
models.sort_values(by='Score', ascending=False)
ids = testdata['PassengerId']
predictions = randomforest.predict(testdata.drop('PassengerId', axis=1))

output = pd.DataFrame({'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)