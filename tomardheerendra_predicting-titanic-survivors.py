# data analysis and wrangling
import numpy as np
import pandas as pd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
%matplotlib inline

# machine learning
#Let's import them as and when needed

import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input/"))
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data.head()
test_data.head()
train_data.info()
print('-'*50)
test_data.info()
train_data.describe()
train_data.describe(include=['O'])
#drop the Cabin and Ticket columns in both dataset. We also don't need the PassengerId in training dataset.

train_data.drop(labels = ['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.drop(labels = ['Ticket', 'Cabin'], axis=1, inplace=True)
print('Training Data')
print(pd.isnull(train_data).sum())
print("-"*50)
print('Testing Data')
print(pd.isnull(test_data).sum())
sns.distplot(train_data['Age'].dropna())
train_data['Age'].fillna(train_data['Age'].median(), inplace= True)
test_data['Age'].fillna(test_data['Age'].median(), inplace= True)
train_data['Embarked'].fillna("S", inplace = True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace = True)
print('Training Data')
print(pd.isnull(train_data).sum())
print("-"*50)
print('Testing Data')
print(pd.isnull(test_data).sum())
train_data.head()
test_data.head()
plt.figure(figsize=(8,4))
plt.tight_layout()
sns.barplot(x='Sex', y='Survived', data=train_data)
plt.title('Distribution of Survival based on Gender')
plt.show()
plt.figure(figsize=(8,4))
plt.tight_layout()
sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.title('Distribution of Survival based on Class')
plt.show()
plt.figure(figsize=(12,6))
plt.tight_layout()
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train_data)
plt.title('Distribution of Survival based on Gender and Class')
plt.show()
g = sns.FacetGrid(train_data, col="Survived", margin_titles=True)
g.map(plt.hist, "Age", color="steelblue", bins=20)
sns.swarmplot(x="Survived", y="Age", hue="Sex", palette=["r", "c", "y"], data=train_data)
sns.pairplot(train_data)
train_data.head()
test_data.head()
train_data['Sex'] = train_data['Sex'].map( {'female': 1, 'male': 0}).astype(int)
train_data.head()
test_data['Sex'] = test_data['Sex'].map( {'female': 1, 'male': 0}).astype(int)
test_data.head()
train_data['Embarked'] = train_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2}).astype(int)
test_data['Embarked'] = test_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2}).astype(int)
train_data.head()
test_data.head()
train_data["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1
test_data["FamilySize"] = test_data["SibSp"] + test_data["Parch"] + 1
train_data['Single'] = train_data.FamilySize.apply(lambda x: 1 if x == 1 else 0)
test_data['Single'] = test_data.FamilySize.apply(lambda x: 1 if x == 1 else 0)
train_data.head()
test_data.head()
train_data['Title'] = train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False) ##Regular Expression <3
test_data['Title'] = test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_data['Title'], train_data['Sex'])
title_list = list(set(train_data['Title']))
title_list
mix = [train_data, test_data] ## Just so save ourself some time and repeated code.

for dt in mix:
    dt['Title'] = dt['Title'].replace(['Dr', 'Col', 'Sir', 'Countess', 'Jonkheer', 'Lady', 'Don', 'Capt', 'Major', 'Rev', \
                                      ], 'Unique')
    dt['Title'] = dt['Title'].replace('Mlle', 'Miss')
    dt['Title'] = dt['Title'].replace('Ms', 'Miss')
    dt['Title'] = dt['Title'].replace('Mme', 'Mrs')
map_title = {'Mrs': 1, 'Miss': 2, 'Mr': 3, 'Master': 4, 'Unique': 5}

for dt in mix:
    dt['Title'] = dt['Title'].map(map_title)
    dt['Title'] = dt['Title'].fillna(0)
    
train_data.head()
train_data = train_data.drop(['Name', 'SibSp', 'Parch'], axis=1)
test_data = test_data.drop(['Name', 'SibSp', 'Parch'], axis=1)
train_data.head()
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# to evaluate model performance, we can use the accuracy_score function.
X_train = train_data.drop(labels=['Survived'], axis=1)
y_train = train_data['Survived']
X_test = test_data.drop('PassengerId',axis=1).copy()
X_train.shape,  y_train.shape, X_test.shape

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
accuracy_log = logreg.score(X_train, y_train)
print("The accuracy for the Logistic Regression is: " + str(accuracy_log))
# Support Vector Machines

svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
accuracy_svc = svc.score(X_train, y_train)
print("The accuracy for Support Vector Machines is:" + str(accuracy_svc))
# K Nearest Neighbours

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
accuracy_knn = knn.score(X_train, y_train)
print("The accuracy of KNN is: " + str(accuracy_knn))
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test)
accuracy_gnb = gaussian.score(X_train, y_train)
print("The accuracy of Gaussian Naive Bayes is: " + str(accuracy_gnb))
# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
Y_pred = linear_svc.predict(X_test)
accuracy_lsvc = linear_svc.score(X_train, y_train)
print("The accuracy of linear SVC is: " + str(accuracy_lsvc))
#Decision Tree

d_tree = DecisionTreeClassifier()
d_tree.fit(X_train, y_train)
Y_pred = d_tree.predict(X_test)
accuracy_d_tree = d_tree.score(X_train, y_train)
print("The accuracy of Decision Tree is: " + str(accuracy_d_tree))
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100) #tried with 80, 90, 100 no difference
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
accuracy_r_forest = random_forest.score(X_train, y_train)
print("The accuracy of Random Forest is: " + str(accuracy_r_forest))
model_performance = pd.DataFrame({
    "Model": ["Logistic Regression", "Support Vector Machines", "K Nearest Neighbours", "Gaussian Naive Bayes",
             "Linear SVC", "Decision Tree", "Random Forest"],
    "Accuracy": [accuracy_log, accuracy_svc, accuracy_knn, accuracy_gnb, accuracy_lsvc, accuracy_d_tree, accuracy_r_forest]
})

model_performance.sort_values(by="Accuracy", ascending=False)
submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": Y_pred
})

#submission.to_csv("../output/titanic.csv", index=False)