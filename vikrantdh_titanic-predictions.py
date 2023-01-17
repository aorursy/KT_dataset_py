import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.tail()
test.head()
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue = 'Sex',data=train,palette = 'Set2')
sns.countplot(x='Survived',hue = 'Pclass',data=train)
sns.distplot(train["Age"].dropna(),kde=False,bins=30,color='red')
sns.countplot(x='SibSp',data=train)
train['Fare'].hist(bins=35,figsize=(13,5),color='violet')
plt.figure(figsize=(10,5))

sns.boxplot(x='Pclass',y='Age',data=train)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
titanic = pd.concat([train,test],ignore_index=True)
titanic['Title'] = titanic['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())

print(titanic['Title'].value_counts())
grouped = titanic.groupby(['Sex','Pclass', 'Title'])  

grouped.Age.median()
titanic['Age'] = grouped.Age.apply(lambda x: x.fillna(x.median()))

titanic.drop("Cabin",axis=1,inplace=True)

freq_port = titanic['Embarked'].dropna().mode()[0]

titanic['Embarked'] = titanic['Embarked'].fillna(freq_port)
titanic['Fare'].fillna(titanic['Fare'].dropna().median(),inplace=True)

titanic['Age'].fillna(titanic['Age'].dropna().median(),inplace=True)
sex_dummies = pd.get_dummies(titanic['Sex'],drop_first=True)

pclass_dummies = pd.get_dummies(titanic['Pclass'], prefix="Pclass")

title_dummies = pd.get_dummies(titanic['Title'], prefix="Title")

titanic = pd.concat([titanic,sex_dummies,pclass_dummies,title_dummies],axis=1)
titanic.drop(['PassengerId','Pclass','Sex','Embarked','Name','Ticket','Title'],axis=1,inplace=True)

titanic.head()
tit_train = titanic[:891]

tit_train.tail()
tit_test = titanic[891:]

tit_test.tail()
sns.heatmap(tit_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(tit_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
X_train = tit_train.drop('Survived',axis=1)

y_train = tit_train['Survived']

X_test = tit_test.drop('Survived',axis=1)
#  LogisticRegression



from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

predict1 = logmodel.predict(X_test).astype(int)

log_accuracy = round(logmodel.score(X_train,y_train)*100, 2)

print(log_accuracy)
#  RandomForestClassifier



from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train,y_train)

predict2 = rfc.predict(X_test).astype(int)

rfc_accuracy = round(rfc.score(X_train,y_train)*100, 2)

print(rfc_accuracy)
#  DecisionTreeClassifier



from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

predict3 = dtree.predict(X_test).astype(int)

dtree_accuracy = round(dtree.score(X_train,y_train)*100, 2)

print(dtree_accuracy)
#  K Nearest Neighbours



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

predict4 = knn.predict(X_test).astype(int)

knn_accuracy = round(knn.score(X_train,y_train)*100, 2)

print(knn_accuracy)
#  Support Vector Machines



from sklearn.svm import SVC

svmmodel = SVC()

svmmodel.fit(X_train,y_train)

predict5 = svmmodel.predict(X_test).astype(int)

svm_accuracy = round(svmmodel.score(X_train,y_train)*100, 2)

print(svm_accuracy)
models = pd.DataFrame({

    'Model': ['Logistic Regression', 'Random Forest', 'Decision Tree', 'KNN','Support Vector Machines'],

    'Score': [log_accuracy, rfc_accuracy, dtree_accuracy, knn_accuracy, svm_accuracy]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({"PassengerId":test['PassengerId'], "Survived":predict1})

submission.to_csv('submission.csv',index=False)