import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')
train.head(10)
train.info()
sns.heatmap(train.isnull(), yticklabels=False, cmap='viridis')
sns.countplot(x='Survived',data=train, hue='Sex')
sns.countplot(x='Survived',data=train, hue='Pclass')
sns.countplot(x='Survived',data=train, hue='SibSp')
sns.distplot(train['Age'].dropna(),kde=False, bins=30)
train.info()
sns.countplot(x='SibSp',data=train, hue='Survived')
train['Fare'].hist(bins=30, figsize=(15,6))
train['Age'].mean()
plt.figure(figsize=(10,4))

sns.boxplot(x='Pclass', y='Age', data=train)
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        

        if Pclass ==1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    

    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age, axis=1)
sns.heatmap(train.isnull())
train.drop('Cabin',axis=1,inplace=True)
train.head()
sns.heatmap(train.isnull())
train.dropna(inplace=True)
sns.heatmap(train.isnull())
sex = pd.get_dummies(train['Sex'],drop_first=True)
sex.head()
embark = pd.get_dummies(train['Embarked'],drop_first=True)
embark.head()
train = pd.concat([train,sex,embark], axis=1)
train.head()
train.drop(['Name','Sex','Embarked','Ticket','PassengerId'], axis=1, inplace=True)
train.head()
train.columns

X = train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S']]



y = train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predictions))
predictions
test = pd.read_csv('../input/titanic/test.csv')
test.head()
test.info()
sns.heatmap(test.isnull())
test['Age'] = test[['Age','Pclass']].apply(impute_age, axis=1)
sns.heatmap(test.isnull())
test.drop('Cabin',axis=1,inplace=True)
test.head()
sex1 = pd.get_dummies(test['Sex'],drop_first=True)
embark1 = pd.get_dummies(test['Embarked'],drop_first=True)
test = pd.concat([test,sex1,embark1], axis=1)
test.head()
test.columns
test.drop(['Name','Sex','Embarked','Ticket','PassengerId'], axis=1, inplace=True)
sns.heatmap(test.isnull())
test.dropna(inplace=True)
sns.heatmap(test.isnull())
test.head(14)
#X_test_1 = test[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q', 'S']]
#X_test_1
# test.to_csv('test1.csv',index=False)
predictions1 = logmodel.predict(test)
predictions1
test_pred = pd.DataFrame({'Survived': predictions1.flatten()})

test_pred
result = pd.concat([test,test_pred], axis=1, sort=False)
result
#result.to_csv('Titanic_test_solution.csv',index=False)