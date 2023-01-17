# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test=pd.read_csv('../input/titanic/test.csv')

train=pd.read_csv('../input/titanic/train.csv')
train.head()
test.head()
print(train.shape)

print(test.shape)
train.info()
test.info()
train.drop(columns=['Cabin'],inplace=True) #Dropping the Cabin column as most of them are Nan 

test.drop(columns=['Cabin'],inplace=True)
train.isnull().sum()
test.isnull().sum()
train['Embarked'].value_counts()
train['Embarked'].fillna('S', inplace=True) #Filling the Nan values of the embarked column with S as it is the most probable one
train['Embarked'].value_counts()
test['Embarked'].fillna('S', inplace=True)
test[test['Fare'].isnull()]

test['Fare'].fillna(test[test['Pclass']==3]['Fare'].mean(), inplace=True)
test.isnull().sum()
train_age=np.random.randint(train['Age'].mean()-train['Age'].std(),train['Age'].mean()+train['Age'].std(), 177) #calculating the probable age 
test_age=np.random.randint(test['Age'].mean()-test['Age'].std(),test['Age'].mean()+test['Age'].std(), 86)
train['Age'][train['Age'].isnull()]=train_age
train.isnull().sum()
test['Age'][test['Age'].isnull()]=test_age
test.isnull().sum()
#EDA
train.groupby(['Pclass'])['Survived'].mean()
train.groupby(['Sex'])['Survived'].mean()
train.groupby(['Embarked'])['Survived'].mean()
sns.distplot(train['Age'][train['Survived']==0])

sns.distplot(train['Age'][train['Survived']==1])
sns.distplot(train['Fare'][train['Survived']==0])

sns.distplot(train['Fare'][train['Survived']==1])
train.drop(columns=['Ticket'],inplace=True)

test.drop(columns=['Ticket'],inplace=True)
train['Family']=train['SibSp']+train['Parch']+1
test['Family']=test['SibSp']+test['Parch']+1
train['Family'].value_counts()
train.groupby(['Family'])['Survived'].mean()
def cal(number):

    if number==1:

        return "Alone"

    elif number>1 and number<5:

        return "Medium"

    else:

        return "Large"
train['Family_size']=train['Family'].apply(cal)

test['Family_size']=test['Family'].apply(cal)
train.head()
train.drop(columns=['SibSp','Parch','Family'],inplace=True)

test.drop(columns=['SibSp','Parch','Family'],inplace=True)
train['Initials']=0

for i in train:

    train['Initials']=train.Name.str.extract('([A-Za-z]+)\.')
train['Initials']
train.drop(columns=['Name'],inplace=True)
test['Initials']=0

for i in train:

    test['Initials']=test.Name.str.extract('([A-Za-z]+)\.')
train['Initials'].value_counts()
test['Initials'].value_counts()
train['Initials'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Master'],['Other','Other','Miss','Other','Other','Mrs','Mrs','Other','Other','Other','Mr','Mr','Other','Mr'],inplace=True)

test['Initials'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Master','Dona'],['Other','Other','Miss','Other','Other','Mrs','Mrs','Other','Other','Other','Mr','Mr','Other','Mr','Other'],inplace=True)

train['Initials'].value_counts()
test.drop(columns=['Name'],inplace=True)
train.head()
train.groupby(['Initials'])['Survived'].mean()

train.info()
PassengerId=test['PassengerId'].values
train.drop(columns=['PassengerId','Fare'],inplace=True)

test.drop(columns=['PassengerId','Fare'],inplace=True)
print(train.shape)

print(test.shape)
train=pd.get_dummies(train,columns=['Pclass','Sex','Embarked','Family_size','Initials'])
test=pd.get_dummies(test,columns=['Pclass','Sex','Embarked','Family_size','Initials'])
train.head()
test.info()
train.info()
print(train.shape)

print(test.shape)
X=train.iloc[:,1:].values

y=train.iloc[:,0].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=1)
from sklearn.tree import DecisionTreeClassifier

clf= DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
param_dist={

    "criterion":["gini","entropy"],

    "max_depth":[1,2,3,4,5,6,7,None],

    "splitter" : ["best", "random"],

    "max_leaf_nodes": [1,2,3,4,5,6,7, None]

}
from sklearn.model_selection import GridSearchCV #using GridSearchCV for better accuracy

grid=GridSearchCV(clf, param_grid=param_dist, cv=10, n_jobs=-1)
grid.fit(X_train, y_train)
grid.best_estimator_
grid.best_score_
X_final=test.iloc[:,:].values
y_final=grid.predict(X_final)
y_final.shape
PassengerId.shape
final=pd.DataFrame()
final
final['PassengerId']=PassengerId

final['Survived']=y_final
final
final.to_csv('submission_final.csv',index=False)