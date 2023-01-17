import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

alldata = [train, test]

print(train.info())

train.head()
#对于Embarked属性，将缺失值填充为频率最高的选项

for data in alldata:

    data['Embarked'] = data['Embarked'].fillna('S')
#测试集中Fare缺失值填充为中位数

test['Fare']=test['Fare'].fillna(test['Fare'].median())
#对年龄，缺失值填充为1个标准差内的随机值

for data in alldata:

    age_mean = data['Age'].mean()

    age_std = data['Age'].std()

    age_null_sum = data['Age'].isnull().sum()

    random_list = np.random.randint(age_mean - age_std, age_mean + age_std,size = age_null_sum)

    data.loc[data['Age'].isnull(),'Age']=random_list

    data['Age'].astype(np.int)
train.info()
test.info()
for data in alldata:

    data['Sex']=data['Sex'].map({'female':0,'male':1}).astype(int)

    data['Embarked'] = data['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
for data in alldata:

    data['family_size'] = data['Parch']+data['SibSp']+1

train[['family_size','Survived']].groupby(['family_size']).mean()
for data in alldata:

    data['Alone'] = 0

    data.loc[data['family_size']==1,'Alone']=1

train[['Alone','Survived']].groupby(['Alone']).mean()
import re

def get_title(name):

    research = re.search('([A-Za-z]+)\.',name)

    if research:

        return research.group(1)

    else:

        return ""

for data in alldata:

    data['title'] = data['Name'].apply(get_title)    
train['title'].value_counts()
for data in alldata:

    data['title'] = data['title'].replace(['Dr','Rev','Major','Col','Lady','Countess','Jonkheer','Capt','Sir','Don'],'Rare')

    data['title'] = data['title'].replace('Mlle','Miss')

    data['title'] = data['title'].replace('Ms','Miss')

    data['title'] = data['title'].replace('Mme','Mrs')

train[['title','Survived']].groupby(['title']).mean()

for data in alldata:

    data['title'] = data['title'].map({'Mr':0,'Rare':1,'Master':2,'Miss':3,'Mrs':4})

    data['title'] = data['title'].fillna(0)

    data['title'].astype('int64')
train.columns
feature = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','family_size','Alone','title']

train_X=train[feature]

train_y=train['Survived']

test_X=test[feature]

train_X.head()
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split,cross_val_score
classfier = [LogisticRegression(),

             DecisionTreeClassifier(),

             RandomForestClassifier(),

             AdaBoostClassifier(),

             SVC(),

             GaussianNB()]

results_cols=['classfier','score']

results = pd.DataFrame(columns=results_cols)

train_X_final,val_X_final,train_y_final,val_y_final = train_test_split(train_X,train_y,test_size=0.2,random_state=0)

dict={}

for clf in classfier:

    name = clf.__class__.__name__

    clf.fit(train_X_final,train_y_final)

    score = cross_val_score(clf,train_X_final,train_y_final,cv=5).mean()

    dict[name]=score

dict

    
pd.DataFrame(dict,index = [1])
model = LogisticRegression()

model.fit(train_X,train_y)

test_pred = model.predict(test_X)
submit = pd.read_csv('../input/titanic/gender_submission.csv')

submit.head()
submission = pd.DataFrame({'PassengerId':test['PassengerId'],

                           'Survived':test_pred})

submission.to_csv('submission_1.csv',index=False)