# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

import warnings
train=pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv("../input/titanic/test.csv")

test2=pd.read_csv("../input/titanic/test.csv")

titanic=pd.concat([train, test], sort=False)

len_train=train.shape[0]
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
titanic.dtypes.sort_values()
titanic.select_dtypes(include='int').head()
titanic.select_dtypes(include='object').head()
titanic.select_dtypes(include='float').head()
titanic.isnull().sum()[titanic.isnull().sum()>0]
train.Fare=train.Fare.fillna(train.Fare.mean())

test.Fare=test.Fare.fillna(train.Fare.mean())
train.Embarked=train.Embarked.fillna(train.Embarked.mode()[0])

test.Embarked=test.Embarked.fillna(train.Embarked.mode()[0])
train['title']=train.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())

test['title']=test.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())
newtitles={

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Jonkheer":   "Royalty",

    "Don":        "Royalty",

    "Sir" :       "Royalty",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "the Countess":"Royalty",

    "Dona":       "Royalty",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Royalty"}
train['title']=train.title.map(newtitles)

test['title']=test.title.map(newtitles)
train.groupby(['title','Sex']).Age.mean()
def newage (cols):

    title=cols[0]

    Sex=cols[1]

    Age=cols[2]

    if pd.isnull(Age):

        if title=='Master' and Sex=="male":

            return 4.57

        elif title=='Miss' and Sex=='female':

            return 21.8

        elif title=='Mr' and Sex=='male': 

            return 32.37

        elif title=='Mrs' and Sex=='female':

            return 35.72

        elif title=='Officer' and Sex=='female':

            return 49

        elif title=='Officer' and Sex=='male':

            return 46.56

        elif title=='Royalty' and Sex=='female':

            return 40.50

        else:

            return 42.33

    else:

        return Age 
train.Age=train[['title','Sex','Age']].apply(newage, axis=1)

test.Age=test[['title','Sex','Age']].apply(newage, axis=1)
warnings.filterwarnings(action="ignore")

plt.figure(figsize=[12,10])

plt.subplot(3,3,1)

sns.barplot('Pclass','Survived',data=train)

plt.subplot(3,3,2)

sns.barplot('SibSp','Survived',data=train)

plt.subplot(3,3,3)

sns.barplot('Parch','Survived',data=train)

plt.subplot(3,3,4)

sns.barplot('Sex','Survived',data=train)

plt.subplot(3,3,5)

sns.barplot('Ticket','Survived',data=train)

plt.subplot(3,3,6)

sns.barplot('Cabin','Survived',data=train)

plt.subplot(3,3,7)

sns.barplot('Embarked','Survived',data=train)

plt.subplot(3,3,8)

sns.distplot(train[train.Survived==1].Age, color='green', kde=False)

sns.distplot(train[train.Survived==0].Age, color='orange', kde=False)

plt.subplot(3,3,9)

sns.distplot(train[train.Survived==1].Fare, color='green', kde=False)

sns.distplot(train[train.Survived==0].Fare, color='orange', kde=False)
train['Relatives']=train.SibSp+train.Parch

test['Relatives']=test.SibSp+test.Parch
warnings.filterwarnings(action="ignore")

plt.figure(figsize=[3,2])

sns.barplot('Relatives','Survived',data=train)
train.drop(['PassengerId','Name','Ticket','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)

test.drop(['PassengerId','Name','Ticket','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)
titanic=pd.concat([train, test], sort=False)
titanic=pd.get_dummies(titanic)

train=titanic[:len_train]

test=titanic[len_train:]
# Lets change type of target

train.Survived=train.Survived.astype('int')

train.Survived.dtype
xtrain=train.drop("Survived",axis=1)

ytrain=train['Survived']

xtest=test.drop("Survived", axis=1)
RF=RandomForestClassifier(random_state=1)

PRF=[{'n_estimators':[10,100],'max_depth':[3,6],'criterion':['gini','entropy']}]

GSRF=GridSearchCV(estimator=RF, param_grid=PRF, scoring='accuracy',cv=2)

scores_rf=cross_val_score(GSRF,xtrain,ytrain,scoring='accuracy',cv=5)
np.mean(scores_rf)
svc=make_pipeline(StandardScaler(),SVC(random_state=1))

r=[0.0001,0.001,0.1,1,10,50,100]

PSVM=[{'svc__C':r, 'svc__kernel':['linear']},

      {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]

GSSVM=GridSearchCV(estimator=svc, param_grid=PSVM, scoring='accuracy', cv=2)

scores_svm=cross_val_score(GSSVM, xtrain.astype(float), ytrain,scoring='accuracy', cv=5)
np.mean(scores_svm)
model=GSSVM.fit(xtrain, ytrain)

pred=model.predict(xtest)

output=pd.DataFrame({'PassengerId':test2['PassengerId'],'Survived':pred})

output.to_csv('submission.csv', index=False)