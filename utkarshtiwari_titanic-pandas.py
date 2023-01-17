# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings(action="ignore")



# Any results you write to the current directory are saved as output.
tra=pd.read_csv('../input/train.csv')

tes=pd.read_csv('../input/test.csv')
x=tra.drop(['Name','PassengerId','Ticket','Survived'],axis=1)

x_t=tes.drop(['Name','PassengerId','Ticket'],axis=1)

x.isna().sum()
x_t.isna().sum()
x.Age=x.Age.fillna(x.Age.mean())

x_t.Age=x_t.Age.fillna(x_t.Age.mean())
x.Cabin=x.Cabin.fillna('U')

x_t.Cabin=x_t.Cabin.fillna('U')
x.Embarked=x.Embarked.fillna('S')

x_t.Embarked=x_t.Embarked.fillna('S')
x_t.Fare=x_t.Fare.fillna(x_t.Fare.mean())
x.Cabin = x.Cabin.map(lambda z: z[0])

x_t.Cabin = x_t.Cabin.map(lambda z: z[0])
#x['Total Family']=x['SibSp']+x['Parch']

#x_t['Total Family']=x_t['SibSp']+x['Parch']
#x=x.drop(['Parch','SibSp'],axis=1)

#x_t=x_t.drop(['Parch','SibSp'],axis=1)
x
x_t
#x['Cabin'] = pd.Categorical(x['Cabin'])
#x['Embarked']=pd.Categorical(x['Embarked'])
x= pd.get_dummies(x)

x_t=pd.get_dummies(x_t)
x.head()
x_t.head()
x=x.drop(['Cabin_T'],axis=1)
y=tra['Survived']
y.head()
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.ensemble import RandomForestClassifier

reg=RandomForestClassifier(n_estimators=100000,random_state=0)

reg.fit(x_train,y_train)
#from sklearn.ensemble import GradientBoostingClassifier

#reg = GradientBoostingClassifier()

#reg.fit(x_train,y_train)
y_pred=reg.predict(x_val)
from sklearn.metrics import accuracy_score

accuracy_score(y_val, y_pred)
pred=reg.predict(x_t)

new_pred=pred.astype(int)

output=pd.DataFrame({'PassengerId':tes['PassengerId'],'Survived':new_pred})

output.to_csv('Titanic.csv', index=False)