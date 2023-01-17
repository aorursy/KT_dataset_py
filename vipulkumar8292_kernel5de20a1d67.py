import numpy as np

import pandas as pd

import os

titanic = pd.read_csv('../input/titanic-survival-data/titanic_data.csv')
titanic.head()
titanic.rename(columns={'SibSp':'Siblings'},inplace=True)
titanic.drop(['PassengerId','Ticket','Cabin'], axis=1,inplace=True)
titanic.isnull().sum()
titanic.dropna(inplace=True)
titanic.isnull().sum()
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
df = titanic[['Sex','Embarked']]
titanic[['Sex','Embarked']]= df.apply(label_enc.fit_transform)
titanic.head()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
l = np.array(titanic['Survived']).reshape(-1,1)
l.astype(int)
y = l
x = titanic[['Pclass','Sex','Age','Siblings','Parch','Fare','Embarked']]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=23)
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(x_train,y_train)
rfc.score(x_test,y_test)