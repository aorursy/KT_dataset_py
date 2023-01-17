import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
full = train.append(test,ignore_index=True)
full.shape
full.info()
full.Embarked.value_counts()
full.Age = full.Age.fillna(full.Age.mean())
full.Fare = full.Fare.fillna(full.Fare.mean())
full.Embarked = full.Embarked.fillna('S')
full.info()
full = full.drop('Cabin',axis=1)
full.info()
full.Sex = full.Sex.map({'male':1,'female':0})
full.head()
full.Embarked.head()
embarked = pd.DataFrame()
embarked = pd.get_dummies(full.Embarked,prefix='Embarked')
embarked.head()
pclass = pd.DataFrame()
pclass = pd.get_dummies(full.Pclass,prefix='Pclass')
pclass.head()
family = pd.DataFrame()
family['Family'] = full.Parch + full.SibSp + 1
family.head()
full.head()
full = pd.concat([full,embarked,pclass,family],axis=1)

full.head()
full = full.drop(['Embarked','Pclass','SibSp','Parch'],axis=1)

full.head()
corr = full.corr()
corr
corr.Survived.sort_values(ascending=False)
full.head()
full_X = pd.concat([full.Age,full.Sex,full.Fare,full.Family,embarked,pclass],axis=1)
full_X.shape
X = full_X.loc[0:890,:]
y = full.loc[0:890,'Survived']
X.shape
test_X = full_X.loc[891:,:]
test_X.shape
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
logreg.score(X_test,y_test)
test_X.head()
pred = logreg.predict(test_X)
pred = pred.astype(int)
ids = full.loc[891:,'PassengerId']
Pred = pd.DataFrame({'PassengerId':ids,'Survived':pred})
Pred.shape
Pred.head()
Pred.to_csv('titanic_pred.csv')
