import pandas as pd



df_train=pd.read_csv("../input/titanic/train.csv")

df_train
df_train.isnull().sum() #check Nan data
df_train['Embarked'].value_counts()
df_train['Embarked']=df_train['Embarked'].fillna('S')

df_train.isnull().sum() #check Nan data
df_train['Age']=df_train.groupby(['Pclass','Name'])['Age'].apply(lambda d: d.fillna(d.median()))

df_train['Age']=df_train['Age'].fillna(df_train['Age'].median())

df_train.isnull().sum() #check Nan data
df_train=pd.get_dummies(df_train, columns=["Pclass","Sex","Embarked"], sparse=True)

df_train
df_train=df_train.drop(['PassengerId','Ticket','Cabin','Name'],axis=1)

df_train['Family']=df_train['SibSp']+df_train['Parch']

df_train
X_train=df_train.drop('Survived',axis=1)

y_train=df_train['Survived']
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score



#make pipline

pipe_SVC=make_pipeline(StandardScaler(),SVC())



#make gridsearch

Crange=[1,1e1,5e1,8e1,9e1,1e2,1.1e2,1.2e2,1e3]

Grange=[8e-3,9e-3,1e-2,1.1e-2,1.2e-2,1e-1,1]

param_grid={'svc__C':Crange,'svc__gamma':Grange}

GS=GridSearchCV(estimator=pipe_SVC,param_grid=param_grid,scoring='accuracy',cv=10)

GS=GS.fit(X_train,y_train)



print(GS.best_score_)

print(GS.best_params_)
df_test=pd.read_csv('../input/titanic/test.csv')

df_test.isnull().sum()
df_test['Name']=df_test['Name'].str.extract(' ([A-Za-z]+).')

df_test['Age']=df_test.groupby(['Pclass','Name'])['Age'].apply(lambda d: d.fillna(d.median()))

df_test['Age']=df_test['Age'].fillna(df_test['Age'].median())

df_test.isnull().sum()
df_test['Fare']=df_test['Fare'].fillna(df_test['Fare'].median())

df_test.isnull().sum()
df_test=pd.get_dummies(df_test, columns=["Pclass","Sex","Embarked"], sparse=True)

df_test['Family']=df_test['SibSp']+df_test['Parch']

X_test=df_test.drop(['PassengerId','Ticket','Cabin','Name'],axis=1)
clf=GS.best_estimator_

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
df_pred=pd.DataFrame(df_test['PassengerId'])

df_pred['Survived']=y_pred
df_pred.to_csv('titanic_submission_svc.csv',index=False)