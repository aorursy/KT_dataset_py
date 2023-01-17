# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,VotingClassifier

import xgboost as xgb

from sklearn.preprocessing import OneHotEncoder, LabelEncoder



from sklearn.linear_model import LassoCV

from sklearn.feature_selection import RFE

from sklearn.svm import SVC,LinearSVC
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

train.head()
print("Train data shape:",train.shape)

print("Test data shape",test.shape)
train.info()
test_id=test['PassengerId']

df=pd.concat([train,test],axis=0)

df.head()

print(df.info())
df=df.drop(['PassengerId','Cabin','Ticket'],axis=1)

df['Age'].fillna(df['Age'].median(),inplace=True)

df['Fare'].fillna(df['Fare'].median(),inplace=True)

df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
df['Familysize']=df['SibSp']+df['Parch']

df['IsAlone']=1

df['IsAlone'].loc[df['Familysize']>=1]=0



df['FareBin']=pd.cut(df['Fare'],4)

df['AgeBin']=pd.cut(df['Age'].astype(int),5)



df['Title']=df['Name'].str.split(",",expand=True)[1].str.split('.',expand=True)[0]

min=10

title_names = df['Title'].value_counts() < min

df['Title']=df['Title'].apply(lambda x: 'Misc' if title_names[x]==True else x)



df.head()
label=LabelEncoder()

df['Sex_Code']=label.fit_transform(df['Sex'])

df['Embarked_Code']=label.fit_transform(df['Embarked'])

df['Title_Code']=label.fit_transform(df['Title'])

df['FareBin_Code']=label.fit_transform(df['FareBin'])

df['Age_Code']=label.fit_transform(df['AgeBin'])



df.head()
df=df.drop(['Name'],axis=1)

df=pd.get_dummies(df)

df.head()
ctest=df[df.Survived.isna()]

ctest=ctest.drop(['Survived'],axis=1)

ctrain=df[df.Survived.notna()]

print(ctrain.shape)

print(ctest.shape)
cX=ctrain.drop(['Survived'],axis=1)

cy=ctrain[['Survived']]
cX_train, cX_test, cy_train, cy_test = train_test_split(cX,cy,stratify=cy,test_size=0.2,random_state=1)
lcv=LassoCV()

lcv.fit(cX_train,cy_train)

lcv_mask=lcv.coef_!=0

print(sum(lcv_mask))
rfe_rf=RFE(estimator=RandomForestClassifier(),n_features_to_select=12,verbose=1)

rfe_rf.fit(cX_train,cy_train)

rf_mask=rfe_rf.support_
rfe_gb=RFE(estimator=GradientBoostingClassifier(),n_features_to_select=12,verbose=1)

rfe_gb.fit(cX_train,cy_train)

gb_mask=rfe_gb.support_
votes=np.sum([lcv_mask,rf_mask,gb_mask],axis=0)

print(votes)

mask=votes>1
print(cX_train.shape)

ccX_train=cX_train.loc[:,mask]

print(ccX_train.shape)



print(cX_test.shape)

ccX_test=cX_test.loc[:,mask]

print(ccX_test.shape)



print(ctest.shape)

cctest=ctest.loc[:,mask]

print(cctest.shape)

ccX_train.head()
lr=LogisticRegression()

lr.fit(ccX_train,cy_train)

y_pred=lr.predict(ccX_test)

print(accuracy_score(cy_test,y_pred))
steps=[('scaler',StandardScaler()),('lr',LogisticRegression())]

lr_pipe=Pipeline(steps)

lr_pipe.fit(ccX_train,cy_train)

y_pred=lr_pipe.predict(ccX_test)

print(accuracy_score(cy_test,y_pred))
knn=KNeighborsClassifier(n_neighbors=9)

knn.fit(cX_train,cy_train)

y_pred=knn.predict(cX_test)

print(accuracy_score(cy_test,y_pred))
param={'knn__n_neighbors':np.arange(1,20)}

steps=[('scaler',StandardScaler()),('knn',KNeighborsClassifier())]

knn_pipe=Pipeline(steps)

grid_knn=GridSearchCV(estimator=knn_pipe,param_grid=param,cv=10,n_jobs=-1)

grid_knn.fit(ccX_train,cy_train)

y_pred=grid_knn.predict(ccX_test)

print(accuracy_score(cy_test,y_pred))

print(grid_knn.best_estimator_)
param={'max_depth':np.arange(3,8),'min_samples_leaf':[0.04,0.06,0.08],'max_features':[0.2,0.4,0.6,0.8]}

dt=DecisionTreeClassifier(random_state=12)

grid_dt=GridSearchCV(estimator=dt,param_grid=param,cv=10,n_jobs=-1)

grid_dt.fit(ccX_train,cy_train)

y_pred=grid_dt.predict(ccX_test)

print(accuracy_score(cy_test,y_pred))

print(grid_dt.best_params_)
param={'n_estimators':[200],'max_depth':np.arange(3,6),'min_samples_leaf':[0.04,0.06,0.08],'max_features':[0.2,0.4,0.6,0.8]}

rf=RandomForestClassifier(random_state=12)

grid_rf=GridSearchCV(estimator=rf,param_grid=param,cv=10,n_jobs=-1)

grid_rf.fit(ccX_train,cy_train)

y_pred=grid_rf.predict(ccX_test)

print(accuracy_score(cy_test,y_pred))

print(grid_rf.best_params_)
xg_cl=xgb.XGBClassifier(objective='binary:logistic',n_estimators=4,seed=123)

xg_cl.fit(ccX_train,cy_train)

y_pred=xg_cl.predict(ccX_test)

print(accuracy_score(cy_test,y_pred))
xg=xgb.XGBClassifier(objective='reg:logistic',seed=123)

params={'n_estimators':[100,200],'max_depth':np.arange(2,6),'alpha':[0.01,0.1,1,10]}

grid_xg=GridSearchCV(estimator=xg,param_grid=params,cv=10,n_jobs=-1)

grid_xg.fit(ccX_train,cy_train)

y_pred=grid_xg.predict(ccX_test)

print(accuracy_score(cy_test,y_pred))

print(grid_xg.best_params_)
dt=DecisionTreeClassifier(max_depth=1,random_state=1)

ada=AdaBoostClassifier(base_estimator=dt,n_estimators=300,learning_rate=0.05)

ada.fit(ccX_train,cy_train)

y_pred=ada.predict(ccX_test)

print(accuracy_score(cy_test,y_pred))


grad=GradientBoostingClassifier(n_estimators=500,learning_rate=0.01)

grad.fit(ccX_train,cy_train)

y_pred=grad.predict(ccX_test)

print(accuracy_score(cy_test,y_pred))
svc=SVC(C=100,random_state=12)

svc.fit(ccX_train,cy_train)

y_pred=svc.predict(ccX_test)

print(accuracy_score(cy_test,y_pred))
lr=LogisticRegression(random_state=12)

knn=KNeighborsClassifier()

dt=DecisionTreeClassifier(random_state=12)

classifiers=[('Logistic',lr_pipe),

            ('knn',grid_knn),

            ('dt',grid_dt),

            ('gradient',grad),

            ('RF',grid_rf),

            ('Ada',ada),

            ('XGb',xg_cl),

            ('XgbGrid',grid_xg)]

vc=VotingClassifier(estimators=classifiers)

vc.fit(ccX_train,cy_train)

y_pred=vc.predict(ccX_test)

print(accuracy_score(cy_test,y_pred))
#test_1=test.drop(['PassengerId'],axis=1)

test_1=cctest

test_2=pd.DataFrame(test_1,columns=test_1.columns)

ans=vc.predict(test_2)



sub=pd.DataFrame({

    'PassengerId':test_id.astype(int),

    'Survived':ans.astype(int)

})

print(sub.head())

sub.to_csv('submissions.csv',index=False)