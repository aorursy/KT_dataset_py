# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory







# Any results you write to the current directory are saved as output.
#importing Train and Test Data

train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

    
data=train.append(test,sort='False')
data.head(10)
data.describe()
data.info()
data.isnull().sum()
list=['Embarked','Fare','Parch','PassengerId','Pclass','Sex','SibSp']

for i in list:

    print('{:15} {}'.format(i,data[i].value_counts().shape[0]))
sns.FacetGrid(data,col='Survived').map(plt.hist,'Age',bins=20)
sns.FacetGrid(data,col='Survived',row='Pclass').map(plt.hist,'Age',bins=20)
sns.FacetGrid(data,row='Embarked').map(sns.pointplot,'Pclass','Survived','Sex')
sns.FacetGrid(data,col='Survived',row='Embarked').map(sns.barplot,'Sex','Fare',ci=None)
sns.factorplot(data=data,y='Survived',x='SibSp',kind='bar')
sns.factorplot(data=data,y='Survived',x='Parch',kind='bar')
sns.kdeplot(data['Age'][(data['Survived']==0) & (data['Age'].notnull())],color='Red',shade=True)

sns.kdeplot(data['Age'][(data['Survived']==1) & (data['Age'].notnull())],color='Blue',shade=True)
data.isnull().sum()
#1 fare

data['Fare'].fillna(data[(data['Pclass'] == 3) & (data['Age'] >= 60) & (data['Sex'] == 'male') & (data['Embarked'] == 'S')].Fare.mean(),inplace=True)

data['Fare'].isnull().sum()
cabin=pd.get_dummies(pd.cut(data['Fare'],bins=[0,8,15,32,520],labels=[0,1,2,3]))

dum=pd.DataFrame(data['Fare'])

dum
#2 Embarked

data['Embarked'] = data['Embarked'].fillna('C')

data['Embarked'].isnull().sum()
cabin=pd.get_dummies(data['Embarked'])

dum=pd.concat([dum,cabin],axis=1)

dum.head(5)
#3 name

data['sname'] = data['Name'].str.split(',').str[0].str.lower()

data['tname'] = data['Name'].str.split(',').str[1].str.split('.').str[0].str.lower().str.strip()
cabin1=pd.get_dummies(data['tname'])

dum=pd.concat([dum,cabin1],axis=1)

dum.head(5)
#4 sex

cabin2=pd.get_dummies(data['Sex'])

dum=pd.concat([dum,cabin2],axis=1)

dum.head(5)
#5 pclass 

cabin3=pd.get_dummies(data['Pclass'])

dum=pd.concat([dum,cabin3],axis=1)

dum.head(5)
#7 parch & sibsp

data['fsize']=data['SibSp']+data['Parch']+1

data['fsize'].describe()

data[data['sname']=='mcgowan']

pd.crosstab(data.sname,data.SibSp)
cabin4=pd.get_dummies(data['fsize'])

dum=pd.concat([dum,cabin4],axis=1)

dum.head(5)
#8 ticket

data['sticket']=data['Ticket'].str[0].str.lower()
cabin6=pd.get_dummies(data['sticket'])

dum=pd.concat([dum,cabin6],axis=1)

dum.head(5)
dum.shape
#9 cabin

cabin5=pd.get_dummies(data['Cabin'].str[0].str.lower().fillna(0))

dum=pd.concat([dum,cabin5],axis=1)

dum.head()
dum.columns = np.arange(61).astype('str')
dum.head()
train_x=dum[data['Age'].notnull()]

train_y=data['Age'][data['Age'].notnull()]

test_x=dum[data['Age'].isnull()]
# 11- Handling 'Age' features with XGBoost



from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor





best_para = {'base_score': 0.5, 'booster': 'dart', 

             'colsample_bylevel': 1, 'colsample_bytree': 0.6, 'gamma': 0.0, 

             'learning_rate': 0.1, 'max_delta_step': 0, 'max_depth': 3, 

             'min_child_weight': 7, 'missing': None, 'n_estimators': 55, 'n_jobs': 1, 

             'nthread': None, 'objective': 'reg:linear', 'random_state': 0, 'reg_alpha': 0.01, 'reg_lambda': 1, 'scale_pos_weight': 1, 'seed': None, 'silent': True, 

             'subsample': 0.8}

model_a = XGBRegressor(**best_para)

print(model_a)

mo=model_a.fit(train_x,train_y)

mo
mo1=mo.predict(test_x)

mo1

data['Age'][data['Age'].isnull()]=mo1
data['Age'].isnull().sum()
fig,ax=plt.subplots(figsize=(20,6))



sns.boxplot(data=data,x='Pclass',y='Age')

sns.swarmplot(data=data,x='Pclass',y='Age')

plt.show()
data['Age'].describe()
cabin15=pd.get_dummies(pd.cut(data['Age'],bins=[0,10,20,30,40,50,60,70,80,90,100],labels=[0,1,2,3,4,5,6,7,8,9]))

dum=pd.concat([dum,cabin15],axis=1)
dum.head(10)
ag=18

def pers(h):

    age,sex=h

    if age<=ag:

        return 'child'

    elif sex=='male':

        return 'male_adult'

    else:

        return 'female_adult'

    

dum.columns
data['person']=data[['Age','Sex']].apply(pers,axis=1)

data.head()

dum.columns=np.arange(71).astype('str')

dum.head()
# Building a model



from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score, KFold
train.test_num=data[data['Survived'].notnull()].shape[0]

train.test_num
x_train=dum[:train.test_num]

y_train=data['Survived'][:train.test_num]

x_test=dum[train.test_num:]
x_test
#decision tree

regg=DecisionTreeRegressor(random_state=43)

regg.fit(x_train,y_train)
#kfold

kf = KFold(n_splits=5, random_state=1, shuffle=False)

scores = cross_val_score(regg, x_train.values, y_train.values.ravel())



regg.fit(x_train.values, y_train.values.ravel())



score = regg.score(x_train.values, y_train.values.ravel())



print("Accuracy: {:.2f}+/-({:.2f}) {}".format(scores.mean() * 100, scores.std() * 100, ' Cross Validation'))

print("Accuracy: {:.3f}         {}".format(score * 100, ' full test'))
#rand forest

from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=10,random_state=43)
#kfold

kf = KFold(n_splits=5, random_state=1, shuffle=False)

scores = cross_val_score(rf, x_train.values, y_train.values.ravel())



rf.fit(x_train.values, y_train.values.ravel())



score = rf.score(x_train.values, y_train.values.ravel())



print("Accuracy: {:.2f}+/-({:.2f}) {}".format(scores.mean() * 100, scores.std() * 100, ' Cross Validation'))

print("Accuracy: {:.3f}         {}".format(score * 100, ' full test'))
#XGBoost

best_para = {'base_score': 0.5,

             'booster': 'dart',

             'colsample_bylevel': 1,

             'colsample_bytree': 0.6,

             'gamma': 0.3,

             'learning_rate': 0.1,

             'max_delta_step': 0,

             'max_depth': 6,

             'min_child_weight': 7,

             'missing': None,

             'n_estimators': 157,

             'n_jobs': 1,

             'nthread': None,

             'objective': 'binary:logistic',

             'random_state': 0,

             'reg_alpha': 1,

             'reg_lambda': 1,

             'scale_pos_weight': 1,

             'seed': None,

             'silent': True,

             'subsample': 0.9}
model = XGBClassifier(**best_para)





kf = KFold(n_splits=5, random_state=1, shuffle=False)

scores = cross_val_score(model, x_train.values, y_train.values.ravel())



model.fit(x_train.values, y_train.values.ravel())



score = model.score(x_train.values, y_train.values.ravel())



print("Accuracy: {:.2f}+/-({:.2f}) {}".format(scores.mean() * 100, scores.std() * 100, 'Cross Validation'))

print("Accuracy: {:.3f}         {}".format(score * 100, 'full test'))
pre_test = model.predict(x_test.values)



pre_Id = pd.DataFrame()

pre_Id['PassengerId'] = data['PassengerId'][train.test_num: ]

pre_Id.index = pre_Id.PassengerId



pre_Id['Survived'] = pre_test

pre_Id.drop('PassengerId', axis=1, inplace=True)



pre_Id['Survived'] = pre_Id['Survived'].astype('int')



pre_Id.to_csv('Titanic_xgb.csv')

pre_Id.head()
#LGBM

param_grid = {

    'num_leaves': [31],

    'feature_fraction': [0.5],

    'bagging_fraction': [0.95], 

    'reg_alpha': [0.1],

    'learning_rate': [0.1],

    'n_estimators': [50]}
from lightgbm import LGBMRegressor



param_grid = {

    'learning_rate': [0.1],

    'n_estimators': [50]

}


lgb_estimator = LGBMRegressor(boosting_type='gbdt',

                                  objective='regression',

                                  bagging_freq=5,

                                  eval_metric='l1',

                                  max_depth=5, 

                                  max_bin=225,

                                  min_split_gain=0, 

                                  min_child_weight=5, 

                                  min_child_samples=10, 

                                  subsample=1, 

                                  subsample_freq=1, 

                                  colsample_bytree=1,  

                                  reg_lambda=0, 

                                  seed=410) 



gsearch = GridSearchCV(estimator=lgb_estimator, 

                       param_grid=param_grid, 

                       cv=5) 



lgb_model = gsearch.fit(x_train, 

                        y_train)



print(lgb_model.best_params_, lgb_model.best_score_)



#kfold

kf = KFold(n_splits=5, random_state=1, shuffle=False)

scores = cross_val_score(lgb_estimator, x_train.values, y_train.values.ravel())



lgb_estimator.fit(x_train.values, y_train.values.ravel())



score = lgb_estimator.score(x_train.values, y_train.values.ravel())



print("Accuracy: {:.2f}+/-({:.2f}) {}".format(scores.mean() * 100, scores.std() * 100, 'Cross Validation'))

print("Accuracy: {:.3f}         {}".format(score * 100, 'full test'))
lgb=lgb_estimator.fit(train_x,train_y)

#predicting on test set

ypred2=lgb.predict(test_x)
lgbm = model.predict(x_test.values)



lg = pd.DataFrame()

lg['PassengerId'] = data['PassengerId'][train.test_num: ]

lg.index = lg.PassengerId



lg['Survived'] = lgbm

lg.drop('PassengerId', axis=1, inplace=True)



lg['Survived'] = lg['Survived'].astype('int')



pre_Id.to_csv('Titanic_lgbm.csv')

pre_Id.head()