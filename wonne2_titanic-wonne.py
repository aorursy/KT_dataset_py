# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # import matplotlib to draw graph

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv('../input/train.csv')

df_test=pd.read_csv('../input/test.csv')
df_train.describe()

df_train.info()

### Name Initial(Mr,Mrs,Miss,Master,Other)로 나누어 처리.



df_train['Initial']=0

df_train['Initial']= df_train.Name.str.extract('([A-Za-z]+)\.') 

df_train['Initial'].unique()
df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess',

                          'Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other',

                       'Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

df_train['Initial'].unique()

np.where(df_train['Initial'].isnull())



df_test['Initial']=0

df_test['Initial']= df_test.Name.str.extract('([A-Za-z]+)\.')

df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess',

                          'Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other',

                         'Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

df_test['Initial'].unique()

np.where(df_test['Initial']=='Master')
df_train.groupby('Initial').mean()
## Embarked(승선지)의 Null값이 2개이지만 대부분의 승객이 Southampton에서 탔음을 EDA를 통해 봤기때문에 S로 치환처리.

df_train['Embarked'].isnull().sum()

df_train['Embarked'].fillna('S', inplace=True)

df_test['Embarked'].fillna('S', inplace=True)





df_train['Embarked'].isnull().sum()

df_test['Embarked'].isnull().sum()
### 범주형 Data를 카테고리화 해줌. (범주형 -> 숫자로 변)

df_train['Initial']=df_train['Initial'].map({'Mr':0 , 'Miss':1 , 'Mrs':2 , 'Master':3, 'Other':4})

df_train.Initial.unique()

df_test['Initial']=df_test['Initial'].map({'Mr':0 , 'Miss':1 , 'Mrs':2 , 'Master':3, 'Other':4})



df_train['Sex']=df_train['Sex'].map({'male':0, 'female':1})

df_train.Sex.unique() 

df_test['Sex']=df_test['Sex'].map({'male':0, 'female':1})



df_train['Embarked']=df_train['Embarked'].map({'S':0, 'Q':1, 'C':2})

df_train.Embarked.unique() 

df_test['Embarked']=df_test['Embarked'].map({'S':0, 'Q':1, 'C':2})

df_test.Embarked.unique() 
### Fare의 Scewness가 한쪽으로 치우쳐져 있음을 알 수있었다.

### 그러므로 Scewness를 줄이기 위해 Log Scaling을 진행한다.

### sqrt로 진행했을때 Scewness가 별로 줄지 않음을 알 수 있었다.

#정규화. 의미가 없음

#df_train['Fare'] = (df_train['Fare']-min(df_train['Fare'])) / (max(df_train['Fare'])-min(df_train['Fare']))

#df_test['Fare'] = (df_test['Fare']-min(df_test['Fare'])) / (max(df_test['Fare'])-min(df_test['Fare']))



# Scewness를 더 줄이기 위해 log를 두번취해줌.

df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i>0 else 0)

df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i>0 else 0)

df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i>0 else 0)

df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i>0 else 0)



# histogram

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = sns.distplot(df_train['Fare'], color='b', 

                 label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')





### 분석에 사용될 수 없는 Data를 삭제해줌.

df_train=df_train.drop(['Name','Ticket','Cabin'],axis=1)

df_test=df_test.drop(['Name','Ticket','Cabin'],axis=1)
##Age의 Null값을 initial로 평균내 채워준 데이터 분석용 df

df_fillage=df_train

df_fillagetest=df_test
df_fillage['FamilySize']=df_fillage['SibSp']+df_fillage['Parch']

df_fillagetest['FamilySize']=df_fillagetest['SibSp']+df_fillagetest['Parch']



df_fillage=df_fillage.drop(['SibSp','Parch'],axis=1)

df_fillagetest=df_fillagetest.drop(['SibSp','Parch'],axis=1)
df_fillage.loc[(df_fillage.Age.isnull())&(df_fillage.Initial==0),'Age']=33

df_fillage.loc[(df_fillage.Age.isnull())&(df_fillage.Initial==1),'Age']=22

df_fillage.loc[(df_fillage.Age.isnull())&(df_fillage.Initial==2),'Age']=36

df_fillage.loc[(df_fillage.Age.isnull())&(df_fillage.Initial==3),'Age']=5

df_fillage.loc[(df_fillage.Age.isnull())&(df_fillage.Initial==4),'Age']=46



df_fillagetest.loc[(df_fillagetest.Age.isnull())&(df_fillagetest.Initial==0),'Age']=33

df_fillagetest.loc[(df_fillagetest.Age.isnull())&(df_fillagetest.Initial==1),'Age']=22

df_fillagetest.loc[(df_fillagetest.Age.isnull())&(df_fillagetest.Initial==2),'Age']=36

df_fillagetest.loc[(df_fillagetest.Age.isnull())&(df_fillagetest.Initial==3),'Age']=5

df_fillagetest.loc[(df_fillagetest.Age.isnull())&(df_fillagetest.Initial==4),'Age']=46
df_fillage['Age_cat'] = 0

df_fillage.loc[df_fillage['Age'] < 10, 'Age_cat'] = 0

df_fillage.loc[(10 <= df_fillage['Age']) & (df_fillage['Age'] < 20), 'Age_cat'] = 1

df_fillage.loc[(20 <= df_fillage['Age']) & (df_fillage['Age'] < 30), 'Age_cat'] = 2

df_fillage.loc[(30 <= df_fillage['Age']) & (df_fillage['Age'] < 40), 'Age_cat'] = 3

df_fillage.loc[(40 <= df_fillage['Age']) & (df_fillage['Age'] < 50), 'Age_cat'] = 4

df_fillage.loc[(50 <= df_fillage['Age']) & (df_fillage['Age'] < 60), 'Age_cat'] = 5

df_fillage.loc[(60 <= df_fillage['Age']) & (df_fillage['Age'] < 70), 'Age_cat'] = 6

df_fillage.loc[70 <= df_fillage['Age'], 'Age_cat'] = 7





df_fillagetest['Age_cat'] = 0

df_fillagetest.loc[df_fillagetest['Age'] < 10, 'Age_cat'] = 0

df_fillagetest.loc[(10 <= df_fillagetest['Age']) & (df_fillagetest['Age'] < 20), 'Age_cat'] = 1

df_fillagetest.loc[(20 <= df_fillagetest['Age']) & (df_fillagetest['Age'] < 30), 'Age_cat'] = 2

df_fillagetest.loc[(30 <= df_fillagetest['Age']) & (df_fillagetest['Age'] < 40), 'Age_cat'] = 3

df_fillagetest.loc[(40 <= df_fillagetest['Age']) & (df_fillagetest['Age'] < 50), 'Age_cat'] = 4

df_fillagetest.loc[(50 <= df_fillagetest['Age']) & (df_fillagetest['Age'] < 60), 'Age_cat'] = 5

df_fillagetest.loc[(60 <= df_fillagetest['Age']) & (df_fillagetest['Age'] < 70), 'Age_cat'] = 6

df_fillagetest.loc[70 <= df_fillagetest['Age'], 'Age_cat'] = 7
#df_fillage = pd.get_dummies(df_fillage, columns=['Initial'], prefix='Initial')

#df_fillagetest = pd.get_dummies(df_fillagetest, columns=['Initial'], prefix='Initial')



#df_fillage = pd.get_dummies(df_fillage, columns=['Age_cat'], prefix='Age')

#df_fillagetest = pd.get_dummies(df_fillagetest, columns=['Age_cat'], prefix='Age')



df_fillage = pd.get_dummies(df_fillage, columns=['Pclass'], prefix='Pclass')

df_fillagetest = pd.get_dummies(df_fillagetest, columns=['Pclass'], prefix='Pclass')  



df_fillage = pd.get_dummies(df_fillage, columns=['Sex'], prefix='Sex')

df_fillagetest = pd.get_dummies(df_fillagetest, columns=['Sex'], prefix='Sex')



df_fillage = pd.get_dummies(df_fillage, columns=['Embarked'], prefix='Embarked')

df_fillagetest = pd.get_dummies(df_fillagetest, columns=['Embarked'], prefix='Embarked')
# 필요없는 feature 제거

df_fillage=df_fillage.drop(['PassengerId'],axis=1)

df_fillage.info()

df_fillagetest=df_fillagetest.drop(['PassengerId'],axis=1)



df_fillage=df_fillage.drop(['Age'],axis=1)

df_fillage.info()

df_fillagetest=df_fillagetest.drop(['Age'],axis=1)



df_fillage=df_fillage.drop(['Initial'],axis=1)

df_fillage.info()

df_fillagetest=df_fillagetest.drop(['Initial'],axis=1)
############################## RFC 분석

import sklearn

from sklearn.ensemble import RandomForestClassifier  

from sklearn import metrics 

from sklearn.model_selection import train_test_split



X_train = df_fillage.drop('Survived', axis=1).values

target_label = df_fillage['Survived'].values

X_test = df_fillagetest.values

X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=10)



model=RandomForestClassifier(max_features=6, min_samples_leaf=3, n_estimators= 100, n_jobs= 2)

model.fit(X_tr,y_tr)

prediction = model.predict(X_vld)

prediction

print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))

(prediction == y_vld).sum()/prediction.shape[0]
######################################

import xgboost as xgb



model_xgb = xgb.XGBClassifier(max_depth=9, learning_rate=0.01, n_estimators=500, reg_alpah=1.1,

                             colsample_bytree=0.9, subsample=0.9, n_jobs=5)

model_xgb.fit(X_tr, y_tr, eval_set=[(X_vld, y_vld)], verbose=False, early_stopping_rounds=50)

pred_xgb = model_xgb.predict(X_vld)

score_xgb = metrics.accuracy_score(pred_xgb, y_vld)

print("XGBoost Test score: ", score_xgb)
#####################################

import lightgbm as lgbm

from sklearn.model_selection import GridSearchCV

# 파라미터 조합

# Grid Search를 이용한 Hyper Parameter 선정과정.

'''

param_grid = [

    {'max_depth':[6,7,8,9,10], 'lambda_l1':[0.1], 'lambda_l2':[0.01], 'learning_rate':[0.01],

                               'n_estimators':[500,1000], 'reg_alpha':[1.1], 'colsample_bytree':[0.8,0.9], 'subsample':[0.8,0.9], 'n_jobs':[3,5,7,9]}

]

lgbmcl = lgbm.LGBMClassifier()

# grid search

grid_search = GridSearchCV(lgbmcl, param_grid, cv=5,

                          scoring='accuracy',

                          return_train_score=True)

grid_search.fit(X_tr, y_tr)

grid_search.best_params_

'''



model_lgbm = lgbm.LGBMClassifier(max_depth=6, lambda_l1=0.1, lambda_l2=0.01, learning_rate=0.01,

                               n_estimators=500, reg_alpha=1.1, colsample_bytree=0.9, subsample=0.8, n_jobs=3)

model_lgbm.fit(X_tr, y_tr, eval_set=[(X_vld, y_vld)], verbose=False, early_stopping_rounds=50,

              eval_metric="accuracy")

pred_lgbm = model_lgbm.predict(X_vld)

score_lgbm = metrics.accuracy_score(pred_lgbm, y_vld)

print("LightGBM Test Score: ", score_lgbm)
#############################

import catboost as cboost



model_cboost = cboost.CatBoostClassifier(depth=9, reg_lambda=0.1, learning_rate=0.01, iterations=500)

model_cboost.fit(X_tr, y_tr, eval_set=[(X_vld, y_vld)], verbose=False, early_stopping_rounds=50)

pred_cboost = model_cboost.predict(X_vld)

score_cboost = metrics.accuracy_score(pred_cboost, y_vld)

print("CatBoost Test Score: ", score_cboost)
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization, Activation

from keras.optimizers import Adam

from keras import backend as K



model_mlp = Sequential()

model_mlp.add(Dense(48 ,activation='relu', input_dim=12))

model_mlp.add(Dense(24,activation='relu'))

model_mlp.add(Dense(12,activation='relu'))

model_mlp.add(Dense(1,activation='sigmoid'))



optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)

model_mlp.compile(optimizer=optimizer, 

            loss='binary_crossentropy', 

            metrics=['accuracy'])



hist = model_mlp.fit(X_tr, y_tr, epochs=500, batch_size=30, validation_data=(X_vld,y_vld), verbose=False)

 

pred_mlp = model_mlp.predict_classes(X_vld)[:,0]

score_mlp = metrics.accuracy_score(pred_mlp, y_vld)

print("MLP Test Score: ", score_mlp)

submission = pd.read_csv('../input/sample_submission.csv')

prediction = model_lgbm.predict(X_test)

submission['Survived'] = prediction
submission.to_csv('../input/titanic_submission_Wonne.csv', index=False)
