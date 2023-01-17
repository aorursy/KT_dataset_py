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
import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



from sklearn.ensemble import GradientBoostingClassifier



import lightgbm as lgb

from lightgbm import LGBMClassifier



import matplotlib.pyplot as plt

import requests

import lightgbm as lgb

import optuna

from functools import partial

from sklearn.model_selection import GroupKFold

from sklearn.metrics import mean_squared_log_error
import os

path='/kaggle/input/exam-for-students20200527'

df_train=pd.read_csv(os.path.join(path,'train.csv'))

df_test=pd.read_csv(os.path.join(path,'test.csv'))

df_city=pd.read_csv(os.path.join(path,'city_info.csv'))

df_station=pd.read_csv(os.path.join(path,'station_info.csv'))

sample_submission=pd.read_csv(os.path.join(path,'sample_submission.csv'))
df_train.loc[df_train.TimeToNearestStation=='2H-','MaxTimeToNearestStation']=180

df_test.loc[df_test.TimeToNearestStation=='2H-','MaxTimeToNearestStation']=180

df_train = df_train.rename(columns={'NearestStation': 'Station'})

df_test = df_test.rename(columns={'NearestStation': 'Station'})

df_station = df_station.rename(columns={'Latitude': 'Station_Latitude'})

df_station = df_station.rename(columns={'Longitude': 'Station_Longitude'})

df_train = pd.merge(df_train, df_station, on='Station', how='inner')

df_test = pd.merge(df_test, df_station, on='Station', how='inner')

df_city = df_city.rename(columns={'Latitude': 'city_Latitude'})

df_city = df_city.rename(columns={'Longitude': 'city_Longitude'})

df_train = pd.merge(df_train, df_city, on=['Prefecture','Municipality'], how='inner')

df_test = pd.merge(df_test, df_city, on=['Prefecture','Municipality'], how='inner')
havenullcol = df_train.columns[df_train.isnull().sum()!=0].values

havenullcol 

for col in havenullcol:

    df_train[col+'isnull'] = np.where(df_train[col].isnull(),1,0)

    df_test[col+'isnull'] = np.where(df_test[col].isnull(),1,0)
y_train = df_train.TradePrice

X_train = df_train.drop(['TradePrice'], axis=1)



X_test = df_test
floats = []

for col in df_train.columns:

    if df_train[col].dtype == 'float64':

        floats.append(col)

        

        print(col, df_train[col].nunique())
X_train[floats].fillna('1', inplace=True)

X_test[floats].fillna('1', inplace=True)
# dtypeがobjectのカラム名とユニーク数を確認してみましょう。

cats = []

for col in X_train.columns:

    if (X_train[col].dtype == 'object')  :

        cats.append(col)

        

        print(col, X_train[col].nunique())
target = 'TradePrice'

X_temp = pd.concat([X_train, y_train], axis=1)





for col in cats:

    # X_testはX_trainでエンコーディングする

    summary = X_temp.groupby([col])[target].mean()

    enc_test1 = X_test[col].map(summary) 





    # X_trainのカテゴリ変数をoofでエンコーディングする

    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





    enc_train1 = Series(np.zeros(len(X_train)), index=X_train.index)



    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([col])[target].mean()

        enc_train1.iloc[val_ix] = X_val[col].map(summary)



    X_train[col] = enc_train1

    X_test[col] = enc_test1
#LGBM固定パラメータ設定

def init_lgb_params():

    params = {

            'n_jobs':1,

            'task': 'train',

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': 'rmse',      

            'learning_rate': 0.01,

            'n_estimators': 500,

            'random_state':0

     }

    return params
#Optunaモデルlogloss評価

def opt_objective_logloss_groupkfold(X,y,groups,params,trial):



    params_tuning = {

            'num_leaves':trial.suggest_int('num_leaves',10,200),

            'max_depth':trial.suggest_int('max_depth',3,15),        

            'reg_alpha': trial.suggest_uniform('reg_alpha',0, 10),

            'reg_lambda': trial.suggest_uniform('reg_lambda',0, 10),             

            'subsample': trial.suggest_uniform('subsample',0.5, 1.0),  

            'colsample_bytree': trial.suggest_uniform('colsample_bytree',0.1, 1.0)         

    }



    params.update(params_tuning)



    

    gkf = GroupKFold(n_splits = 5)

    splitter = gkf.split(X,y,groups)





    for train_index, test_index in splitter:

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        y_train, y_test = y.iloc[train_index], y.iloc[test_index]



    model=lgb.LGBMRegressor(**params)

    model.fit(X_train,y_train,eval_set=[(X_test,y_test)],eval_metric="rmse",early_stopping_rounds=10,verbose=0)



    y_pred=model.predict(X_test)

    

    labels=list(sorted(set(y_test)))

        

    RMSLE=np.sqrt(mean_squared_log_error(y_test, np.abs(y_pred)))

    

    return RMSLE
#LGBMパラメータチューニング

def HyperParamTuning(train, key, target, group, ntrial, ncpu):

#    print(train.columns)

    X=train.drop([key,target,group],axis=1)

    y=train[target]

    groups=train[group]



    params=init_lgb_params()

    f=partial(opt_objective_logloss_groupkfold, X, y, groups, params)

    study=optuna.create_study()

    study.optimize(f,n_trials=ntrial,n_jobs=ncpu)

    params.update(study.best_params)

    

    model=lgb.LGBMRegressor(**params)

    

    return model,params
def Prediction(model,train,test,group,key,target):

    X_train=train.drop([key,target,group],axis=1)

    y_train=train[target]    

    X_test=test.drop([key,target,group],axis=1)

    

    model=model.fit(X_train,y_train,eval_metric='rmse')



    y_train_pred=model.predict(X_train)    

    y_test=model.predict(X_test)

    

    RMSLE=np.sqrt(mean_squared_log_error(y_train, np.abs(y_train_pred)))

    

    print('RMSLE(train)=',RMSLE)

    

    return y_test
X_train.fillna(1, inplace=True)

X_test.fillna(1, inplace=True)
serial_id='id'

target='TradePrice'

group='Station'
train = X_train

train['TradePrice'] = y_train

train =train.drop(['Prefecture','Municipality','DistrictName'],axis=1)
model,params=HyperParamTuning(train, serial_id, target, group, 30, 5)
test = X_test

test =test.drop(['Prefecture','Municipality','DistrictName'],axis=1)

test['TradePrice'] = 0

y_pred=Prediction(model, train, test, group, serial_id, target)
train.columns
submission=sample_submission.copy(deep=True)

submission['TradePrice']=np.abs(y_pred)

submission.to_csv('submision.csv',index=False)
submission.describe()