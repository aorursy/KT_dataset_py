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
train=pd.read_csv("/kaggle/input/dataset-for-modeling/concat_train.csv")

test=pd.read_csv("/kaggle/input/dataset-for-modeling/test.csv")

sub = pd.read_csv("/kaggle/input/dataset-for-modeling/submission.csv")
test.columns
test.shape
train.isMobile = train.isMobile.fillna(-999).apply(lambda x: 1 if x== True else 0)              

train.isMobile=train.isMobile*1

test.isMobile = train.isMobile.fillna(-999).apply(lambda x: 1 if x== True else 0)              

test.isMobile=train.isMobile*1
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

import xgboost as xgb

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
train.iloc[:,:-4] = train.iloc[:,:-4].fillna(-999)
Xreturn = train.iloc[:,:-4]

yreturn = train['return']

from sklearn.metrics import f1_score
import sklearn

sklearn.metrics.SCORERS.keys()
params_regression={"objective":"reg:linear",

            "booster" :"gbtree",

            "eval_metric" :"rmse",

            "nthread" : 4,

            "eta" :0.075,

            "max_depth" :7,

            "min_child_weight": 5,

            "gamma" :0.05,

            "subsample": 0.8,

            "colsample_bytree" :0.7,

            "colsample_bylevel" : 0.6,

            "alpha" : 0,

            "lambda" :5}

  
ytarget=np.log1p(train['target'][train['return']>0])

Xtarget = train.iloc[:,:-4][train['return']>0]
xgb1 = xgb.XGBRegressor(**params_regression)

para1={'max_depth': range(3,11), 'min_child_weight': range(1,7,2)}

#para1={'max_depth': range(3,4), 'min_child_weight': range(1,2)}

gs1 = GridSearchCV(xgb1,para1,cv = 5,scoring = 'neg_mean_squared_error',verbose=5)

gs1.fit(Xtarget,ytarget)

print(gs1.best_params_)

params_regression['max_depth']=gs1.best_params_['max_depth']

params_regression['min_child_weight']=gs1.best_params_['min_child_weight']

xgb1 = xgb.XGBRegressor(**params_regression)

para1={'gamma':[i/100.0 for i in range(1,11)]}

gs1 = GridSearchCV(xgb1,para1,cv = 5,scoring = 'neg_mean_squared_error',verbose=5)

gs1.fit(Xtarget,ytarget)

print(gs1.best_params_)

params_regression['gamma']=gs1.best_params_['gamma']

xgb1 = xgb.XGBRegressor(**params_regression)

para1={'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}

gs1 = GridSearchCV(xgb1,para1,cv = 5,scoring = 'neg_mean_squared_error',verbose=5)

gs1.fit(Xtarget,ytarget)

print(gs1.best_params_)

params_regression['subsample']=gs1.best_params_['subsample']

params_regression['colsample_bytree']=gs1.best_params_['colsample_bytree']
xgb1 = xgb.XGBRegressor(**params_regression)

para1={'alpha':[i/10.0 for i in range(0,11)]}

gs1 = GridSearchCV(xgb1,para1,cv = 5,scoring = 'neg_mean_squared_error',verbose=5)

gs1.fit(Xtarget,ytarget)

print(gs1.best_params_)

params_regression['alpha']=gs1.best_params_['alpha']
clf=xgb.XGBRegressor(**params_regression)

X_train, X_test, y_train, y_test = train_test_split(Xtarget, np.log1p(ytarget), test_size=0.33, random_state=42)

clf.fit(X_train, y_train,

        eval_set=[(X_train, y_train), (X_test, y_test)],

        eval_metric='rmse',

        verbose=True)



evals_result = clf.evals_result()
ax = lgb.plot_metric(evals_result, metric='rmse')

plt.show()
params_return={'learning_rate': 0.1,

 'boosting_type': 'gbdt',

 "metric" : "auc",      

 'max_depth': -1,

 'min_child_weight': 0.001,

 'min_child_samples': 119,

 'reg_alpha': 0,

 'reg_lambda': 1,

 'subsample': 1,

 'colsample_bytree': 1,

 'feature_fraction': 0.5,

 'n_job': 4,

 'random_state':42}
lgb1=lgb.LGBMClassifier(**params_return) 

xgb2=xgb.XGBRegressor(**params_regression)
sub['fullVisitorId'] = sub['fullVisitorId'].astype(str)

test['fullVisitorId'] = test['fullVisitorId'].astype(str)
lgb1.fit(Xreturn,yreturn)

xgb2.fit(Xtarget,np.log1p(ytarget))
test.fullVisitorId=test.fullVisitorId.astype(np.double)

test = test.sort_values(by='fullVisitorId')
test.shape


test.fullVisitorId=test.fullVisitorId.astype(np.double)

test = test.sort_values(by='fullVisitorId')

# make prediction

pre_return = lgb1.predict(test[test.columns[:-2]])

pre_revenue = xgb2.predict(test[test.columns[:-2]])



test["return"]=pre_return

test["pre_revenue"]=pre_revenue



sub.index = sub['fullVisitorId'].astype(np.double)

sub = sub.sort_index()

sub=sub.sort_index()



test["target"]=test["return"]*test["pre_revenue"]

sub['PredictedLogRevenue']=pre_revenue



sub.to_csv("lgb.csv",index = False)