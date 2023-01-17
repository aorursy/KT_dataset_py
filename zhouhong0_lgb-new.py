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
train=pd.read_csv("/kaggle/input/google-feature-engineering/concat_train.csv")

test=pd.read_csv("/kaggle/input/google-feature-engineering/test.csv")

sub = pd.read_csv("/kaggle/input/google-feature-engineering/submission.csv")
train=train.iloc[:-41,:]

train.isMobile = train.isMobile.fillna(-999).apply(lambda x: 1 if x== True else 0)

test.isMobile = test.isMobile.fillna(-999).apply(lambda x: 1 if x== True else 0)        
train.iloc[:,:-3] = train.iloc[:,:-3].fillna(-999)

test.iloc[:,:-1] = test.iloc[:,:-1].fillna(-999)
test.shape
train['return'].value_counts()
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
train.replace(-999, 0, inplace=True)

test.replace(-999, 0, inplace=True)
Xreturn = train.iloc[:,:-4]

yreturn = train['return']

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
top10=['visitNumber_max','timeOnSite_mean','visitStartTime_counts','hits_max','pageviews_sum',]
gnb.fit(Xreturn[top10],yreturn)
ytarget=np.log1p(train['target'][train['return']>0])

Xtarget = train.iloc[:,:-4][train['return']>0]
params_regression = {

        'learning_rate': 0.1,

        "objective" : "regression",

        "metric" : "rmse", 

        "max_leaves": 2,

        "num_leaves" : 9,

        "min_child_samples" : 1,

        "min_child_weight":0.01,

        "learning_rate" : 0.1,

        "bagging_fraction" : 0.5,

        "feature_fraction" : 0.6,

        "bagging_frequency" : 1      

    }
lgb2=lgb.LGBMRegressor(**params_regression) 

para2={'max_depth':[-1,1,2,3],'num_leaves':[2,3]}

#para2={'max_depth': [2,-1,7,5], 'num_leaves': [5,20,50,100]}

gs1 = GridSearchCV(lgb2,para2,cv = 4,scoring = 'neg_mean_squared_error',verbose=5)

gs1.fit(Xtarget,ytarget)

print(gs1.best_params_)

params_regression['max_depth']=gs1.best_params_['max_depth']

params_regression['num_leaves']=gs1.best_params_['num_leaves']



# {'max_depth': -1, 'num_leaves': 2}
lgb2=lgb.LGBMRegressor(**params_regression) 

para2={'min_child_samples':range(1,25),'min_child_weight':[0.001, 0.002]}

#para2={'max_depth':[5]}

gs1 = GridSearchCV(lgb2,para2,cv = 4,scoring = 'neg_mean_squared_error',verbose=5)

gs1.fit(Xtarget,ytarget)

print(gs1.best_params_)

params_regression['min_child_samples']=gs1.best_params_['min_child_samples']

params_regression['min_child_weight']=gs1.best_params_['min_child_weight']

#{'min_child_samples': 1, 'min_child_weight': 0.001}
lgb2=lgb.LGBMRegressor(**params_regression) 

#para2={'feature_fraction': [i/10.0 for i in range(1,11)],'bagging_fraction': [i/10.0 for i in range(1,11)]}

para2={'feature_fraction': [i/10.0 for i in range(5,11)],'bagging_fraction': [i/10.0 for i in range(5,11)]}

gs1 = GridSearchCV(lgb2,para2,cv = 4,scoring = 'neg_mean_squared_error',verbose=5)

gs1.fit(Xtarget,ytarget)

print(gs1.best_params_)

params_regression['bagging_fraction']=gs1.best_params_['bagging_fraction']

params_regression['feature_fraction']=gs1.best_params_['feature_fraction']

#{'bagging_fraction': 0.1, 'feature_fraction': 0.2}
params_regression['learning_rate']=0.01
X_train, X_test, y_train, y_test = train_test_split(Xtarget, np.log1p(ytarget), test_size=0.33, random_state=42)
lgb_train = lgb.Dataset(X_train, y_train)

lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
evals_result={}

a2 = lgb.train(params_regression, lgb_train,2000,valid_sets=[lgb_train, lgb_test], evals_result=evals_result,early_stopping_rounds=100)
a2.best_iteration
ax = lgb.plot_metric(evals_result, metric='rmse')

plt.show()
params_regression['num_iteration']=a2.best_iteration
lgb.plot_importance(a2,max_num_features=10)
sub['fullVisitorId'] = sub['fullVisitorId'].astype(str)

test['fullVisitorId'] = test['fullVisitorId'].astype(str)
lgb2=lgb.LGBMRegressor(**params_regression)
lgb2.fit(Xtarget,np.log1p(ytarget))


test.fullVisitorId=test.fullVisitorId.astype(np.double)

test = test.sort_values(by='fullVisitorId')

# make prediction

pre_return = gnb.predict(test[top10])

pre_revenue = lgb2.predict(test[test.columns[:-2]])



test["return"]=pre_return

test["pre_revenue"]=pre_revenue



sub.index = sub['fullVisitorId'].astype(np.double)

sub = sub.sort_index()

sub=sub.sort_index()



sub['PredictedLogRevenue']=pre_revenue * pre_return



sub.to_csv("lgb_with_naviebayes.csv",index = False)