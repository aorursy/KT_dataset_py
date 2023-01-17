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

test.iloc[:,:-2] = test.iloc[:,:-2].fillna(-999)

train.replace(-999, 0, inplace=True)

test.replace(-999, 0, inplace=True)

test.fullVisitorId=test.fullVisitorId.astype(np.double)

test = test.sort_values(by='fullVisitorId')
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

import xgboost as xgb

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
Xreturn = train.iloc[:,:-4]

yreturn = train['return']

top5=['visitNumber_max','timeOnSite_mean','visitStartTime_counts','hits_max','pageviews_sum']
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(Xreturn[top5],yreturn)
xgbbest=xgb.XGBRegressor(alpha=0.0, base_score=0.5, booster='gbtree', colsample_bylevel=0.6,

             colsample_bynode=1, colsample_bytree=0.6, eta=0.075,

             eval_metric='rmse', gamma=0.01, importance_type='gain', 

             learning_rate=0.1, max_delta_step=0, max_depth=3,

             min_child_weight=3, missing=None, n_estimators=100, n_jobs=1,

             nthread=4, objective='reg:linear', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,

             subsample=0.7, verbosity=1)
lgbbest=lgb.LGBMRegressor(bagging_fraction=0.5, bagging_frequency=1, boosting_type='gbdt',

              class_weight=None, colsample_bytree=1.0, feature_fraction=0.6,

              importance_type='split', learning_rate=0.01, max_depth=-1,

              max_leaves=2, metric='rmse', min_child_samples=1,

              min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,

              n_jobs=-1, num_iteration=650, num_leaves=2,

              objective='regression', random_state=None, reg_alpha=0.0,

              reg_lambda=0.0, silent=True, subsample=1.0,

              subsample_for_bin=200000, subsample_freq=0)
from sklearn.ensemble import VotingRegressor

softvoting = VotingRegressor(estimators=[('xgb', xgbbest),('lgb', lgbbest)])
ytarget=np.log1p(train['target'][train['return']>0])

Xtarget = train.iloc[:,:-4][train['return']>0]
from vecstack import stacking

models = [lgbbest,xgbbest]

S_train, S_test = stacking(models,Xtarget,np.log1p(ytarget), test[test.columns[:-2]],regression=True,

                           mode='oof_pred_bag',needs_proba=False,save_dir=None,metric=mean_squared_error,

                           n_folds=4,stratified=True,shuffle=True,random_state=0,verbose=2)
stacking = xgb.XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, 

                      n_estimators=10, max_depth=3)

    

stacking = stacking.fit(S_train, np.log1p(ytarget))

xgbbest.fit(Xtarget,np.log1p(ytarget))
lgbbest.fit(Xtarget,np.log1p(ytarget))
softvoting.fit(Xtarget,np.log1p(ytarget))
sub['fullVisitorId'] = sub['fullVisitorId'].astype(str)

test['fullVisitorId'] = test['fullVisitorId'].astype(str)


# make prediction

pre_return = gnb.predict(test[top5])

pre_revenuesoftvoting = softvoting.predict(test[test.columns[:-2]])

pre_revenuelgb = lgbbest.predict(test[test.columns[:-2]])

pre_revenuexgb = xgbbest.predict(test[test.columns[:-2]])

pre_revenuestacking = stacking.predict(S_test)





sub.index = sub['fullVisitorId'].astype(np.double)

sub = sub.sort_index()

sub=sub.sort_index()





sub['PredictedLogRevenue']=pre_revenuelgb

sub.to_csv("lgb.csv",index = False)

sub['PredictedLogRevenue']=pre_revenuexgb

sub.to_csv("xgb.csv",index = False)

sub['PredictedLogRevenue']=pre_revenuesoftvoting

sub.to_csv("soft_voting.csv",index = False)

sub['PredictedLogRevenue']=pre_revenuestacking

sub.to_csv("stacking.csv",index = False)





sub['PredictedLogRevenue']=pre_revenuelgb * pre_return

sub.to_csv("lgb_nb.csv",index = False)

sub['PredictedLogRevenue']=pre_revenuexgb * pre_return

sub.to_csv("xgb_nb.csv",index = False)

sub['PredictedLogRevenue']=pre_revenuesoftvoting * pre_return

sub.to_csv("soft_voting_nb.csv",index = False)

sub['PredictedLogRevenue']=pre_revenuestacking * pre_return

sub.to_csv("stacking_nb.csv",index = False)

Models_score={'lgb':0.88901,'xgb':0.89157,'soft_voting':0.88937,'stacking':0.88843,

             'lgb_nb':0.88744,'xgb_nb':0.88817,'soft_voting_nb':0.88937,'stacking_nb':0.88843}
import seaborn as sns

from pylab import rcParams

%matplotlib inline
modeldf=pd.DataFrame(list(Models_score.items()), columns=['Model', 'RMSE'])

rcParams['figure.figsize'] = 25, 10

rcParams['font.size'] = 15

sns.set_style("darkgrid")

ax = sns.lineplot(x="Model", y="RMSE", data=modeldf)