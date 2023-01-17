import pandas as pd

import numpy as np

import numba

import gc

import graphviz

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import lightgbm as lgb

import sklearn

from datetime import datetime

from tqdm import tqdm

from yellowbrick.classifier import ROCAUC,DiscriminationThreshold

print(sklearn.__version__,pd.__version__, np.__version__, lgb.__version__, numba.__version__)



from sklearn import model_selection, preprocessing, metrics, svm,linear_model

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate, StratifiedKFold

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import auc, roc_auc_score, roc_curve,  explained_variance_score, precision_recall_curve,average_precision_score,accuracy_score, classification_report

from sklearn.preprocessing import StandardScaler



%matplotlib inline

import ipywidgets as widget

from ipywidgets import interact, SelectMultiple

INTERACTIVE = True

pd.set_option('max_columns', 150)

pd.set_option('max_rows', 150)

pd.set_option('max_colwidth', 400)

pd.set_option('max_seq_items', 400)

pd.set_option('max_info_rows', 250)

pd.set_option('max_info_columns', 250)
dtypestrain = {}

dtypestrain['ID_code'] = 'category'

dtypestrain['target'] = 'int8'

for i in range(0,200):

    dtypestrain['var_' + str(i)] = 'float32'

    

dtypestest = {}

dtypestest['ID_code'] = 'category'

for i in range(0,200):

    dtypestest['var_' + str(i)] = 'float32'
%%time

train = pd.read_csv('../input/train.csv', dtype=dtypestrain).drop(['var_184','var_199'], axis=1)

test = pd.read_csv('../input/test.csv', dtype=dtypestest).drop(['var_184','var_199'], axis=1)
gc.collect()
X_train, X_val, y_train, y_val = train_test_split(train.drop(['ID_code','target'],axis=1), train['target'], test_size=0.3, shuffle=True)
LGBGBDT_PARAM = {

    'random_state' : 1981,

    'n_estimators' : 2000,

    'learning_rate': 0.1,

    'num_leaves': 16,

    'max_depth': 4,

    'metric' : ['auc','binary_logloss'],

    'boosting_type' : 'gbdt',

    'objective' : 'binary',

    'reg_alpha' : 2.03,

    'reg_lambda' : 4.7,

    'feature_fraction' : 0.8, #colsample_bytree

    'feature_fraction_seed' : 1981, 

    'max_bins' : 100,

    'min_split_gain': 0.0148,

    'min_child_weight' : 7.835, #min_sum_hessian_in_leaf 

    'min_data_in_leaf' : 1000, #min_child_samples

    'random_state' : 1981, # Updated from 'seed'

    'subsample' : .912, #also known as Bagging fraction!

    'subsample_freq' : 200, # also known as bagging frequency!

    'boost_from_average' : False,

    'verbose_eval' : 50,

    'is_unbalance' : True,

    #'scale_pos_weight' : 5.1,

    }



LGBGBDT = lgb.LGBMClassifier( **LGBGBDT_PARAM,

                    silent=-1,

#                     device = 'gpu', 

#                     gpu_platform_id = 0,

#                     gpu_device_id = 0,

#                     gpu_use_dp = True,

                )
%%time

LGBGBDT_FIT = LGBGBDT.fit(X_train, y_train,eval_set=[(X_val,y_val)], eval_metric= ['auc','binary_logloss'], early_stopping_rounds=100, verbose=50)
y_true, y_pred = train['target'], LGBGBDT.predict(train.drop(['ID_code','target'],axis=1))

print(classification_report(y_true, y_pred))

print(roc_auc_score(y_true, y_pred))

print()
LGBDART_PARAM = {

    'random_state' : 1981,

    'n_estimators' : 500, #very slow with 2000!

    'learning_rate': 0.1,

    'num_leaves': 16,

    'max_depth': 4,

    'metric' : ['auc','binary_logloss'],

    'boosting_type' : 'dart',

    'drop_rate' : 0.2,    ##only for DART

    'max_drop' : 100,    ##only for DART

    'objective' : 'binary',

    'reg_alpha' : 2.03,

    'reg_lambda' : 4.7,

    'feature_fraction' : 0.8, #colsample_bytree

    'feature_fraction_seed' : 1981, 

    'max_bins' : 100,

    'min_split_gain': 0.0148,

    'min_child_weight' : 7.835, #min_sum_hessian_in_leaf 

    'min_data_in_leaf' : 1000, #min_child_samples

    'random_state' : 1981, # Updated from 'seed'

    'subsample' : .912, #also known as Bagging fraction!

    'subsample_freq' : 200, # also known as bagging frequency!

    'boost_from_average' : False,

    'verbose_eval' : 50,

    'is_unbalance' : True,

    #'scale_pos_weight' : 5.1,

    }



LGBDART = lgb.LGBMClassifier( **LGBDART_PARAM,

                    silent=-1,

#                     device = 'gpu', 

#                     gpu_platform_id = 0,

#                     gpu_device_id = 0,

#                     gpu_use_dp = True,

                )
%%time

LGBDART_FIT = LGBDART.fit(X_train, y_train, 

                  eval_set=[(X_val,y_val)], 

                  eval_metric= ['auc','binary_logloss'], 

                  #early_stopping_rounds=100, 

                  verbose=50)
y_true, y_pred = train['target'], LGBDART.predict(train.drop(['ID_code','target'],axis=1))

print(classification_report(y_true, y_pred))

print(roc_auc_score(y_true, y_pred))

print()
LGBREGQNT_PARAM = {

    'random_state' : 1981,

    'n_estimators' : 2000,

    'learning_rate': 0.1,

    'num_leaves': 16,

    'max_depth': 4,

    'metric' : ['auc','quantile'],

    'boosting_type' : 'gbdt',

    'objective' : 'quantile',

    'reg_alpha' : 2.03,

    'reg_lambda' : 4.7,

    'feature_fraction' : 0.8, #colsample_bytree

    'feature_fraction_seed' : 1981, 

    'max_bins' : 100,

    'min_split_gain': 0.0148,

    'min_child_weight' : 7.835, #min_sum_hessian_in_leaf 

    'min_data_in_leaf' : 1000, #min_child_samples

    'random_state' : 1981, # Updated from 'seed'

    'subsample' : .912, #also known as Bagging fraction!

    'subsample_freq' : 200, # also known as bagging frequency!

    'boost_from_average' : False,

    'verbose_eval' : 50,

    'is_unbalance' : True,

    #'scale_pos_weight' : 5.1,

    }



LGBREGQNT = lgb.LGBMRegressor( **LGBREGQNT_PARAM,

                    silent=-1,

#                     device = 'gpu', 

#                     gpu_platform_id = 0,

#                     gpu_device_id = 0,

#                     gpu_use_dp = True,

                )
%%time

LGBREGQNT_FIT = LGBREGQNT.fit(X_train, y_train, 

                  eval_set=[(X_val,y_val)], 

                  eval_metric= ['auc','quantile'], 

                  early_stopping_rounds=100, 

                  verbose=50)
y_true, y_pred = train['target'], LGBREGQNT.predict(train.drop(['ID_code','target'],axis=1))

y_predbool = y_pred > 0.5

print(classification_report(y_true, y_predbool))

print(roc_auc_score(y_true, y_pred))

print()
LGBGOSS_PARAM = {

    'random_state' : 1981,

    'n_estimators' : 2000,

    'learning_rate': 0.1,

    'num_leaves': 16,

    'max_depth': 4,

    'metric' : ['auc'],

    'boosting_type' : 'goss',

    'objective' : 'binary',

    'reg_alpha' : 2.03,

    'reg_lambda' : 4.7,

    'feature_fraction' : 0.8, #colsample_bytree

    'feature_fraction_seed' : 1981, 

    'max_bins' : 100,

    'min_split_gain': 0.0148,

    'min_child_weight' : 7.835, #min_sum_hessian_in_leaf 

    'min_data_in_leaf' : 1000, #min_child_samples

    'random_state' : 1981, # Updated from 'seed'

    #'subsample' : .912, #also known as Bagging fraction! (not used for goss)

    #'subsample_freq' : 200, # also known as bagging frequency! (not used for goss)

    'boost_from_average' : False,

    'verbose_eval' : 50,

    'is_unbalance' : True,

    #'scale_pos_weight' : 5.1,

    }



LGBGOSS = lgb.LGBMClassifier( **LGBGOSS_PARAM,

                    silent=-1,

#                     device = 'gpu', 

#                     gpu_platform_id = 0,

#                     gpu_device_id = 0,

#                     gpu_use_dp = True,

                )
%%time

LGBGOSS_FIT = LGBGOSS.fit(X_train, y_train, 

                  eval_set=[(X_val,y_val)], 

                  eval_metric= ['auc','binary_logloss'], 

                  early_stopping_rounds=100, 

                  verbose=50)
y_true, y_pred = train['target'], LGBGOSS.predict(train.drop(['ID_code','target'],axis=1))

print(classification_report(y_true, y_pred))

print(roc_auc_score(y_true, y_pred))

print()
gc.collect()
predGBDT = LGBGBDT.predict_proba(test.drop('ID_code',axis=1))

scoresGBDT = pd.DataFrame({'ID_code' : test['ID_code'], 'target' : predGBDT[:,-1]})

scoresGBDT.to_csv('Submission_LGBGBDT_isimbalan_16LV_4MD_LR1_{date}.csv'.format(

        date=datetime.today().strftime('%Y%m%d_%H%M')), index=False)
predDART = LGBDART.predict_proba(test.drop('ID_code',axis=1))

scoresDART = pd.DataFrame({'ID_code' : test['ID_code'], 'target' : predDART[:,-1]})

scoresDART.to_csv('Submission_LGBDART_isimbalan_16LV_4MD_LR1_{date}.csv'.format(

        date=datetime.today().strftime('%Y%m%d_%H%M')), index=False)
predREGQNT = LGBREGQNT.predict(test.drop('ID_code',axis=1))

scoresREGQNT = pd.DataFrame({'ID_code' : test['ID_code'], 'target' : predREGQNT})

scoresREGQNT.to_csv('Submission_LGBREGQNT_isimbalan_16LV_4MD_LR1_{date}.csv'.format(

        date=datetime.today().strftime('%Y%m%d_%H%M')), index=False)
predGOSS = LGBGOSS.predict_proba(test.drop('ID_code',axis=1))

scoresGOSS = pd.DataFrame({'ID_code' : test['ID_code'], 'target' : predGOSS[:,-1]})

scoresGOSS.to_csv('Submission_LGBGOSS_isimbalan_16LV_4MD_LR1_{date}.csv'.format(

        date=datetime.today().strftime('%Y%m%d_%H%M')), index=False)
scoresALL= pd.DataFrame({'ID_code' : test['ID_code'], 

                      'targetGBDT' : predGBDT[:,-1],

                      'targetDART' : predDART[:,-1],

                      'targetREGQNT' : predREGQNT,

                      'targetGOSS' : predGOSS[:,-1],

                      'targetAVG' : (predGBDT[:,-1] + 

                                     predDART[:,-1] + 

                                     predREGQNT + 

                                     predGOSS[:,-1]) / 4

                      })

scoresALL.to_csv('Submission_LGBALL_isimbalan_16LV_4MD_LR1_{date}.csv'.format(

        date=datetime.today().strftime('%Y%m%d_%H%M')), index=False)
def plot_feature_imp_gain(features=list(X_train.columns), models=[LGBGBDT], feature_count=50, plot_all=True):

    

    x = pd.DataFrame()



    for model in models:

        scores_df = pd.DataFrame()

        scores_df['feature'] = list(X_train.columns)

        scores_df['split_score'] = model.booster_.feature_importance(importance_type='split')

        scores_df['gain_score'] = model.booster_.feature_importance(importance_type='gain')

        x = pd.concat([scores_df,x])

        if plot_all == True:

            plt.figure(figsize=(20, 10))

            ax = plt.subplot(121)

            sns.barplot(x='split_score', y='feature', data=scores_df.sort_values('split_score', ascending=False).iloc[0:feature_count], ax=ax)

            ax.set_title('Feature scores wrt split importances - ' + str(model.get_params(deep=False)['boosting_type']), fontweight='bold', fontsize=14)

            # Plot Gain importances

            ax = plt.subplot(122)

            sns.barplot(x='gain_score', y='feature', data=scores_df.sort_values('gain_score', ascending=False).iloc[0:feature_count], ax=ax)

            ax.set_title('Feature scores wrt gain importances - ' + str(model.get_params(deep=False)['boosting_type']), fontweight='bold', fontsize=14)

            plt.tight_layout()

        else:

            continue

    

    plt.figure(figsize=(20, 25))

    ax = plt.subplot(121)

    sns.barplot(x='split_score', y='feature', data=x.sort_values('split_score', ascending=False), ax=ax)

    ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)

    # Plot Gain importances

    ax = plt.subplot(122)

    sns.barplot(x='gain_score', y='feature', data=x.sort_values('gain_score', ascending=False), ax=ax)

    ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)

    plt.tight_layout()

plot_feature_imp_gain(models=[LGBGBDT,LGBDART,LGBREGQNT,LGBGOSS], plot_all=True)

plt.figure(figsize=(25, 5))

ax = plt.subplot(131)

sns.heatmap(scoresALL.corr(method='pearson'))

ax = plt.subplot(132)

sns.heatmap(scoresALL.corr(method='kendall'))

ax = plt.subplot(133)

sns.heatmap(scoresALL.corr(method='spearman'))