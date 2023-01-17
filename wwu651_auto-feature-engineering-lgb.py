import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()



import lightgbm as lgb

import xgboost as xgb



# sklearn tools for model training and assesment

from sklearn.feature_selection import SelectFromModel,RFE

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from sklearn.metrics import roc_curve, auc, accuracy_score,roc_auc_score

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.base import clone



import gc

import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings("ignore")
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

train.head()
train.info()
sns.countplot(train['target'])
# Checking missing values

print(train.isnull().values.any())

print(test.isnull().values.any())
# Features that have high correlations with the target

features=[]

cor=[]

for feature in train.iloc[:,2:].columns:

    if (train['target'].corr(train[feature])>0.05)|(train['target'].corr(train[feature])<-0.05):

        features.append(feature)

        cor.append(train['target'].corr(train[feature]))



df_corr=pd.DataFrame({'Features': features,'Correlations':cor}).sort_values(by='Correlations').set_index('Features')



df_corr.plot(kind='barh',figsize=(10,8))
# Feature with high skewness

featuresSkew=[]

skewness=[]



for feature in train.iloc[:,2:].columns:

    if (train[feature].skew()>=0.5) | (train[feature].skew()<=-0.5) :

        featuresSkew.append(feature)

        skewness.append(train[feature].skew())



df_skew=pd.DataFrame({'Features':featuresSkew,'Skewness':skewness})

df_skew
import featuretools as ft

es = ft.EntitySet(id='Santander')



es.entity_from_dataframe(dataframe=train[features],

                         entity_id='train',

                         make_index = True,

                         index='index')



fm, feat= ft.dfs(entityset=es, 

                 target_entity='train',

                 trans_primitives=['multiply_numeric','add_numeric'],

                 max_depth=1)
train=pd.concat((train,fm.iloc[:,len(features):]),axis=1)

# release some memory

del fm

gc.collect()

fm=pd.DataFrame()

train.info()
es.entity_from_dataframe(dataframe=test[features],

                         entity_id='test',

                         make_index = True,

                         index='index')



fm_test, feat= ft.dfs(entityset=es, 

                 target_entity='test',

                 trans_primitives=['multiply_numeric','add_numeric'],

                 max_depth=1)
test=pd.concat((test,fm_test.iloc[:,len(features):]),axis=1)

# release some memory

del fm_test

gc.collect()

fm_test=pd.DataFrame()

test.info()
# Cross validate model with Kfold stratified cross val

random_state = 222

kfold = StratifiedKFold(n_splits=12,shuffle=True,random_state=random_state)

pred_val = np.zeros(len(train))

pred_test_baseline = np.zeros(len(test))

feature_base=train.columns.tolist()[2:202]
# https://www.kaggle.com/jesucristo/90-lines-solution-0-901-fast

param = {

        'bagging_freq': 5,

        'bagging_fraction': 0.4,

        'boost_from_average':'false',

        'boost': 'gbdt',

        'feature_fraction': 0.05,

        'learning_rate': 0.01,

        'max_depth': -1,  

        'metric':'auc',

        'min_data_in_leaf': 80,

        'min_sum_hessian_in_leaf': 10.0,

        'num_leaves': 13,

        'num_threads': 8,

        'tree_learner': 'serial',

        'objective': 'binary', 

        'verbosity': 1

    }
# Split X and y

X_train=train.iloc[:,2:]

y_train=train['target']

X_test=test.iloc[:,1:]
# Baseline model: LightGBM with no feature engineering and tunning.

for foldIdx, (trn_idx, val_idx) in enumerate(kfold.split(X_train.loc[:,feature_base], y_train)):

    print("Fold {}".format(foldIdx))

    lgbm_base=lgb.LGBMClassifier(n_estimators=100000,random_state=random_state,**param)

    lgbm_base.fit(X_train.iloc[trn_idx][feature_base],y_train[trn_idx],

                  eval_set=[(X_train.iloc[trn_idx][feature_base],y_train[trn_idx]),(X_train.iloc[val_idx][feature_base],y_train[val_idx])],

                  early_stopping_rounds = 3000,

                  verbose=5000)

    pred_val[val_idx] = lgbm_base.predict_proba(X_train.loc[val_idx,feature_base], num_iteration=lgbm_base.best_iteration_)[:,1]

    pred_test_baseline += lgbm_base.predict_proba(X_test[feature_base],num_iteration=lgbm_base.best_iteration_)[:,1] / kfold.n_splits
# Evaluation

print('AUC score: %.5f' % roc_auc_score(train['target'],pred_val))
# Feature selection: Drop features that are highly correlated

corr_matrix=X_train.corr().abs()

upper=corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.8

to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]
features_selected=[fea for fea in X_train.columns if fea not in to_drop]
# Fit the lightGBM with embeded_lgb_feature

pred_val = np.zeros(len(train))

pred_test = np.zeros(len(test))

for foldIdx, (trn_idx, val_idx) in enumerate(kfold.split(X_train.loc[:,features_selected], y_train)):

    print("Fold {}".format(foldIdx))

    lgbm=lgb.LGBMClassifier(n_estimators=100000,random_state=random_state,**param)

    lgbm.fit(X_train.iloc[trn_idx][features_selected],y_train[trn_idx],

                  eval_set=[(X_train.iloc[trn_idx][features_selected],y_train[trn_idx]),(X_train.iloc[val_idx][features_selected],y_train[val_idx])],

                  early_stopping_rounds = 3000,

                  verbose=5000)

    pred_val[val_idx] = lgbm.predict_proba(X_train.loc[val_idx,features_selected], num_iteration=lgbm.best_iteration_)[:,1]

    pred_test += lgbm.predict_proba(X_test[features_selected],num_iteration=lgbm.best_iteration_)[:,1] / kfold.n_splits
# Evaluation

print('AUC score: %.5f' % roc_auc_score(y_train,pred_val))
# Submission

submission_baseline = pd.DataFrame({'ID_code': test.ID_code.values,

                           'target':pred_test_baseline})

submission_baseline.to_csv("LGBM_baseline.csv", index=False)

submission = pd.DataFrame({'ID_code': test.ID_code.values,

                           'target':pred_test})

submission.to_csv("LGBM_V3.csv", index=False)