import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import gc
train_identity = pd.read_csv('../input/train_identity.csv')

train_transaction = pd.read_csv('../input/train_transaction.csv')
train_transaction['isFraud'].value_counts()
train = train_transaction.merge(train_identity, on='TransactionID', how='left')

del train_identity, train_transaction

gc.collect()
test_identity = pd.read_csv('../input/test_identity.csv')

test_transaction = pd.read_csv('../input/test_transaction.csv')

test = test_transaction.merge(test_identity, on='TransactionID', how='left')

del test_identity, test_transaction

gc.collect()
# Create a new value for the transaction decimal



train['TransactionDecimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)

test['TransactionDecimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)
# Create a new values from continuous variable 'id_02'



train['id_02_bins'] = train['id_02']

train['id_02_bins'] = pd.cut(train['id_02_bins'], 10, labels=False)

test['id_02_bins'] = test['id_02']

test['id_02_bins'] = pd.cut(test['id_02_bins'], 10, labels=False)
# Create a new values from continuous variable 'id_11'



train['id_11_flag'] = np.where(train['id_11'].isnull(),'F','T')

test['id_11_flag'] = np.where(test['id_11'].isnull(),'F','T')

train['id_11_flag'] = train['id_11_flag'].astype('category')

test['id_11_flag'] = test['id_11_flag'].astype('category')



train['id_11_residual'] = train['id_11']

train['id_11_residual'] = np.where(train['id_11'] == 100.0,'100',np.where(train['id_11_residual'].isnull(),'None','Residual'))

train['id_11_residual'] = train['id_11_residual'].astype('category')

test['id_11_residual'] = test['id_11']

test['id_11_residual'] = np.where(test['id_11'] == 100.0,'100',np.where(test['id_11_residual'].isnull(),'None','Residual'))

test['id_11_residual'] = test['id_11_residual'].astype('category')
# Create a new value for the Datetime

# https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature

train['Transaction_DOW'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1) % 7)

test['Transaction_DOW'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)

train['Transaction_H'] = np.floor(train['TransactionDT'] / 3600) % 24

test['Transaction_H'] = np.floor(test['TransactionDT'] / 3600) % 24
from sklearn.model_selection import train_test_split

from bayes_opt import BayesianOptimization

from sklearn.metrics import roc_auc_score, roc_curve, auc

import lightgbm as lgb

import warnings

warnings.filterwarnings("ignore")
# Convert categorical data types to object for Bayesian Optimization



cat_types = ['ProductCD','card1','card2','card3','card4','card5','card6','addr1','addr2','P_emaildomain','R_emaildomain',

             'M1','M2','M3','M4','M5','M6','M7','M8','M9','DeviceType','DeviceInfo','id_12','id_13','id_14','id_15','id_16',

             'id_17','id_18','id_19','id_20','id_21','id_22','id_23','id_24','id_25','id_26','id_27','id_28','id_29','id_30',

             'id_31','id_32','id_33','id_34','id_35','id_36','id_37','id_38','id_02_bins','id_11_flag','id_11_residual']



train[cat_types] = train[cat_types].astype('object')

test[cat_types] = test[cat_types].astype('object')
train = train.fillna(-999)
from sklearn.preprocessing import LabelEncoder



for f in train.columns:

    if  train[f].dtype=='object': 

        lbl = LabelEncoder()

        lbl.fit(list(train[f].values) + list(test[f].values))

        train[f] = lbl.transform(list(train[f].values))

        test[f] = lbl.transform(list(test[f].values))  

train = train.reset_index()

test = test.reset_index()

features = list(train)

features.remove('isFraud')

target = 'isFraud'
# https://www.kaggle.com/vincentlugat/ieee-lgb-bayesian-opt



bayesian_tr_idx, bayesian_val_idx = train_test_split(train, test_size = 0.3, random_state = 42, stratify = train[target])

bayesian_tr_idx = bayesian_tr_idx.index

bayesian_val_idx = bayesian_val_idx.index
def LGB_bayesian(

    #learning_rate,

    num_leaves, 

    bagging_fraction,

    feature_fraction,

    min_child_weight, 

    min_data_in_leaf,

    max_depth,

    reg_alpha,

    reg_lambda

     ):

    

    # LightGBM expects next three parameters need to be integer. 

    num_leaves = int(num_leaves)

    min_data_in_leaf = int(min_data_in_leaf)

    max_depth = int(max_depth)



    assert type(num_leaves) == int

    assert type(min_data_in_leaf) == int

    assert type(max_depth) == int

    



    param = {

              'num_leaves': num_leaves, 

              'min_data_in_leaf': min_data_in_leaf,

              'min_child_weight': min_child_weight,

              'bagging_fraction' : bagging_fraction,

              'feature_fraction' : feature_fraction,

              #'learning_rate' : learning_rate,

              'max_depth': max_depth,

              'reg_alpha': reg_alpha,

              'reg_lambda': reg_lambda,

              'objective': 'binary',

              'save_binary': True,

              'seed': 11,

              'feature_fraction_seed': 11,

              'bagging_seed': 11,

              'drop_seed': 11,

              'data_random_seed': 11,

              'boosting_type': 'gbdt',

              'verbose': 1,

              'is_unbalance': False,

              'boost_from_average': True,

              'metric':'auc'}    

    

    oof = np.zeros(len(train))

    trn_data= lgb.Dataset(train.iloc[bayesian_tr_idx][features].values, label=train.iloc[bayesian_tr_idx][target].values)

    val_data= lgb.Dataset(train.iloc[bayesian_val_idx][features].values, label=train.iloc[bayesian_val_idx][target].values)



    clf = lgb.train(param, trn_data,  num_boost_round=50, valid_sets = [trn_data, val_data], verbose_eval=0, early_stopping_rounds = 50)

    

    oof[bayesian_val_idx]  = clf.predict(train.iloc[bayesian_val_idx][features].values, num_iteration=clf.best_iteration)  

    

    score = roc_auc_score(train.iloc[bayesian_val_idx][target].values, oof[bayesian_val_idx])



    return score
# Using Bayesian Optimization



# https://www.kaggle.com/vincentlugat/ieee-lgb-bayesian-opt



bounds_LGB = {

    'num_leaves': (31, 500), 

    'min_data_in_leaf': (20, 200),

    'bagging_fraction' : (0.1, 0.9),

    'feature_fraction' : (0.1, 0.9),

    #'learning_rate': (0.01, 0.3),

    'min_child_weight': (0.00001, 0.01),   

    'reg_alpha': (1, 2), 

    'reg_lambda': (1, 2),

    'max_depth':(-1,50),

}
# https://www.kaggle.com/vincentlugat/ieee-lgb-bayesian-opt



LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=42)
# https://www.kaggle.com/vincentlugat/ieee-lgb-bayesian-opt



init_points = 10

n_iter = 15
# https://www.kaggle.com/vincentlugat/ieee-lgb-bayesian-opt



with warnings.catch_warnings():

    warnings.filterwarnings('ignore')

    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
LGB_BO.max['target']
LGB_BO.max['params']
# Rebuild dataset



train_identity = pd.read_csv('../input/train_identity.csv')

train_transaction = pd.read_csv('../input/train_transaction.csv')

train = train_transaction.merge(train_identity, on='TransactionID', how='left')

del train_identity, train_transaction

gc.collect()



test_identity = pd.read_csv('../input/test_identity.csv')

test_transaction = pd.read_csv('../input/test_transaction.csv')

test = test_transaction.merge(test_identity, on='TransactionID', how='left')

del test_identity, test_transaction

gc.collect()



# Create a new value for the transaction decimal



train['TransactionDecimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)

test['TransactionDecimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)



# Create a new values from continuous variable 'id_02'



train['id_02_bins'] = train['id_02']

train['id_02_bins'] = pd.cut(train['id_02_bins'], 10, labels=False)

test['id_02_bins'] = test['id_02']

test['id_02_bins'] = pd.cut(test['id_02_bins'], 10, labels=False)



# Create a new values from continuous variable 'id_11'



train['id_11_flag'] = np.where(train['id_11'].isnull(),'F','T')

test['id_11_flag'] = np.where(test['id_11'].isnull(),'F','T')

train['id_11_flag'] = train['id_11_flag'].astype('category')

test['id_11_flag'] = test['id_11_flag'].astype('category')



train['id_11_residual'] = train['id_11']

train['id_11_residual'] = np.where(train['id_11'] == 100.0,'100',np.where(train['id_11_residual'].isnull(),'None','Residual'))

train['id_11_residual'] = train['id_11_residual'].astype('category')

test['id_11_residual'] = test['id_11']

test['id_11_residual'] = np.where(test['id_11'] == 100.0,'100',np.where(test['id_11_residual'].isnull(),'None','Residual'))

test['id_11_residual'] = test['id_11_residual'].astype('category')



# Create a new value for the Datetime

# https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature

train['Transaction_DOW'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1) % 7)

test['Transaction_DOW'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)

train['Transaction_H'] = np.floor(train['TransactionDT'] / 3600) % 24

test['Transaction_H'] = np.floor(test['TransactionDT'] / 3600) % 24
# Convert categorical data types to category; from Vesta's comments



cat_types = ['ProductCD','card1','card2','card3','card4','card5','card6','addr1','addr2','P_emaildomain','R_emaildomain',

             'M1','M2','M3','M4','M5','M6','M7','M8','M9','DeviceType','DeviceInfo','id_12','id_13','id_14','id_15','id_16',

             'id_17','id_18','id_19','id_20','id_21','id_22','id_23','id_24','id_25','id_26','id_27','id_28','id_29','id_30',

             'id_31','id_32','id_33','id_34','id_35','id_36','id_37','id_38']



train[cat_types] = train[cat_types].astype('category')

test[cat_types] = test[cat_types].astype('category')
# Prepare for model building

train_ID = train['TransactionID']

test_ID = test['TransactionID']

y = train['isFraud']

X = train.drop(['TransactionID','TransactionDT','isFraud'], axis=1)

X_test = test.drop(['TransactionID','TransactionDT'], axis=1)



del train, test

gc.collect()
# Quickly ensure comparable structures

X.shape, X_test.shape
from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score



# With some help from https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm-w-gpu



params = {'num_leaves': int(LGB_BO.max['params']['num_leaves']),

          'min_child_weight': LGB_BO.max['params']['min_child_weight'],

          'feature_fraction': LGB_BO.max['params']['feature_fraction'],

          'bagging_fraction': LGB_BO.max['params']['bagging_fraction'],

          'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']),

          'objective': 'binary',

          'max_depth': int(LGB_BO.max['params']['max_depth']),

          'learning_rate': 0.006883242363721497,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          'seed': 11,

          'drop_seed': 11,

          'feature_fraction_seed': 11,

          'data_random_seed': 11,

          'metric': 'auc',

          'verbosity': 1,

          'reg_alpha': LGB_BO.max['params']['reg_alpha'],

          'reg_lambda': LGB_BO.max['params']['reg_lambda'],

          'random_state': 47

         }
# https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm-w-gpu



NFOLDS = 5

folds = KFold(n_splits=NFOLDS)



columns = X.columns

splits = folds.split(X, y)

y_preds = np.zeros(X_test.shape[0])

y_oof = np.zeros(X.shape[0])

score = 0



# feature_importances = pd.DataFrame()

# feature_importances['feature'] = columns

  

for fold_n, (train_index, valid_index) in enumerate(splits):

    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    dtrain = lgb.Dataset(X_train, label=y_train)

    dvalid = lgb.Dataset(X_valid, label=y_valid)



    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)

    

#     feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

    

    y_pred_valid = clf.predict(X_valid)

    y_oof[valid_index] = y_pred_valid

    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")

    

    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS

    y_preds += clf.predict(X_test) / NFOLDS

    

    del X_train, X_valid, y_train, y_valid

    gc.collect()

    

print(f"\nMean AUC = {score}")

print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")
sub = pd.read_csv('../input/sample_submission.csv')

sub['isFraud'] = y_preds

sub.to_csv('submission.csv', index=False)