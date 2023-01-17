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
# Convert categorical data types to category; from Vesta's comments



cat_types = ['ProductCD','card1','card2','card3','card4','card5','card6','addr1','addr2','P_emaildomain','R_emaildomain',

             'M1','M2','M3','M4','M5','M6','M7','M8','M9','DeviceType','DeviceInfo','id_12','id_13','id_14','id_15','id_16',

             'id_17','id_18','id_19','id_20','id_21','id_22','id_23','id_24','id_25','id_26','id_27','id_28','id_29','id_30',

             'id_31','id_32','id_33','id_34','id_35','id_36','id_37','id_38']



train[cat_types] = train[cat_types].astype('category')

test[cat_types] = test[cat_types].astype('category')
# Create a new value for the transaction decimal



train['TransactionDecimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)

test['TransactionDecimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)
# Create a new value for the Datetime

# https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature

train['Transaction_DOW'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1) % 7)

test['Transaction_DOW'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)

train['Transaction_H'] = np.floor(train['TransactionDT'] / 3600) % 24

test['Transaction_H'] = np.floor(test['TransactionDT'] / 3600) % 24
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

import lightgbm as lgb

from sklearn.metrics import roc_auc_score



# With some help from https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm-w-gpu



params = {'num_leaves': 500,

          'min_child_weight': 0.03454472573214212,

          'feature_fraction': 0.3797454081646243,

          'bagging_fraction': 0.4181193142567742,

          'min_data_in_leaf': 106,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.006883242363721497,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": 'auc',

          "verbosity": -1,

          'reg_alpha': 0.3899927210061127,

          'reg_lambda': 0.6485237330340494,

          'random_state': 47,

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