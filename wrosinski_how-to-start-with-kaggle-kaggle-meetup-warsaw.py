import gc
import glob
import os
import time

import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

gc.enable()

pd.options.display.max_rows = 96
pd.options.display.max_columns = 128
input_dir = '../input/'
input_files = sorted(glob.glob(input_dir + '*'))

input_files
%%time

train = pd.read_csv('../input/train.csv', parse_dates=['first_active_month'])
test = pd.read_csv('../input/test.csv', parse_dates=['first_active_month'])
sample_submission = pd.read_csv('../input/sample_submission.csv')


merchant = pd.read_csv('../input/merchants.csv')
new_merchant = pd.read_csv('../input/new_merchant_transactions.csv')
# historical = pd.read_csv('../input/historical_transactions.csv')
train.head()
test.head()
merchant.head()
new_merchant.head()
print('Train NaN:\n\n{}\n'.format(np.sum(pd.isnull(train))))
print('Test NaN:\n\n{}\n'.format(np.sum(pd.isnull(test))))
print('Merchant NaN:\n\n{}\n'.format(np.sum(pd.isnull(merchant))))
print('New Merchant NaN:\n\n{}\n'.format(np.sum(pd.isnull(new_merchant))))
new_merchant = new_merchant.dropna(subset=['merchant_id'])
merchant_id_num = len(merchant['merchant_id'].unique())
new_merchant_id_num = len(new_merchant['merchant_id'].unique())
merchant_id_intersect = len(np.intersect1d(new_merchant.merchant_id, merchant.merchant_id))

print('Merchant IDs: {}'.format(merchant_id_num))
print('New merchant IDs: {}'.format(new_merchant_id_num))
print('Merchants ID intersection: {}'.format(merchant_id_intersect))
train_card_id_num = len(train['card_id'].unique())
test_card_id_num = len(test['card_id'].unique())
train_card_id_intersect = len(np.intersect1d(new_merchant.card_id, train.card_id))
test_card_id_intersect = len(np.intersect1d(new_merchant.card_id, test.card_id))

print('train card IDs: {}'.format(train_card_id_num))
print('test card IDs: {}'.format(test_card_id_num))
print('train card IDs intersection: {}'.format(train_card_id_intersect))
print('test card IDs intersection: {}'.format(test_card_id_intersect))
train_id_frac = train_card_id_intersect / train_card_id_num
test_id_frac = test_card_id_intersect / test_card_id_num

print('train frac: {:.3f}, test frac: {:.3f}'.format(train_id_frac, test_id_frac))
# Get columns of each type
def get_column_types(df):

    categorical_columns = [
        col for col in df.columns if df[col].dtype == 'object']
    categorical_columns_int = [
        col for col in df.columns if df[col].dtype == 'int']
    numerical_columns = [
        col for col in df.columns if df[col].dtype == 'float']

    categorical_columns = [
        x for x in categorical_columns if 'id' not in x]
    categorical_columns_int = [
        x for x in categorical_columns_int if 'id' not in x]

    return categorical_columns, categorical_columns_int, numerical_columns


# Rename columns after grouping for easy merge and access
def rename_columns(df):
    
    df.columns = pd.Index(['{}{}'.format(
        c[0], c[1].upper()) for c in df.columns.tolist()])
    
    return df
merchant_cat_feats, merchant_catint_feats, merchant_num_feats = get_column_types(merchant)

print('Categorical features to encode: {}'.format(merchant_cat_feats))
print('\nCategorical int features: {}'.format(merchant_catint_feats))
print('\nNumerical features: {}'.format(merchant_num_feats))
new_merchant_cat_feats, new_merchant_catint_feats, new_merchant_num_feats = get_column_types(new_merchant)

print('Categorical features to encode: {}'.format(new_merchant_cat_feats))
print('\nCategorical int features: {}'.format(new_merchant_catint_feats))
print('\nNumerical features: {}'.format(new_merchant_num_feats))
# Let's create set of aggregates, which will be used for features grouping.
# One for categorical and one for numerical features.

aggs_num_basic = ['mean', 'min', 'max', 'sum']
aggs_cat_basic = ['mean', 'sum', 'count']
# Encode string features to numbers:
# If encoding train and test separately, remember to keep the features mapping between the two!

for c in new_merchant_cat_feats:
    print('Encoding: {}'.format(c))
    new_merchant[c] = pd.factorize(new_merchant[c])[0]
    
for c in merchant_cat_feats:
    print('Encoding: {}'.format(c))
    merchant[c] = pd.factorize(merchant[c])[0]
    
new_merchant
merchant_card_id_cat = merchant.groupby(['merchant_id'])[merchant_cat_feats].agg(aggs_cat_basic)
merchant_card_id_num = merchant.groupby(['merchant_id'])[merchant_num_feats].agg(aggs_num_basic)

merchant_card_id_cat = rename_columns(merchant_card_id_cat)
merchant_card_id_num = rename_columns(merchant_card_id_num)
merchant_card_id_cat.head()
new_merchant_ = new_merchant.set_index('merchant_id').join(merchant_card_id_cat, how='left')
new_merchant_ = new_merchant_.join(merchant_card_id_num, how='left')
_, new_merchant_catint_feats2, new_merchant_num_feats2 = get_column_types(new_merchant_)

print('\nCategorical int features: {}'.format(new_merchant_catint_feats2))
print('\nNumerical features: {}'.format(new_merchant_num_feats2))
new_merchant_card_id_cat = new_merchant_.groupby(['card_id'])[new_merchant_catint_feats2].agg(aggs_cat_basic)
new_merchant_card_id_num = new_merchant_.groupby(['card_id'])[new_merchant_num_feats2].agg(aggs_num_basic)

new_merchant_card_id_cat = rename_columns(new_merchant_card_id_cat)
new_merchant_card_id_num = rename_columns(new_merchant_card_id_num)
train_ = train.set_index('card_id').join(new_merchant_card_id_cat, how='left')
train_ = train_.join(new_merchant_card_id_num, how='left')

test_ = test.set_index('card_id').join(new_merchant_card_id_cat, how='left')
test_ = test_.join(new_merchant_card_id_num, how='left')


del train, test
gc.collect()
y = train_.target
X = train_.drop(['target'], axis=1)
X_test = test_.copy()


features_to_remove = ['first_active_month']

X = X.drop(features_to_remove, axis=1)
X_test = X_test.drop(features_to_remove, axis=1)


# Assert that set of features is the same for both train and test DFs:
assert np.all(X.columns == X_test.columns)


del train_, test_
gc.collect()
np.sum(pd.isnull(X)) / X.shape[0]
np.sum(pd.isnull(X_test)) / X_test.shape[0]
# KFold splits
kf = KFold(n_splits=5, shuffle=True, random_state=1337)
# Column names:
train_cols = X.columns.tolist()


# LGB model parameters:
params = {'learning_rate': 0.03,
          'boosting': 'gbdt', 
          'objective': 'regression', 
          'metric': 'rmse',
          'num_leaves': 64,
          'min_data_in_leaf': 6,
          'max_bin': 255,
          'bagging_fraction': 0.7,
          'lambda_l2': 1e-4,
          'max_depth': 12,
          'seed': 1337,
          'nthreads': 6}


# Placeholders for out-of-fold predictions
oof_val = np.zeros((X.shape[0]))
oof_test = np.zeros((5, X_test.shape[0]))


i = 0 # Placeholder for fold indexing
for tr, val in kf.split(X, y):
    
    print('Fold: {}'.format(i + 1))
    
    # Split into training and validation part
    X_tr, y_tr = X.iloc[tr, :], y.iloc[tr]
    X_val, y_val = X.iloc[val, :], y.iloc[val]
    
    # Create Dataset objects for lgb model
    dtrain = lgb.Dataset(X_tr.values, y_tr.values, feature_name=train_cols)
    dvalid = lgb.Dataset(X_val.values, y_val.values,
                         feature_name=train_cols, reference=dtrain)
    
    # Train model
    lgb_model = lgb.train(params, dtrain, 
                      num_boost_round=1000, 
                      valid_sets=(dvalid,), 
                      valid_names=('valid',), 
                      verbose_eval=25, 
                      early_stopping_rounds=20)
    
    # Save predictions for each fold
    oof_val[val] = lgb_model.predict(X_val)
    oof_test[i, :] = lgb_model.predict(X_test)
    
    i += 1
# Check RMSE for training set:
valid_rmse = mean_squared_error(y, oof_val) ** .5

print('Valid RMSE: {:.4f}'.format(valid_rmse))
# Average test predcitions across folds:
test_preds = oof_test.mean(axis=0)

# Create submission:
sample_submission['target'] = test_preds
sample_submission.to_csv("submission_trial.csv", index=False)
sample_submission.head()
