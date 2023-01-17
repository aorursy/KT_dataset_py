import os



import numpy as np

import pandas as pd

from sklearn import preprocessing

import random

seed = 10

np.random.seed(seed)

random.seed(seed)

os.environ['PYTHONHASHSEED'] = str(seed)

np.random.seed(seed)
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
%%time

train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')

sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')

train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')
train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



print(train.shape)

print(test.shape)



y_train = train['isFraud'].copy()



# Drop target, fill in NaNs

X_train = train.drop('isFraud', axis=1)

X_test = test.copy()



del train, test



# Label Encoding

for f in X_train.columns:

    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(X_train[f].values) + list(X_test[f].values))

        X_train[f] = lbl.transform(list(X_train[f].values))

        X_test[f] = lbl.transform(list(X_test[f].values)) 
col_del = []

for i in range(339):

    col = "V" + str(i+1)

    s = train_transaction[col].fillna(0).map(lambda x:0 if x%1 == 0 else 1).sum()

    if s > 300000:

        print(col,s)

        col_del.append(col)

features = [x for x in X_train.columns if x not in col_del]
train_transaction['hour'] = train_transaction['TransactionDT'].map(lambda x:(x//3600)%24)

test_transaction['hour'] = test_transaction['TransactionDT'].map(lambda x:(x//3600)%24)

train_transaction['weekday'] = train_transaction['TransactionDT'].map(lambda x:(x//(3600 * 24))%7)

test_transaction['weekday'] = test_transaction['TransactionDT'].map(lambda x:(x//(3600 * 24))%7)



del train_transaction, train_identity, test_transaction, test_identity
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

imp.fit(X_train[features])

X_train[features] = imp.transform(X_train[features])
imp.fit(X_test[features])

X_test[features] = imp.transform(X_test[features])

del imp
print(X_train[features])
%%time

# WARNING! THIS CAN DAMAGE THE DATA 



X_train = reduce_mem_usage(X_train)

X_test = reduce_mem_usage(X_test)
import gc

gc.collect()
from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

import lightgbm as lgb

cate = [x for x in X_train.columns if (x == 'ProductCD' or  x.startswith("addr") or x.startswith("card") or 

                                       x.endswith("domain") or x.startswith("Device")) and not x.endswith("count") ]

print(cate)

params = {'application': 'binary',

          'boosting': 'gbdt',

          'metric': 'auc',

          'max_depth': 16,

          'learning_rate': 0.03,

          'bagging_fraction': 0.9,

          'feature_fraction': 0.9,

          'verbose': -1,

          'lambda_l1': 0.1,

          'lambda_l2': 0.01,

          'num_leaves': 500,

          'min_child_weight': 3,

          'data_random_seed': 17,

          'nthreads':4}



early_stop = 500

verbose_eval = 30

num_rounds = 600

# 

folds = 3

kf = KFold(n_splits = folds, shuffle = True, random_state=seed)

y_preds = np.zeros(X_test.shape[0])

feature_importance_df = pd.DataFrame()

i = 0

for tr_idx, val_idx in kf.split(X_train[features], y_train):



    

    X_tr = X_train.iloc[tr_idx, :]

    y_tr = y_train.iloc[tr_idx]

    d_train = lgb.Dataset(X_tr, label=y_tr,categorical_feature = cate)

    watchlist = []

    

    

    model = lgb.train(params,

                      train_set=d_train,

                      num_boost_round=num_rounds,

                      valid_sets=watchlist,

                      verbose_eval=verbose_eval)

        

    

    y_preds+= model.predict(X_test[features]) / folds

    

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = X_tr.columns

    fold_importance_df["importance"] = model.feature_importance()

    fold_importance_df["fold"] = i + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    i+=1

    del X_tr,d_train
print(y_preds)
sample_submission['isFraud'] = y_preds

sample_submission.to_csv('LightGBM.csv')