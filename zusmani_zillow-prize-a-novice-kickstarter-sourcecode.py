# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

import gc

from sklearn.linear_model import LinearRegression

import random

import datetime as dt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train = pd.read_csv('../input/train_2016_v2.csv' , parse_dates=["transactiondate"])

prop = pd.read_csv('../input/properties_2016.csv')

sample = pd.read_csv('../input/sample_submission.csv')





print( "\nProcessing data for LightGBM ..." )

for c, dtype in zip(prop.columns, prop.dtypes):

    if dtype == np.float64:

        prop[c] = prop[c].astype(np.float32)



df_train = train.merge(prop, how='left', on='parcelid')

df_train.fillna(df_train.mean(),inplace = True)



x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 

                         'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag'], axis=1)



y_train = df_train['logerror'].values

print(x_train.shape, y_train.shape)

train_columns = x_train.columns



for c in x_train.dtypes[x_train.dtypes == object].index.values:

    x_train[c] = (x_train[c] == True)



del df_train; gc.collect()







x_train = x_train.values.astype(np.float32, copy=False)

d_train = lgb.Dataset(x_train, label=y_train)







##### RUN LIGHTGBM

params = {}

params['max_bin'] = 20

params['learning_rate'] = 0.0025 # shrinkage_rate

params['boosting_type'] = 'gbdt'

params['objective'] = 'regression'

params['metric'] = 'l1'          # or 'mae'

params['sub_feature'] = 0.345    

params['bagging_fraction'] = 0.85 # sub_row

params['bagging_freq'] = 40

params['num_leaves'] = 512        # num_leaf

params['min_data'] = 500         # min_data_in_leaf

params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf

params['verbose'] = 0

params['feature_fraction_seed'] = 2

params['bagging_seed'] = 3



print("\nFitting LightGBM model ...")

clf = lgb.train(params, d_train, 430)

del d_train; gc.collect()

del x_train; gc.collect()



print("\nPrepare for LightGBM prediction ...")

print("   Read sample file ...")

sample = pd.read_csv('../input/sample_submission.csv')

print("   ...")

sample['parcelid'] = sample['ParcelId']

print("   Merge with property data ...")

df_test = sample.merge(prop, on='parcelid', how='left')

print("   ...")

del sample, prop; gc.collect()

print("   ...")



x_test = df_test[train_columns]

print("   ...")

del df_test; gc.collect()



print("   Preparing x_test...")

for c in x_test.dtypes[x_test.dtypes == object].index.values:

    x_test[c] = (x_test[c] == True)

print("   ...")

x_test = x_test.values.astype(np.float32, copy=False)

print("Test shape :", x_test.shape)

print("\nStart LightGBM prediction ...")

p_test = clf.predict(x_test)



del x_test; gc.collect()



print( "\nUnadjusted LightGBM predictions:" )

print( pd.DataFrame(p_test).head() )



train = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])

properties = pd.read_csv("../input/properties_2016.csv")

submission = pd.read_csv("../input/sample_submission.csv")

train = pd.merge(train, properties, how='left', on='parcelid')

y = train['logerror'].values

test = pd.merge(submission, properties, how='left', left_on='ParcelId', right_on='parcelid')

properties = []

test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']

test_columns = ['201610','201611','201612','201710','201711','201712']

print( pd.DataFrame(p_test).head() )

for i in range(len(test_dates)):

    test['transactiondate'] = test_dates[i]

    pred = p_test

    submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]

    print('predict...', i)

print( submission.head() )

from datetime import datetime

print( "\nWriting results to disk ..." )

submission.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print( "\nFinished ...")



# Any results you write to the current directory are saved as output.