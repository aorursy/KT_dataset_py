import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import zipfile

import glob

import seaborn as snb

%matplotlib inline
'''import os 



sample = "kaggle/input/wns_analytics-wizard/sample_submission_IPsBlCT.zip"

test = "kaggle/input/wns_analytics-wizard/test_aq1FGdB.zip"

train = "kaggle/input/wns_analytics-wizard/train_NA17Sgz.zip"

all_zip_files = [sample, test, train]



print(all_zip_files)

for file in all_zip_files:

  with zipfile.ZipFile(file, 'r') as zip_ref:

    zip_ref.extractall()  ''' 
train_df = pd.read_csv('../input/wns-analytics-wizard/train_NA17Sgz/train.csv')

'''view_log_df = pd.read_csv(train + '/view_log.csv')

item_data_df = pd.read_csv(train + '/item_data.csv')

test_df = pd.read_csv(test + '/test.csv')

sample_df = pd.read_csv(sample + '/sample_submission.csv')'''
sample_df.head()
test_df.head()
item_data_df.head()
item_data_df.info()
item_data_df.describe().transpose()
plt.figure(figsize=(9,9))

plt.subplot(3,3,1)

item_data_df['category_1'].hist()

plt.subplot(3,3,2)

item_data_df['category_2'].hist()

plt.subplot(3,3,3)

item_data_df['category_3'].hist()

plt.show()
item_data_df['product_type'].hist()
item_data_df['item_price'].hist()
view_log_df.head()
print(view_log_df.isna().sum())

view_log_df.info()
print("No of users: ", view_log_df['user_id'].unique().size)
values = view_log_df['device_type'].value_counts().values

index = view_log_df['device_type'].value_counts().index

print(view_log_df['device_type'].value_counts())

snb.barplot(index, values)
view_log_df['item_id'].hist()
view_log_df['item_id'].hist()
train_df.head()
values = train_df['os_version'].value_counts().values

idx = train_df['os_version'].value_counts().index



snb.barplot(idx, values)
train_df['is_4G'].hist()
values = train_df['is_click'].value_counts().values

idx = train_df['is_click'].value_counts().index



snb.barplot(idx,values)
import re

print("View_log shape:", view_log_df.shape)

print("Item_data shape:", item_data_df.shape)

print("No of users in view_log:", view_log_df['user_id'].unique().size)

print("No of items in view_log:", view_log_df['item_id'].unique().size)

print("No of items in item_data:", item_data_df['item_id'].unique().size)



item_view_df = item_data_df.merge(view_log_df, how='left', on='item_id')

item_view_df = item_view_df.fillna(0)

print("Item_view data shape:", item_view_df.shape)



item_view_df_v1 = item_view_df[['user_id', 'item_id', 'session_id', 'product_type', 'device_type']].copy()

item_view_df_v1 = pd.get_dummies(item_view_df_v1)

print("shape: ", item_view_df_v1.shape)

item_view_df_v1.head()
train_df.head()

train_df_v2 = train_df.copy()



train_df_v2 = train_df_v2.merge(user_visit_log, on='user_id',how='left')

train_df_v2.head()
from sklearn import preprocessing



def date_time_split(df, col_name):

  month = df[col_name].apply(lambda x: int(x.split('-')[1]))

  day = df[col_name].apply(lambda x: int(x.split('-')[2].split(' ')[0]))

  hour = df[col_name].apply(lambda x: int(x.split('-')[2].split(' ')[1].split(':')[0]))

  minutes = df[col_name].apply(lambda x: int(x.split('-')[2].split(' ')[1].split(':')[1]))

  return month, day, hour, minutes



def get_clean_df(df, is_test_df=False, scale_data=False):

  month, day, hour, minute = date_time_split(df,'impression_time')

  df_v1 = df.copy()

  os_version = pd.get_dummies(df_v1['os_version'])

  df_v1 = df_v1.drop(['impression_id', 'impression_time', 'os_version'], axis=1)

  df_v2 = pd.concat([df_v1, os_version], axis=1)

  

  if is_test_df:

    all_cols = ['user_id', 'intermediate', 'latest', 'old', 'app_code', 'is_4G']

  else:

    all_cols = ['user_id', 'intermediate', 'latest', 'old', 'app_code', 'is_4G', 'is_click']

    

  df_v3 = df_v2[all_cols]

  df_v3.insert(1, 'month', month)

  df_v3.insert(2, 'day', day)

  df_v3.insert(3, 'hour', hour)

  df_v3.insert(4, 'minute', minute)

  

  if scale_data:

    df_v3 = preprocessing.scale(df_v3)

  return df_v3

from sklearn.model_selection import train_test_split



norm_train_df = get_clean_df(train_df)



features = 	['user_id', 'month', 'day', 'app_code', 'is_4G', 'is_click']



norm_train_df = norm_train_df[features]



train_x , cv_x, train_y, cv_y = train_test_split(norm_train_df[norm_train_df.columns[:-1]], norm_train_df['is_click'], shuffle=True, test_size=0.25)

#train_x, cv_x, train_y, cv_y = train_test_split(norm_train_df[:,:-1], norm_train_df[:,-1], shuffle=True, test_size=0.25)



print("Train shape:", norm_train_df.shape)

print("train data info:", norm_train_df.info())

norm_train_df.head()
norm_test_df = get_clean_df(test_df, is_test_df=True)



norm_test_df = norm_test_df[features[:-1]]

print("Test shape:", norm_test_df.shape)

print("Test data info:", norm_test_df.info())

norm_test_df.head()
'''import lightgbm as lgbm

from sklearn.metrics import roc_auc_score



model = lgbm.LGBMClassifier(n_estimators=1000, learning_rate=0.001)

#model.fit(X = norm_train_df[norm_train_df.columns[:-1]], y = norm_train_df['is_click'])

model.fit(train_x, train_y)



y_pred = model.predict(cv_x)



roc_auc_score(cv_y, y_pred)'''
import xgboost as xgb



params = {

    "objective": "binary:logistic",

    "booster" : "gbtree",

    "eval_metric": "auc",

    "eta": 0.03,

    "max_depth": 8,

    "subsample": 0.5,

    "colsample_bytree": 0.5,

    "alpha": 0.1

    }

dtrain = xgb.DMatrix(norm_train_df[norm_train_df.columns[:-1]], norm_train_df['is_click'])

#dtrain = xgb.DMatrix(train_x, train_y)

dvalid = xgb.DMatrix(cv_x, cv_y)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

gbm = xgb.train(params, dtrain, 1500, evals=watchlist, early_stopping_rounds=20, verbose_eval=True)
#y_pred = gbm.predict(xgb.DMatrix(cv_x))

y_pred = gbm.predict(xgb.DMatrix(norm_test_df))



#roc_auc_score(cv_y, y_pred)
final_output = pd.concat([test_df['impression_id'], pd.DataFrame({'is_click': y_pred})], axis=1)

final_output.to_csv('output.csv', index=False)