# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



!pip install -i https://test.pypi.org/simple/  litemort==0.1.7

from LiteMORT import *



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 100)



from itertools import product

from sklearn.preprocessing import LabelEncoder



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from xgboost import XGBRegressor

from xgboost import plot_importance



def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)



import time

import sys

import gc

import pickle
item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

sample_submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
sales_train['item_price'].max()
plt.figure(figsize=(10,4))

plt.xlim(-100, 3000)

sns.boxplot(x=sales_train.item_cnt_day)



plt.figure(figsize=(10,4))

plt.xlim(sales_train.item_price.min(), sales_train.item_price.max()*1.1)

sns.boxplot(x=sales_train.item_price)
sales_train = sales_train[sales_train.item_price<100000]

sales_train = sales_train[sales_train.item_cnt_day<1001]
median = sales_train[(sales_train.shop_id==32)&(sales_train.item_id==2973)&(sales_train.date_block_num==4)&(sales_train.item_price>0)].item_price.median()

sales_train.loc[sales_train.item_price<0, 'item_price'] = median
# Якутск Орджоникидзе, 56

sales_train.loc[sales_train.shop_id == 0, 'shop_id'] = 57

test.loc[test.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный"

sales_train.loc[sales_train.shop_id == 1, 'shop_id'] = 58

test.loc[test.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м²

sales_train.loc[sales_train.shop_id == 10, 'shop_id'] = 11

test.loc[test.shop_id == 10, 'shop_id'] = 11
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'

shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])

shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'

shops['city_code'] = LabelEncoder().fit_transform(shops['city'])

shops = shops[['shop_id','city_code']]



item_categories['split'] = item_categories['item_category_name'].str.split('-')

item_categories['type'] = item_categories['split'].map(lambda x: x[0].strip())

item_categories['type_code'] = LabelEncoder().fit_transform(item_categories['type'])

# if subtype is nan then type

item_categories['subtype'] = item_categories['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

item_categories['subtype_code'] = LabelEncoder().fit_transform(item_categories['subtype'])

item_categories = item_categories[['item_category_id','type_code', 'subtype_code']]



items = pd.merge(items, item_categories, on=['item_category_id'], how='left')

sales_train = pd.merge(sales_train, shops, on=['shop_id'], how='left')

sales_train = pd.merge(sales_train, items, on=['item_id'], how='left')

test = pd.merge(test, shops, on=['shop_id'], how='left')

test = pd.merge(test, items, on=['item_id'], how='left')



sales_train.drop(['item_name'], axis=1, inplace=True)

test.drop(['item_name'], axis=1, inplace=True)

del shops

del item_categories

del items

gc.collect()
train = sales_train.drop(['date'], axis = 1)

grouped_train = train.groupby(['date_block_num', 'shop_id', 'item_id', 'item_price'], 

                              as_index=False).sum()

grouped_train['total_sales'] = grouped_train['item_price'] * grouped_train['item_cnt_day']

sales_train.shape, grouped_train.shape
del sales_train

del train

gc.collect()
train_in_test = grouped_train[grouped_train['item_id'].isin(test['item_id'])]
del grouped_train

gc.collect()
def downcast_dtypes(df):

    float_cols = [c for c in df if df[c].dtype == "float64"]

    int_cols = [c for c in df if df[c].dtype in ["int64", "int32", "int16"]]

    df[float_cols] = df[float_cols].astype(np.float32)

    df[int_cols] = df[int_cols].astype(np.int8)

    return df
downcast_train = downcast_dtypes(train_in_test)

downcast_test = downcast_dtypes(test)
del train_in_test

gc.collect()
downcast_train.shape, downcast_test.shape
downcast_train.info()
train_stats = downcast_train.describe()

train_stats = train_stats.transpose()

def norm(x):

  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(downcast_train)

normed_test_data = norm(downcast_test)

normed_test_data=normed_test_data.drop(['ID', 'date_block_num', 'item_cnt_day', 'item_price', 'total_sales'], axis=1)
normed_train_data['date_block_num'].unique()
normed_train_data.head()
normed_test_data.head()
X_train = normed_train_data[normed_train_data.date_block_num < 1.4].drop(['item_price', 'item_cnt_day', 'total_sales'], axis=1)

Y_train = normed_train_data[normed_train_data.date_block_num < 1.4]['total_sales']

X_valid = normed_train_data[normed_train_data.date_block_num > 1.4].drop(['item_price', 'item_cnt_day', 'total_sales'], axis=1)

Y_valid = normed_train_data[normed_train_data.date_block_num > 1.4]['total_sales']

X_test = normed_test_data[['shop_id', 'item_id', 'city_code', 'item_category_id', 'type_code', 'subtype_code']]

X_test.insert(0, 'date_block_num', 1.5)
del downcast_train

del downcast_test

gc.collect()
X_train.shape
X_train.fillna(0, inplace=True)

Y_train.fillna(0, inplace=True)

X_test.fillna(0, inplace=True)
params={'num_leaves': 550,   

        'n_estimators':1000,

        'early_stopping_rounds':20,

        'feature_fraction': 1,     

        'bagging_fraction': 1,

        'max_bin': 512,

        'max_depth': 10,

        'min_child_weight': 300,    #'min_data_in_leaf': 300,

        'learning_rate': 0.1,

        'objective': 'regression',

        'boosting_type': 'gbdt',

        'verbose': 1,

        'metric': {'rmse'}

}



print(f"Call LiteMORT... ")    

t0=time.time()

model = LiteMORT(params).fit(X_train,Y_train,eval_set=[(X_valid, Y_valid)])

print(f"LiteMORT......OK time={time.time()-t0:.4g} model={model}")



#Y_pred = model.predict(X_valid).clip(0, 20)

#score = np.sqrt(mean_squared_error(Y_pred, Y_valid))

#Y_test = model.predict(X_test).clip(0, 20)

#print(f"score={score}")
#model = XGBRegressor(

#    max_depth=8,

#    n_estimators=1000,

#    min_child_weight=300,

#    colsample_bytree=0.8, 

#    subsample=0.8, 

#    eta=0.3,    

#    seed=42)



#model.fit(

#    X_train, 

#    Y_train, 

#    eval_metric="rmse", 

#    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 

#    verbose=True, 

#    early_stopping_rounds = 10)
from sklearn.metrics import mean_squared_error



Y_pred = model.predict(X_valid).clip(0, 20)

Y_test = model.predict(X_test).clip(0, 20)

score = np.sqrt(mean_squared_error(Y_pred, Y_valid))

print(score)

submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": Y_test

})

submission.to_csv('xgb_submission.csv', index=False)