import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from itertools import product

from xgboost import XGBRegressor

from sklearn.preprocessing import LabelEncoder

import gc

%matplotlib inline
train = pd.read_csv("../input/sales_train.csv")

items = pd.read_csv("../input/items.csv")

shops = pd.read_csv("../input/shops.csv")

cats = pd.read_csv("../input/item_categories.csv")

test = pd.read_csv("../input/test.csv").set_index('ID')
train.head(2)
items.head(7)
shops.head(7)
cats.head(7)
test.head(2)
shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])

shops['city_code'] = LabelEncoder().fit_transform(shops['city'])

shops = shops[['shop_id','city_code']]

shops.head()
cats['type'] = cats['item_category_name'].str.split('-').map(lambda x: x[0].strip())

cats['type_code'] = LabelEncoder().fit_transform(cats['type'])

# if subtype is nan then type

cats['subtype'] = cats['item_category_name'].str.split('-').map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])

cats = cats[['item_category_id','type_code', 'subtype_code']]
cats.head()
train[train['item_price'] < 0]
median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()

train.loc[train.item_price<0, 'item_price'] = median
train['date'] = pd.to_datetime(train['date'])

train['month'] = train['date'].apply(lambda time: time.month)

train['year']  = train['date'].apply(lambda time: time.year)
train['revenue'] = train['item_price'] *  train['item_cnt_day']
test['date_block_num'] = 34

test['month'] = 11

test['year'] = 2015
total = pd.concat([train.drop('date',axis=1), test], ignore_index=True, sort=False)

total.fillna(0,inplace=True)

total.head()
group = total.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})

group.columns = ['item_cnt_month']



total = pd.merge(total, group, on=['date_block_num','shop_id','item_id'], how='left')

total['item_cnt_month'] = total['item_cnt_month'].fillna(0)



total.head()
items.drop('item_name',axis=1,inplace=True)

total = pd.merge(total, items, on='item_id', how='left')



total = pd.merge(total, cats, on='item_category_id', how='left')



total = pd.merge(total, shops, on='shop_id', how='left')

total.head()
def lag(lags, col):

    

    for i in lags:

        

        total[col+'_lag_'+str(i)] = total[col].shift(i)

        

    total.fillna(0, inplace=True)
lag([1,2,3,6,12], 'item_cnt_month')
group = total.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})

group.columns = ['date_avg_item_cnt']



total = pd.merge(total, group, on=['date_block_num'], how='left')

total['date_avg_item_cnt'] = total['date_avg_item_cnt'].fillna(0)



lag([1,2,3,6,12], 'date_avg_item_cnt')
group = total.groupby(['date_block_num','item_id']).agg({'item_cnt_month': ['mean']})

group.columns = ['date_item_avg_item_cnt']



total = pd.merge(total, group, on=['date_block_num','item_id'], how='left')

total['date_item_avg_item_cnt'] = total['date_item_avg_item_cnt'].fillna(0)



lag([1,2,3,6,12], 'date_item_avg_item_cnt')
group = total.groupby(['date_block_num','shop_id']).agg({'item_cnt_month': ['mean']})

group.columns = ['date_shop_avg_item_cnt']



total = pd.merge(total, group, on=['date_block_num','shop_id'], how='left')

total['date_shop_avg_item_cnt'] = total['date_shop_avg_item_cnt'].fillna(0)



lag([1,2,3,6,12], 'date_shop_avg_item_cnt')
group = total.groupby(['date_block_num','shop_id','item_category_id']).agg({'item_cnt_month': ['mean']})

group.columns = ['date_shop_cat_avg_item_cnt']



total = pd.merge(total, group, on=['date_block_num','shop_id','item_category_id'], how='left')

total['date_shop_cat_avg_item_cnt'] = total['date_shop_cat_avg_item_cnt'].fillna(0)



lag([1], 'date_shop_avg_item_cnt')
group = total.groupby(['date_block_num','item_category_id']).agg({'item_cnt_month': ['mean']})

group.columns = ['date_cat_avg_item_cnt']



total = pd.merge(total, group, on=['date_block_num','item_category_id'], how='left')

total['date_cat_avg_item_cnt'] = total['date_cat_avg_item_cnt'].fillna(0)



lag([1], 'date_shop_avg_item_cnt')
group = total.groupby(['date_block_num','city_code','item_id']).agg({'item_cnt_month': ['mean']})

group.columns = ['date_city_item_avg_item_cnt']



total = pd.merge(total, group, on=['date_block_num','city_code','item_id'], how='left')

total['date_city_item_avg_item_cnt'] = total['date_city_item_avg_item_cnt'].fillna(0)



lag([1], 'date_shop_avg_item_cnt')
#total= total[total['date_block_num'] > 11 ]

#total.head(5)
X_train = total[total.date_block_num < 33].drop(['item_cnt_month'], axis=1)

y_train = total[total.date_block_num < 33]['item_cnt_month']

X_valid = total[total.date_block_num == 33].drop(['item_cnt_month'], axis=1)

y_valid = total[total.date_block_num == 33]['item_cnt_month']

X_test = total[total.date_block_num == 34].drop(['item_cnt_month'], axis=1)

del train

del items

del shops

del cats 

del group

del total

gc.collect()
model = XGBRegressor(

    max_depth=8,

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    seed=42)



model.fit(

    X_train, 

    y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, y_train), (X_valid, y_valid)], 

    verbose=True, 

    early_stopping_rounds = 10)
y_pred = model.predict(X_valid).clip(0, 20)

y_test = model.predict(X_test).clip(0, 20)
submission = pd.DataFrame({"ID":test.index, "item_cnt_month":y_test})

submission.to_csv('submission.csv',index=False)