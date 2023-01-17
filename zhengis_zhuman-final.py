import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import xgboost as xgb

from sklearn import preprocessing

from itertools import product

import os

%matplotlib inline
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

sales_train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv',parse_dates=["date"])

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
test.head()
sales_train.head()
shops.head()
items.head()
item_categories.head()


item_categories['big_category_name'] = item_categories['item_category_name'].map(lambda x: x.split(' - ')[0])

item_categories['big_category_name'].value_counts()


item_categories.loc[

    item_categories['big_category_name']=='Чистые носители (штучные)','big_category_name'

] = 'Чистые носители'





item_categories.loc[

    item_categories['big_category_name']=='Чистые носители (шпиль)','big_category_name'

] = 'Чистые носители'



item_categories['big_category_name'].value_counts()
item_categories
item_categories=item_categories.drop('item_category_name',axis=1)
item_categories
items
items=items.merge(item_categories)
items
items=items.drop(['item_name','item_category_id'],axis=1)

items


shops['city_name'] = shops['shop_name'].map(lambda x: x.split(' ')[0])



shops['city_name'].value_counts()
shops.loc[shops['city_name']=='!Якутск','city_name'] = 'Якутск'



shops['city_name'].value_counts()
shops
shops=shops.drop('shop_name',axis=1)

shops
sales_train['date_sales'] = sales_train['item_cnt_day'] * sales_train['item_price']

sales_train
sales_train[sales_train['item_cnt_day']<0].count()
sales_train[sales_train['item_cnt_day']<0]
sales_train.index[sales_train['item_cnt_day']<0]
sales_train=sales_train.drop(sales_train.index[sales_train['item_cnt_day']<0])

sales_train
mon_shop_item_cnt = sales_train[

    ['date_block_num','shop_id','item_id','item_cnt_day']

].groupby(

    ['date_block_num','shop_id','item_id'],

    as_index=False

).sum().rename(columns={'item_cnt_day':'mon_shop_item_cnt'})

mon_shop_item_cnt
for_graph1=mon_shop_item_cnt[['date_block_num','mon_shop_item_cnt']].groupby('date_block_num').sum()

for_graph1
mon_shop_item_sales = sales_train[

    ['date_block_num','shop_id','item_id','date_sales']

].groupby(

    ['date_block_num','shop_id','item_id'],

    as_index=False

).sum().rename(columns={'date_sales':'mon_shop_item_sales'})

mon_shop_item_sales
for_graph2=mon_shop_item_sales[['date_block_num','mon_shop_item_sales']].groupby('date_block_num').sum()

for_graph2
plt.title('graph2')

plt.xlabel('date_block_num')

plt.ylabel('month_sales')

plt.plot('mon_shop_item_sales',data=for_graph2)
plt.title('graph1')

plt.xlabel('date_block_num')

plt.ylabel('month_num')

plt.plot('mon_shop_item_cnt',data=for_graph1)
data_matome=mon_shop_item_cnt.merge(mon_shop_item_sales)

data_matome
items
shops.head()
data_matome=data_matome.merge(shops)
data_matome
data_matome=data_matome.merge(items)
data_matome
for_graph3= data_matome.groupby(

    ['date_block_num','big_category_name'],

    as_index=False

).sum()

plt.figure(figsize=(20, 10))

sns.lineplot(x='date_block_num',y='mon_shop_item_cnt',data=for_graph3,hue='big_category_name')

plt.title('Montly item counts by big category')
for_graph4= data_matome.groupby(

    ['date_block_num','city_name'],

    as_index=False

).sum()

plt.figure(figsize=(20, 10))

sns.lineplot(x='date_block_num',y='mon_shop_item_cnt',data=for_graph4,hue='city_name')

plt.title('Montly item counts by city')


train_full_comb = pd.DataFrame()

for i in range(35):

    mid = test[['shop_id','item_id']]

    mid['date_block_num'] = i

    train_full_comb = pd.concat([train_full_comb,mid],axis=0)
train_full_comb
train_test = pd.merge(

    train_full_comb,

    mon_shop_item_cnt,

    on=['date_block_num','shop_id','item_id'],

    how='left'

)



train_test
train_test = pd.merge(

    train_test,

    mon_shop_item_sales,

    on=['date_block_num','shop_id','item_id'],

    how='left'

)
train_test
train_test = pd.merge(

    train_test,

    items[['item_id','big_category_name']],

    on='item_id',

    how='left'

)
train_test
items
shops
train_test = pd.merge(

    train_test,

    shops[['shop_id','city_name']],

    on='shop_id',

    how='left'

)
train_test
train_test['mon_shop_item_cnt'] = train_test['mon_shop_item_cnt'].clip(0,20)


lag_col_list = ['mon_shop_item_cnt','mon_shop_item_sales']



lag_num_list = [1,2,3,11,12]





train_test = train_test.sort_values(

    ['shop_id', 'item_id','date_block_num'],

    ascending=[True, True,True]

).reset_index(drop=True)



train_test


for lag_col in lag_col_list:

    for lag in lag_num_list:

        set_col_name =  lag_col + '_' +  str(lag)

        df_lag = train_test[['shop_id', 'item_id','date_block_num',lag_col]].sort_values(

            ['shop_id', 'item_id','date_block_num'],

            ascending=[True, True,True]

        ).reset_index(drop=True).shift(lag).rename(columns={lag_col: set_col_name})

        train_test = pd.concat([train_test, df_lag[set_col_name]], axis=1)
train_test


train_test=train_test.fillna(0)
train_test
train_=train_test[(train_test['date_block_num']<=33)&(train_test['date_block_num']>=12)].reset_index(drop=True)

train_
test_ = train_test[train_test['date_block_num']==34].reset_index(drop=True)

 
test_
from sklearn.preprocessing import LabelEncoder

obj_col_list = ['big_category_name','city_name']

for obj_col in obj_col_list:

    le = LabelEncoder()

    train_[obj_col] = pd.DataFrame({obj_col:le.fit_transform(train_[obj_col])})

    test_[obj_col] = pd.DataFrame({obj_col:le.fit_transform(test_[obj_col])})
train_y = train_['mon_shop_item_cnt']

train_X = train_.drop(columns=['mon_shop_item_cnt','mon_shop_item_sales','date_block_num'])

test_X = test_.drop(columns=['mon_shop_item_cnt','mon_shop_item_sales','date_block_num'])
train_y
train_X
list(le.classes_)
import xgboost as xgb

dm_train = xgb.DMatrix(train_X, label=train_y)
param = {

    'max_depth': 10, 

    'eta': 0.3, 

    'objective': 'reg:squarederror',

    'colsample_bytree':1.0,

    'colsample_bylevel':0.3,

    'subsample':1,

    'gamma':0,

    'lambda':1,

    'alpha':0,

    'min_child_weight':0.5

    

}
num_round=1000,

model = xgb.train(param, dm_train)
xgb.plot_importance(model)


xgb.to_graphviz(model)
regr = xgb.XGBRegressor(max_depth = 10, min_child_weight=0.5, subsample = 1, eta = 0.3, num_round = 1000, seed = 1)


dm_test = xgb.DMatrix(test_X)



y_pred = model.predict(dm_test)
y_pred
np.sqrt(

    np.mean(

        np.square(

            np.array(

                np.array(train_y) - model.predict(dm_train)

            )

        )

    )

)
test_y = model.predict(dm_test)

test_X['item_cnt_month'] = test_y

submission = pd.merge(

    test,

    test_X[['shop_id','item_id','item_cnt_month']],

    on=['shop_id','item_id'],

    how='left'

)

submission[['ID','item_cnt_month']].to_csv('submission_2.csv', index=False)
from statsmodels.tsa.arima_model import ARIMA

ar = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

ar.date=ar.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))

ar=ar.groupby(["date_block_num"])["item_cnt_day"].sum()

ar.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')

ar=ar.reset_index()

ar=ar.loc[:,["index","item_cnt_day"]]

ar.columns = ['confirmed_date','count']

ar.head()
model = ARIMA(ar['count'].values, order=(1, 2, 1))

fit_model = model.fit(trend='c', full_output=True, disp=True)

fit_model.summary()
fit_model.plot_predict()

plt.title('Forecast vs Actual')

pd.DataFrame(fit_model.resid).plot()

forcast = fit_model.forecast(steps=6)

pred_y = forcast[0].tolist()

pred = pd.DataFrame(pred_y)
model = xgb.XGBRegressor(max_depth = 10, min_child_weight=0.5, subsample = 1, eta = 0.3, num_round = 1000, seed = 1)