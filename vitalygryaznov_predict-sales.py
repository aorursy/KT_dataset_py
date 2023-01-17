import numpy as np 

import pandas as pd 

import json

import matplotlib.pyplot as plt

from datetime import datetime,timedelta

import re as re

from itertools import product

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
sub_path_to_data = '/kaggle/input/competitive-data-science-predict-future-sales/'

categories = pd.read_csv(sub_path_to_data + 'item_categories.csv')

items = pd.read_csv(sub_path_to_data + 'items.csv')

sales_train = pd.read_csv(sub_path_to_data + 'sales_train.csv')

shops = pd.read_csv(sub_path_to_data + 'shops.csv')

test = pd.read_csv(sub_path_to_data + 'test.csv')
sales_train.head(1)
sales_train.shape
test.head(1)
test.shape
items.head(1)
items.shape
categories.head(1)
categories.shape
shops.head(1)
shops.shape
sales_train.info()
sales_train.describe(include=[np.number]).T
sales_train.isnull().sum()
duplicate_sales = sales_train.loc[sales_train.duplicated(keep=False)]
duplicate_sales
sales_train.loc[sales_train.duplicated(subset= ['date', 'shop_id', 'item_id'], keep=False) &\

                ~sales_train.duplicated(keep=False)]
sales_train.loc[sales_train.duplicated(subset= ['date', 'shop_id', 'item_id', 'item_price'], keep=False) & ~sales_train.duplicated(keep=False)]
sales_train.drop_duplicates(keep="first", inplace=True)
sales_train.loc[sales_train.shop_id == 0, 'shop_id'] = 57

test.loc[test.shop_id == 0, 'shop_id'] = 57



sales_train.loc[sales_train.shop_id == 1, 'shop_id'] = 58

test.loc[test.shop_id == 1, 'shop_id'] = 58



sales_train.loc[sales_train.shop_id == 10, 'shop_id'] = 11

test.loc[test.shop_id == 10, 'shop_id'] = 11
sales_train.loc[~sales_train['date'].str.match('^[0-3]\d\.[0-1]\d\.20\d\d$')]
sales_train['date'] =  pd.to_datetime(sales_train['date'], format='%d.%m.%Y')
sales_sum = sales_train.groupby(['date'])['item_cnt_day'].sum().reset_index().set_index("date")
sales_sum.loc[sales_sum.item_cnt_day > 7000]
sales_train.loc[sales_train.item_price > 50000]
sales_train = sales_train.loc[sales_train.item_price < 50000]
sales_sum.plot(kind='bar', color='black', figsize=(24,6))
train = sales_train.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].sum().reset_index()
train.rename(columns={"item_cnt_day": "item_cnt_month"}, inplace=True)
train['item_cnt_month'] = train['item_cnt_month'].fillna(0).clip(0,20)
shops_list = train['shop_id'].unique()



fig, axs = plt.subplots(30,2, figsize=(20, 200), facecolor='w', edgecolor='k')

fig.subplots_adjust(hspace = .5, wspace=.1)

axs = axs.ravel()



for index, shop_id in enumerate(shops_list):

    shop_sales = train.loc[train['shop_id'] == shop_id]

    sales_per_month = shop_sales.groupby('date_block_num')['item_cnt_month'].sum().to_frame()

    

    axs[index].plot(sales_per_month.index, sales_per_month['item_cnt_month'], 'o-')

    axs[index].set_xticks(sales_per_month.index)

    axs[index].grid()

    

    axs[index].title.set_text("sales for shop {0}".format(shop_id))

    axs[index].set_xlabel('month')

    axs[index].set_ylabel('number of sales')

    

plt.show()
shop_ids_test = test.shop_id.unique()

for shop_id in [0,1,11,20,8]:

    print (shop_id in shop_ids_test)
#train = train.loc[~(train.shop_id.isin([0,1,11,20,8]))]

test['date_block_num'] = 34

train = pd.concat([train, test.drop('ID', axis=1)], ignore_index=True, sort=False, keys=['date_block_num', 'shop_id', 'item_id'])

train.fillna(0, inplace=True)
index_cols = ['shop_id', 'item_id', 'date_block_num']



grid = [] 

for block_num in train.date_block_num.unique():

    cur_shops = train[train['date_block_num']==block_num]['shop_id'].unique()

    cur_items = train[train['date_block_num']==block_num]['item_id'].unique()

    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))



grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)



train = pd.merge(grid,train,how='left',on=index_cols).fillna(0)



train.sort_values(['date_block_num','shop_id','item_id'],inplace=True)
cumsum = train.groupby('item_id')['item_cnt_month'].cumsum() - train['item_cnt_month']

cumc = train.groupby('item_id').cumcount() + 1

train['item_target_enc'] = cumsum/cumc

train['item_target_enc'].fillna(0.3343, inplace=True) 



encoded_feature = train['item_target_enc'].values
train['month'] = train['date_block_num'].map(lambda month: month-12*(month//12))
def number_of_weekens_in_month(first_day):

    ndays = first_day.daysinmonth

    weekends = 0

    for i in range(ndays):

        if (pd.to_datetime(first_day + timedelta(days=(np.long(i)))).dayofweek in [5, 6]): 

            weekends = weekends + 1

    return weekends
first_days_map = sales_train.groupby('date_block_num')['date'].min().map(lambda date: date.replace(day=1))
train['number_of_weekends'] = train['date_block_num'].map(first_days_map.map(number_of_weekens_in_month))
sales_train['month'] = sales_train['date_block_num'].map(lambda month: month-12*(month//12))
train['days_in_month'] = train['date_block_num'].map(first_days_map.map(lambda day: day.daysinmonth))
def specify_the_accuracy_group(row):

    num_of_block = row.date_block_num - 1

    if (num_of_block != -1):

        values = train.loc[(train['shop_id']==row.shop_id) & (train['item_id']==row.item_id)&\

                         (train['date_block_num']==num_of_block),['item_cnt_month']].values

        return 0 if (values.size==0) else values[0][0]

    else:

        return 0
def previous_months_value(df, collumn, offset):

    previous_month_values = df.copy()

    previous_month_values['date_block_num'] = previous_month_values['date_block_num'] + offset

    previous_month_values.rename(columns={collumn: 'prev_month_' + str(offset) + '_' + collumn}, inplace=True)

    return pd.merge(df, previous_month_values[['date_block_num','shop_id','item_id','prev_month_' + str(offset) + '_' + collumn]], on=['date_block_num','shop_id','item_id'], how='left')
offsets = [1,2,3,9]
column_to_offset = 'item_cnt_month'
for offset in offsets:

    train = previous_months_value(train, column_to_offset, offset)

    values = {'prev_month_' + str(offset) + '_' + column_to_offset: 0}

    train = train.fillna(value=values)
item_price_map = sales_train.loc[sales_train.item_price > 0].groupby(['item_id'])['item_price'].mean()
train['price'] = train['item_id'].map(item_price_map)
first_month_product_appeared = sales_train.groupby(['shop_id', 'item_id'])['date_block_num'].min().reset_index()

first_month_product_appeared.rename(columns={'date_block_num': 'first_appeared'}, inplace=True)

train = pd.merge(train,first_month_product_appeared, on=['shop_id','item_id'], how='left')

train['first_appeared'] = train['date_block_num'] - train['first_appeared'] + 1

train['first_appeared'] = train['first_appeared'].map(lambda n: 0 if (n < 0) else n)

values = {'first_appeared': 0}

train = train.fillna(value=values)
new_shop = train.groupby('shop_id')['date_block_num'].min().map(lambda month: 1 if (month > 10) else 0)
train['new_shop'] = train['shop_id'].map(new_shop)
sales_incompleate_data = train.groupby('shop_id')['date_block_num'].max().map(lambda month: 1 if (month < 33) else 0)
train['incompleate_data'] = train['shop_id'].map(sales_incompleate_data)
shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])

shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'

shops['city_code'] = LabelEncoder().fit_transform(shops['city'])

shops = shops[['shop_id','city_code']]
categories['split'] = categories['item_category_name'].str.split('-')

categories['type'] = categories['split'].map(lambda x: x[0].strip())

categories['type_code'] = LabelEncoder().fit_transform(categories['type'])



categories['subtype'] = categories['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

categories['subtype_code'] = LabelEncoder().fit_transform(categories['subtype'])

categories = categories[['item_category_id','type_code', 'subtype_code']]

train = pd.merge(train, shops, on=['shop_id'], how='left')
train.drop('month', axis=1, inplace=True)
train = pd.merge(train, items, on=['item_id'], how='left')

train = pd.merge(train, categories, on=['item_category_id'], how='left')

train['city_code'] = train['city_code'].astype(np.int8)

train['item_category_id'] = train['item_category_id'].astype(np.int8)

train['type_code'] = train['type_code'].astype(np.int8)

train['subtype_code'] = train['subtype_code'].astype(np.int8)
train.drop(['item_name'], axis=1, inplace=True)
X_train = train[train.date_block_num < 33].drop(['item_cnt_month'], axis=1)

Y_train = train[train.date_block_num < 33]['item_cnt_month']

X_valid = train[train.date_block_num == 33].drop(['item_cnt_month'], axis=1)

Y_valid = train[train.date_block_num == 33]['item_cnt_month']

X_test = train[train.date_block_num == 34].drop(['item_cnt_month'], axis=1)
baseline_preds = np.full((len(X_valid)), train.loc[train.date_block_num < 33, 'item_cnt_month'].mean())

rmse_b = np.sqrt(mean_squared_error(Y_valid.values, baseline_preds))

print("RMSE: %f" % (rmse_b))


parameters = {'tree_method': 'exact'}

model = XGBRegressor(

    max_depth=8,

    tree_method='exact',

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    seed=42)



model.fit(

    X_train, 

    Y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 

    verbose=True, 

    early_stopping_rounds = 10)
X_test.days_in_month = 30

X_test.number_of_weekends = 9
X_test = pd.merge(test.drop('ID', axis=1), X_test, on=['shop_id', 'item_id', 'date_block_num'], how='left')
Y_test = model.predict(X_test).clip(0, 20)



submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": Y_test

})

submission.to_csv('xgb_submission.csv', index=False)