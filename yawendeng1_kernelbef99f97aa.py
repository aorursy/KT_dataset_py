import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import seaborn as sns

import time

import gc

import pickle

from itertools import product

from sklearn.preprocessing import LabelEncoder

sales_train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
sales_by_item_id = sales_train.pivot_table(index=['item_id'],values=['item_cnt_day'], 

                                        columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()

sales_by_item_id.columns = sales_by_item_id.columns.droplevel().map(str)

sales_by_item_id = sales_by_item_id.reset_index(drop=True).rename_axis(None, axis=1)

sales_by_item_id.columns.values[0] = 'item_id'

shops['shop_name'] = shops['shop_name'].apply(lambda x: x.lower()).str.replace('[^\w\s]', '').str.replace('\d+','').str.strip()

shops['shop_city'] = shops['shop_name'].str.partition(' ')[0]

shops['shop_type'] = shops['shop_name'].apply(lambda x: 'мтрц' if 'мтрц' in x else 'трц' if 'трц' in x else 'трк' if 'трк' in x else 'тц' if 'тц' in x else 'тк' if 'тк' in x else 'NO_DATA')

shops.head()

shops['shop_city_code'] = LabelEncoder().fit_transform(shops['shop_city'])

shops['shop_type_code'] = LabelEncoder().fit_transform(shops['shop_type'])

shops.head()

lines1 = [26,27,28,29,30,31]

lines2 = [81,82]

for index in lines1:

    category_name = categories.loc[index,'item_category_name']

#    print(category_name)

    category_name = category_name.replace('Игры','Игры -')

#    print(category_name)

    categories.loc[index,'item_category_name'] = category_name

for index in lines2:

    category_name = categories.loc[index,'item_category_name']

#    print(category_name)

    category_name = category_name.replace('Чистые','Чистые -')

#    print(category_name)

    categories.loc[index,'item_category_name'] = category_name

category_name = categories.loc[32,'item_category_name']

#print(category_name)

category_name = category_name.replace('Карты оплаты','Карты оплаты -')

#print(category_name)

categories.loc[32,'item_category_name'] = category_name

categories.head()
categories['split'] = categories['item_category_name'].str.split('-')

categories['type'] = categories['split'].map(lambda x:x[0].strip())

categories['subtype'] = categories['split'].map(lambda x:x[1].strip() if len(x)>1 else x[0].strip())

categories = categories[['item_category_id','type','subtype']]

categories.head()

categories['cat_type_code'] = LabelEncoder().fit_transform(categories['type'])

categories['cat_subtype_code'] = LabelEncoder().fit_transform(categories['subtype'])

categories.head()

#Month_revenue

ts = time.time()

matrix = []

cols = ['date_block_num','shop_id','item_id']

for i in range(34):

    sales = sales_train[sales_train.date_block_num==i]

    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))

    

matrix = pd.DataFrame(np.vstack(matrix), columns=cols)

matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)

matrix['shop_id'] = matrix['shop_id'].astype(np.int8)

matrix['item_id'] = matrix['item_id'].astype(np.int16)

matrix.sort_values(cols,inplace=True)

time.time() - ts





groupby = sales_train.groupby(['item_id','shop_id','date_block_num']).agg({'item_cnt_day':'sum'})

groupby.columns = ['item_cnt_month']

groupby.reset_index(inplace=True)

matrix = matrix.merge(groupby, on = ['item_id','shop_id','date_block_num'], how = 'left')

matrix['item_cnt_month'] = matrix['item_cnt_month'].fillna(0).clip(0,20).astype(np.float16)

matrix.head()



sales_train['revenue'] = sales_train['item_price'] *  sales_train['item_cnt_day']

monthrevenue = sales_train.groupby(['item_id','shop_id','date_block_num']).agg({'revenue':'sum'})

monthrevenue.columns = ['month_revenue']

monthrevenue.reset_index(inplace=True)

matrix = matrix.merge(monthrevenue, on = ['item_id','shop_id','date_block_num'], how = 'left')

matrix.head()



test['date_block_num'] = 34

test['date_block_num'] = test['date_block_num'].astype(np.int8)

test['shop_id'] = test['shop_id'].astype(np.int8)

test['item_id'] = test['item_id'].astype(np.int16)

test.shape



cols = ['date_block_num','shop_id','item_id']

matrix = pd.concat([matrix, test[['item_id','shop_id','date_block_num']]], ignore_index=True, sort=False, keys=cols)

matrix.fillna(0, inplace=True) # 34 month

matrix.head()
matrix.head(10)
ts = time.time()

matrix = matrix.merge(items[['item_id','item_category_id']], on = ['item_id'], how = 'left')

matrix = matrix.merge(categories[['item_category_id','cat_type_code','cat_subtype_code']], on = ['item_category_id'], how = 'left')

matrix = matrix.merge(shops[['shop_id','shop_city_code','shop_type_code']], on = ['shop_id'], how = 'left')

matrix['shop_city_code'] = matrix['shop_city_code'].astype(np.int8)

matrix['shop_type_code'] = matrix['shop_type_code'].astype(np.int8)

matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)

matrix['cat_type_code'] = matrix['cat_type_code'].astype(np.int8)

matrix['cat_subtype_code'] = matrix['cat_subtype_code'].astype(np.int8)

time.time() - ts

matrix.head()
matrix['item_price_unit'] = matrix['month_revenue'] // matrix['item_cnt_month']

matrix['item_price_unit'].fillna(0, inplace=True)



gp_item_price = sales_train.sort_values('date_block_num').groupby(['item_id'], as_index=False).agg({'item_price':[np.min, np.max]})

gp_item_price.columns = ['item_id', 'hist_min_item_price', 'hist_max_item_price']

 

matrix = pd.merge(matrix, gp_item_price, on='item_id', how='left')



matrix.head()
lag_list = [1, 2, 3]

 

for lag in lag_list:

    ft_name = ('item_price_unit_shifted%s' % lag)

    matrix[ft_name] = matrix.sort_values('date_block_num').groupby(['shop_id', 'item_category_id', 'item_id'])['item_price_unit'].shift(lag)

    # Fill the empty shifted features with 0

    matrix[ft_name].fillna(0, inplace=True)

    

    matrix.head()
lag_list = [1, 2, 3]

 

for lag in lag_list:

    ft_name = ('month_revenue_shifted%s' % lag)

    matrix[ft_name] = matrix.sort_values('date_block_num').groupby(['shop_id', 'item_category_id', 'item_id'])['month_revenue'].shift(lag)

    # Fill the empty shifted features with 0

    matrix[ft_name].fillna(0, inplace=True)
matrix.head()
matrix['year'] = matrix['date_block_num'].apply(lambda x: ((x//12) + 2013))

matrix['month'] = matrix['date_block_num'].apply(lambda x: (x % 12))

matrix['month'] = matrix['date_block_num'] % 12

days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])

matrix['days'] = matrix['month'].map(days).astype(np.int8)

holidays = pd.Series([8,3,3,0,9,3,0,0,0,0,1,0])

matrix['holidays'] = matrix['month'].map(holidays).astype(np.int8)

matrix.tail()
matrix.tail()


matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')

matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')

matrix.head()
X_train = matrix.query('date_block_num >= 3 and date_block_num < 28').copy().drop(['item_cnt_month'], axis=1)

Y_train = matrix.query('date_block_num >= 3 and date_block_num < 28').copy()['item_cnt_month']

X_valid = matrix.query('date_block_num >= 28 and date_block_num < 33').copy().drop(['item_cnt_month'], axis=1)

Y_valid = matrix.query('date_block_num >= 28 and date_block_num < 33').copy()['item_cnt_month']

X_test = matrix[matrix.date_block_num == 33].drop(['item_cnt_month'], axis=1)