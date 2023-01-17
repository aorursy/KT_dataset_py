import numpy as np

from numpy.random import seed

from numpy.random import randn



import pandas as pd



from sklearn import linear_model

from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.graphics.gofplots import qqplot

import statsmodels.api as sm



import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('seaborn-whitegrid')



import seaborn as sns

sns.set(style="whitegrid")



import math

get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')



import csv

import time



import scipy.stats as stats









items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

test=pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")

items_categories=pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

sample_submission=pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")





items.head(10)
sales_train.info()
test.info()
shops.info()
sales_train[sales_train.isnull().any(axis=1)].head() 
test[test.isnull().any(axis=1)].head(10)
sample_submission[sample_submission.isnull().any(axis=1)].head()
plt.plot(sales_train['item_id'], sales_train['item_price'], 'o', color='blue');
sales_train[sales_train.item_price > 250000]
items_categories[items_categories.item_category_id == 65]
shops[shops.shop_id == 12]
sales_train_sub = sales_train

sales_train_sub['date'] =  pd.to_datetime(sales_train_sub['date'],format= '%d.%m.%Y')

sales_train_sub['month'] = pd.DatetimeIndex(sales_train_sub['date']).month

sales_train_sub['year'] = pd.DatetimeIndex(sales_train_sub['date']).year

sales_train_sub = sales_train_sub.iloc[:,1:8]

sales_train_sub.head(10)

items
import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 100)



from itertools import product

from sklearn.preprocessing import LabelEncoder



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import time

import sys

import gc

import pickle

sys.version_info
sales_train.head()
print('train size, item in train, shop in train', sales_train.shape[0], sales_train.item_id.nunique(), sales_train.shop_id.nunique())

print('train size, item in train, shop in train', test.shape[0], test.item_id.nunique(),test.shop_id.nunique())

print('new items:', len(list(set(test.item_id) - set(test.item_id).intersection(set(sales_train.item_id)))), len(list(set(test.item_id))), len(test))
sales_train.isnull().sum()
sale_by_month = sales_train.groupby('date_block_num')['item_cnt_day'].sum()

sale_by_month.plot()
block_item_shop_sale = sales_train.groupby(['date_block_num','item_id','shop_id'])['item_cnt_day'].sum()

block_item_shop_sale.clip(0,20).plot.hist(bins=20)
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


sales_train.loc[sales_train.shop_id == 0, 'shop_id'] = 57

test.loc[test.shop_id == 0, 'shop_id'] = 57



sales_train.loc[sales_train.shop_id == 1, 'shop_id'] = 58

test.loc[test.shop_id == 1, 'shop_id'] = 58



sales_train.loc[sales_train.shop_id == 10, 'shop_id'] = 11

test.loc[test.shop_id == 10, 'shop_id'] = 11
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'

shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])

shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'

shops['city_code'] = LabelEncoder().fit_transform(shops['city'])

shops = shops[['shop_id','city_code']]



items_categories['split'] = items_categories['item_category_name'].str.split('-')

items_categories['type'] = items_categories['split'].map(lambda x: x[0].strip())

items_categories['type_code'] = LabelEncoder().fit_transform(items_categories['type'])

# if subtype is nan then type

items_categories['subtype'] = items_categories['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

items_categories['subtype_code'] = LabelEncoder().fit_transform(items_categories['subtype'])

items_categories =items_categories[['item_category_id','type_code', 'subtype_code']]



items.drop(['item_name'], axis=1, inplace=True)
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
sales_train['revenue'] = sales_train['item_price'] *  sales_train['item_cnt_day']
ts = time.time()

group = sales_train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})

group.columns = ['item_cnt_month']

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on=cols, how='left')

matrix['item_cnt_month'] = (matrix['item_cnt_month']

                                .fillna(0)

                                .clip(0,20) # NB clip target here

                                .astype(np.float32))

time.time() - ts
test['date_block_num'] = 36

test['date_block_num'] = test['date_block_num'].astype(np.int8)

test['shop_id'] = test['shop_id'].astype(np.int8)

test['item_id'] = test['item_id'].astype(np.int16)
ts = time.time()

matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)

matrix.fillna(0, inplace=True) # 36 ay

time.time() - ts
ts = time.time()

matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')

matrix = pd.merge(matrix, items, on=['item_id'], how='left')

matrix = pd.merge(matrix, items_categories, on=['item_category_id'], how='left')

matrix['city_code'] = matrix['city_code'].astype(np.int8)

matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)

matrix['type_code'] = matrix['type_code'].astype(np.int8)

matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)

time.time() - ts
def lag_feature(df, lags, col):

    tmp = df[['date_block_num','shop_id','item_id',col]]

    for i in lags:

        shifted = tmp.copy()

        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]

        shifted['date_block_num'] += i

        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')

    return df
ts = time.time()

matrix = lag_feature(matrix, [1,2,3,6,12], 'item_cnt_month')

time.time() - ts


def add_group_stats(matrix_, groupby_feats, target, enc_feat, last_periods):

    if not 'date_block_num' in groupby_feats:

        print ('date_block_num must in groupby_feats')

        return matrix_

    

    group = matrix_.groupby(groupby_feats)[target].sum().reset_index()

    max_lags = np.max(last_periods)

    for i in range(1,max_lags+1):

        shifted = group[groupby_feats+[target]].copy(deep=True)

        shifted['date_block_num'] += i

        shifted.rename({target:target+'_lag_'+str(i)},axis=1,inplace=True)

        group = group.merge(shifted, on=groupby_feats, how='left')

    group.fillna(0,inplace=True)

    for period in last_periods:

        lag_feats = [target+'_lag_'+str(lag) for lag in np.arange(1,period+1)]

        # we do not use mean and svd directly because we want to include months with sales = 0

        mean = group[lag_feats].sum(axis=1)/float(period)

        mean2 = (group[lag_feats]**2).sum(axis=1)/float(period)

        group[enc_feat+'_avg_sale_last_'+str(period)] = mean

        group[enc_feat+'_std_sale_last_'+str(period)] = (mean2 - mean**2).apply(np.sqrt)

        group[enc_feat+'_std_sale_last_'+str(period)].replace(np.inf,0,inplace=True)

        # divide by mean, this scales the features for NN

        group[enc_feat+'_avg_sale_last_'+str(period)] /= group[enc_feat+'_avg_sale_last_'+str(period)].mean()

        group[enc_feat+'_std_sale_last_'+str(period)] /= group[enc_feat+'_std_sale_last_'+str(period)].mean()

    cols = groupby_feats + [f_ for f_ in group.columns.values if f_.find('_sale_last_')>=0]

    matrix = matrix_.merge(group[cols], on=groupby_feats, how='left')

    return matrix
ts = time.time()

matrix = add_group_stats(matrix, ['date_block_num', 'item_id'], 'item_cnt_month', 'item', [6,12])

matrix = add_group_stats(matrix, ['date_block_num', 'shop_id'], 'item_cnt_month', 'shop', [6,12])

matrix = add_group_stats(matrix, ['date_block_num', 'item_category_id'], 'item_cnt_month', 'category', [12])

matrix = add_group_stats(matrix, ['date_block_num', 'city_code'], 'item_cnt_month', 'city', [12])

matrix = add_group_stats(matrix, ['date_block_num', 'type_code'], 'item_cnt_month', 'type', [12])

matrix = add_group_stats(matrix, ['date_block_num', 'subtype_code'], 'item_cnt_month', 'subtype', [12])

time.time() - ts
#first use target encoding each group, then shift month to creat lag features

def target_encoding(matrix_, groupby_feats, target, enc_feat, lags):

    print ('target encoding for',groupby_feats)

    group = matrix_.groupby(groupby_feats).agg({target:'mean'})

    group.columns = [enc_feat]

    group.reset_index(inplace=True)

    matrix = matrix_.merge(group, on=groupby_feats, how='left')

    matrix[enc_feat] = matrix[enc_feat].astype(np.float16)

    matrix = lag_feature(matrix, lags, enc_feat)

    matrix.drop(enc_feat, axis=1, inplace=True)

    return matrix
ts = time.time()

matrix = target_encoding(matrix, ['date_block_num'], 'item_cnt_month', 'date_avg_item_cnt', [1])

matrix = target_encoding(matrix, ['date_block_num', 'item_id'], 'item_cnt_month', 'date_item_avg_item_cnt', [1,2,3,6,12])

matrix = target_encoding(matrix, ['date_block_num', 'shop_id'], 'item_cnt_month', 'date_shop_avg_item_cnt', [1,2,3,6,12])

matrix = target_encoding(matrix, ['date_block_num', 'item_category_id'], 'item_cnt_month', 'date_cat_avg_item_cnt', [1])

matrix = target_encoding(matrix, ['date_block_num', 'shop_id', 'item_category_id'], 'item_cnt_month', 'date_shop_cat_avg_item_cnt', [1])

matrix = target_encoding(matrix, ['date_block_num', 'city_code'], 'item_cnt_month', 'date_city_avg_item_cnt', [1])

matrix = target_encoding(matrix, ['date_block_num', 'item_id', 'city_code'], 'item_cnt_month', 'date_item_city_avg_item_cnt', [1])

time.time() - ts
ts = time.time()

group = sales_train.groupby(['date_block_num','shop_id']).agg({'revenue': ['sum']})

group.columns = ['date_shop_revenue']

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')

matrix['date_shop_revenue'] = matrix['date_shop_revenue'].astype(np.float32)



group = group.groupby(['shop_id']).agg({'date_shop_revenue': ['mean']})

group.columns = ['shop_avg_revenue']

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on=['shop_id'], how='left')

matrix['shop_avg_revenue'] = matrix['shop_avg_revenue'].astype(np.float32)



matrix['delta_revenue'] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']

matrix['delta_revenue'] = matrix['delta_revenue'].astype(np.float16)



matrix = lag_feature(matrix, [1], 'delta_revenue')



matrix.drop(['date_shop_revenue','shop_avg_revenue','delta_revenue'], axis=1, inplace=True)

time.time() - ts
matrix['month'] = matrix['date_block_num'] % 12

matrix['year'] = (matrix['date_block_num'] / 12).astype(np.int8)
#Month since last sale for each shop/item pair.

ts = time.time()

last_sale = pd.DataFrame()

for month in range(1,35):    

    last_month = matrix.loc[(matrix['date_block_num']<month)&(matrix['item_cnt_month']>0)].groupby(['item_id','shop_id'])['date_block_num'].max()

    df = pd.DataFrame({'date_block_num':np.ones([last_month.shape[0],])*month,

                       'item_id': last_month.index.get_level_values(0).values,

                       'shop_id': last_month.index.get_level_values(1).values,

                       'item_shop_last_sale': last_month.values})

    last_sale = last_sale.append(df)

last_sale['date_block_num'] = last_sale['date_block_num'].astype(np.int8)



matrix = matrix.merge(last_sale, on=['date_block_num','item_id','shop_id'], how='left')

time.time() - ts
ts = time.time()

last_sale = pd.DataFrame()

for month in range(1,35):    

    last_month = matrix.loc[(matrix['date_block_num']<month)&(matrix['item_cnt_month']>0)].groupby('item_id')['date_block_num'].max()

    df = pd.DataFrame({'date_block_num':np.ones([last_month.shape[0],])*month,

                       'item_id': last_month.index.values,

                       'item_last_sale': last_month.values})

    last_sale = last_sale.append(df)

last_sale['date_block_num'] = last_sale['date_block_num'].astype(np.int8)



matrix = matrix.merge(last_sale, on=['date_block_num','item_id'], how='left')

time.time() - ts
ts = time.time()

matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')

matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')

time.time() - ts