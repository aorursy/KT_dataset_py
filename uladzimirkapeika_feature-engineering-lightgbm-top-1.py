import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from itertools import product

from xgboost import XGBRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb

import calendar

from datetime import datetime

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

sales = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

item_cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
sns.boxplot(x=sales.item_cnt_day)
sns.boxplot(x=sales.item_price)
train = sales[(sales.item_price < 100000) & (sales.item_price > 0)]

train = train[sales.item_cnt_day < 1001]
print(shops[shops.shop_id.isin([0, 57])]['shop_name'])

print(shops[shops.shop_id.isin([1, 58])]['shop_name'])

print(shops[shops.shop_id.isin([40, 39])]['shop_name'])
train.loc[train.shop_id == 0, 'shop_id'] = 57

test.loc[test.shop_id == 0, 'shop_id'] = 57



train.loc[train.shop_id == 1, 'shop_id'] = 58

test.loc[test.shop_id == 1, 'shop_id'] = 58



train.loc[train.shop_id == 40, 'shop_id'] = 39

test.loc[test.shop_id == 40, 'shop_id'] = 39
index_cols = ['shop_id', 'item_id', 'date_block_num']



df = [] 

for block_num in train['date_block_num'].unique():

    cur_shops = train.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()

    cur_items = train.loc[sales['date_block_num'] == block_num, 'item_id'].unique()

    df.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))



df = pd.DataFrame(np.vstack(df), columns = index_cols,dtype=np.int32)



#Add month sales

group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})

group.columns = ['item_cnt_month']

group.reset_index(inplace=True)



df = pd.merge(df, group, on=index_cols, how='left')

df['item_cnt_month'] = (df['item_cnt_month']

                                .fillna(0)

                                .clip(0,20)

                                .astype(np.float16))

df.head(5)
test['date_block_num'] = 34

test['date_block_num'] = test['date_block_num'].astype(np.int8)

test['shop_id'] = test['shop_id'].astype(np.int8)

test['item_id'] = test['item_id'].astype(np.int16)

df = pd.concat([df, test], ignore_index=True, sort=False, keys=index_cols)

df.fillna(0, inplace=True)
shops['city'] = shops['shop_name'].apply(lambda x: x.split()[0].lower())

shops.loc[shops.city == '!????????????', 'city'] = '????????????'

shops['city_code'] = LabelEncoder().fit_transform(shops['city'])



coords = dict()

coords['????????????'] = (62.028098, 129.732555, 4)

coords['????????????'] = (44.609764, 40.100516, 3)

coords['????????????????'] = (55.8094500, 37.9580600, 1)

coords['????????????????'] = (53.4305800, 50.1190000, 3)

coords['??????????????'] = (59.2239000, 39.8839800, 2)

coords['??????????????'] = (51.6720400, 39.1843000, 3)

coords['????????????????'] = (0, 0, 0)

coords['??????????????????'] = (55.5952800, 38.1202800, 1)

coords['????????????????-??????????????'] = (0, 0, 0)

coords['????????????'] = (55.7887400, 49.1221400, 4)

coords['????????????'] = (54.5293000, 36.2754200, 4)

coords['??????????????'] = (55.0794400, 38.7783300, 4)

coords['????????????????????'] = (56.0183900, 92.8671700, 4)

coords['??????????'] = (51.7373300, 36.1873500, 3)

coords['????????????'] = (55.7522200, 37.6155600, 1)

coords['????????????'] = (55.9116300, 37.7307600, 1)

coords['??.????????????????'] = (56.3286700, 44.0020500, 4)

coords['??????????????????????'] = (55.0415000, 82.9346000, 4)

coords['????????'] = (54.9924400, 73.3685900, 4)

coords['????????????????????????'] = (47.2313500, 39.7232800, 3)

coords['??????'] = (59.9386300, 30.3141300, 2)

coords['????????????'] = (53.2000700, 50.1500000, 4)

coords['??????????????'] = (56.3000000, 38.1333300, 4)

coords['????????????'] = (61.2500000, 73.4166700, 4)

coords['??????????'] = (56.4977100, 84.9743700, 4)

coords['????????????'] = (57.1522200, 65.5272200, 4)

coords['??????'] = (54.7430600, 55.9677900, 4)

coords['??????????'] = (55.8970400, 37.4296900, 1)

coords['????????????????'] = (0, 0, 0)

coords['??????????'] = (55.1477000, 37.4772800, 4)

coords['??????????????????'] = (57.6298700, 39.8736800, 2) 



shops['city_coord_1'] = shops['city'].apply(lambda x: coords[x][0])

shops['city_coord_2'] = shops['city'].apply(lambda x: coords[x][1])

shops['country_part'] = shops['city'].apply(lambda x: coords[x][2])



shops = shops[['shop_id', 'city_code', 'city_coord_1', 'city_coord_2', 'country_part']]
df = pd.merge(df, shops, on=['shop_id'], how='left')
map_dict = {

            '???????????? ???????????????? (??????????????)': '???????????? ????????????????',

            '???????????? ???????????????? (??????????)' : '???????????? ????????????????',

            'PC ': '????????????????????',

            '??????????????????': '?????????????????? '

            }



items = pd.merge(items, item_cats, on='item_category_id')



items['item_category'] = items['item_category_name'].apply(lambda x: x.split('-')[0])

items['item_category'] = items['item_category'].apply(lambda x: map_dict[x] if x in map_dict.keys() else x)

items['item_category_common'] = LabelEncoder().fit_transform(items['item_category'])



items['item_category_code'] = LabelEncoder().fit_transform(items['item_category_name'])

items = items[['item_id', 'item_category_common', 'item_category_code']]
df = pd.merge(df, items, on=['item_id'], how='left')
def count_days(date_block_num):

    year = 2013 + date_block_num // 12

    month = 1 + date_block_num % 12

    weeknd_count = len([1 for i in calendar.monthcalendar(year, month) if i[6] != 0])

    days_in_month = calendar.monthrange(year, month)[1]

    return weeknd_count, days_in_month, month



map_dict = {i: count_days(i) for i in range(35)}



df['weeknd_count'] = df['date_block_num'].apply(lambda x: map_dict[x][0])

df['days_in_month'] = df['date_block_num'].apply(lambda x: map_dict[x][1])
first_item_block = df.groupby(['item_id'])['date_block_num'].min().reset_index()

first_item_block['item_first_interaction'] = 1



first_shop_item_buy_block = df[df['date_block_num'] > 0].groupby(['shop_id', 'item_id'])['date_block_num'].min().reset_index()

first_shop_item_buy_block['first_date_block_num'] = first_shop_item_buy_block['date_block_num']
df = pd.merge(df, first_item_block[['item_id', 'date_block_num', 'item_first_interaction']], on=['item_id', 'date_block_num'], how='left')

df = pd.merge(df, first_shop_item_buy_block[['item_id', 'shop_id', 'first_date_block_num']], on=['item_id', 'shop_id'], how='left')



df['first_date_block_num'].fillna(100, inplace=True)

df['shop_item_sold_before'] = (df['first_date_block_num'] < df['date_block_num']).astype('int8')

df.drop(['first_date_block_num'], axis=1, inplace=True)



df['item_first_interaction'].fillna(0, inplace=True)

df['shop_item_sold_before'].fillna(0, inplace=True)

 

df['item_first_interaction'] = df['item_first_interaction'].astype('int8')  

df['shop_item_sold_before'] = df['shop_item_sold_before'].astype('int8') 
def lag_feature(df, lags, col):

    tmp = df[['date_block_num','shop_id','item_id',col]]

    for i in lags:

        shifted = tmp.copy()

        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]

        shifted['date_block_num'] += i

        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')

        df[col+'_lag_'+str(i)] = df[col+'_lag_'+str(i)].astype('float16')

    return df
#Add sales lags for last 3 months

df = lag_feature(df, [1, 2, 3], 'item_cnt_month')
#Add avg shop/item price



index_cols = ['shop_id', 'item_id', 'date_block_num']

group = train.groupby(index_cols)['item_price'].mean().reset_index().rename(columns={"item_price": "avg_shop_price"}, errors="raise")

df = pd.merge(df, group, on=index_cols, how='left')



df['avg_shop_price'] = (df['avg_shop_price']

                                .fillna(0)

                                .astype(np.float16))



index_cols = ['item_id', 'date_block_num']

group = train.groupby(['date_block_num','item_id'])['item_price'].mean().reset_index().rename(columns={"item_price": "avg_item_price"}, errors="raise")





df = pd.merge(df, group, on=index_cols, how='left')

df['avg_item_price'] = (df['avg_item_price']

                                .fillna(0)

                                .astype(np.float16))



df['item_shop_price_avg'] = (df['avg_shop_price'] - df['avg_item_price']) / df['avg_item_price']

df['item_shop_price_avg'].fillna(0, inplace=True)



df = lag_feature(df, [1, 2, 3], 'item_shop_price_avg')

df.drop(['avg_shop_price', 'avg_item_price', 'item_shop_price_avg'], axis=1, inplace=True)
#Add target encoding for items for last 3 months 

item_id_target_mean = df.groupby(['date_block_num','item_id'])['item_cnt_month'].mean().reset_index().rename(columns={"item_cnt_month": "item_target_enc"}, errors="raise")

df = pd.merge(df, item_id_target_mean, on=['date_block_num','item_id'], how='left')



df['item_target_enc'] = (df['item_target_enc']

                                .fillna(0)

                                .astype(np.float16))



df = lag_feature(df, [1, 2, 3], 'item_target_enc')

df.drop(['item_target_enc'], axis=1, inplace=True)
#Add target encoding for item/city for last 3 months 

item_id_target_mean = df.groupby(['date_block_num','item_id', 'city_code'])['item_cnt_month'].mean().reset_index().rename(columns={

    "item_cnt_month": "item_loc_target_enc"}, errors="raise")

df = pd.merge(df, item_id_target_mean, on=['date_block_num','item_id', 'city_code'], how='left')



df['item_loc_target_enc'] = (df['item_loc_target_enc']

                                .fillna(0)

                                .astype(np.float16))



df = lag_feature(df, [1, 2, 3], 'item_loc_target_enc')

df.drop(['item_loc_target_enc'], axis=1, inplace=True)
#Add target encoding for item/shop for last 3 months 

item_id_target_mean = df.groupby(['date_block_num','item_id', 'shop_id'])['item_cnt_month'].mean().reset_index().rename(columns={

    "item_cnt_month": "item_shop_target_enc"}, errors="raise")



df = pd.merge(df, item_id_target_mean, on=['date_block_num','item_id', 'shop_id'], how='left')



df['item_shop_target_enc'] = (df['item_shop_target_enc']

                                .fillna(0)

                                .astype(np.float16))



df = lag_feature(df, [1, 2, 3], 'item_shop_target_enc')

df.drop(['item_shop_target_enc'], axis=1, inplace=True)
#For new items add avg category sales for last 3 months

item_id_target_mean = df[df['item_first_interaction'] == 1].groupby(['date_block_num','item_category_code'])['item_cnt_month'].mean().reset_index().rename(columns={

    "item_cnt_month": "new_item_cat_avg"}, errors="raise")



df = pd.merge(df, item_id_target_mean, on=['date_block_num','item_category_code'], how='left')



df['new_item_cat_avg'] = (df['new_item_cat_avg']

                                .fillna(0)

                                .astype(np.float16))



df = lag_feature(df, [1, 2, 3], 'new_item_cat_avg')

df.drop(['new_item_cat_avg'], axis=1, inplace=True)
#For new items add avg category sales in a separate store for last 3 months

item_id_target_mean = df[df['item_first_interaction'] == 1].groupby(['date_block_num','item_category_code', 'shop_id'])['item_cnt_month'].mean().reset_index().rename(columns={

    "item_cnt_month": "new_item_shop_cat_avg"}, errors="raise")



df = pd.merge(df, item_id_target_mean, on=['date_block_num','item_category_code', 'shop_id'], how='left')



df['new_item_shop_cat_avg'] = (df['new_item_shop_cat_avg']

                                .fillna(0)

                                .astype(np.float16))



df = lag_feature(df, [1, 2, 3], 'new_item_shop_cat_avg')

df.drop(['new_item_shop_cat_avg'], axis=1, inplace=True)
def lag_feature_adv(df, lags, col):

    tmp = df[['date_block_num','shop_id','item_id',col]]

    for i in lags:

        shifted = tmp.copy()

        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)+'_adv']

        shifted['date_block_num'] += i

        shifted['item_id'] -= 1

        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')

        df[col+'_lag_'+str(i)+'_adv'] = df[col+'_lag_'+str(i)+'_adv'].astype('float16')

    return df



df = lag_feature_adv(df, [1, 2, 3], 'item_cnt_month')
df.fillna(0, inplace=True)

df = df[(df['date_block_num'] > 2)]

df.head()
df.columns
#Save dataset

df.drop(['ID'], axis=1, inplace=True, errors='ignore')

df.to_pickle('df.pkl')
df = pd.read_pickle('df.pkl')

df.info()
X_train = df[df.date_block_num < 33].drop(['item_cnt_month'], axis=1)

Y_train = df[df.date_block_num < 33]['item_cnt_month']

X_valid = df[df.date_block_num == 33].drop(['item_cnt_month'], axis=1)

Y_valid = df[df.date_block_num == 33]['item_cnt_month']

X_test = df[df.date_block_num == 34].drop(['item_cnt_month'], axis=1)

del df
feature_name = X_train.columns.tolist()



params = {

    'objective': 'mse',

    'metric': 'rmse',

    'num_leaves': 2 ** 7 - 1,

    'learning_rate': 0.005,

    'feature_fraction': 0.75,

    'bagging_fraction': 0.75,

    'bagging_freq': 5,

    'seed': 1,

    'verbose': 1

}



feature_name_indexes = [ 

                        'country_part', 

                        'item_category_common',

                        'item_category_code', 

                        'city_code',

]



lgb_train = lgb.Dataset(X_train[feature_name], Y_train)

lgb_eval = lgb.Dataset(X_valid[feature_name], Y_valid, reference=lgb_train)



evals_result = {}

gbm = lgb.train(

        params, 

        lgb_train,

        num_boost_round=3000,

        valid_sets=(lgb_train, lgb_eval), 

        feature_name = feature_name,

        categorical_feature = feature_name_indexes,

        verbose_eval=5, 

        evals_result = evals_result,

        early_stopping_rounds = 100)

lgb.plot_importance(

    gbm, 

    max_num_features=50, 

    importance_type='gain', 

    figsize=(12,8));
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

Y_test = gbm.predict(X_test[feature_name]).clip(0, 20)



submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": Y_test

})

submission.to_csv('gbm_submission.csv', index=False)