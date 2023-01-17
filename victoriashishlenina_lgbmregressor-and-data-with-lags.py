import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime

import copy

import matplotlib as mpl

from statsmodels.tsa.seasonal import seasonal_decompose

from dateutil.parser import parse

import statsmodels.api as sm

from sklearn.metrics import mean_squared_error

from math import sqrt

import collections

from sklearn.model_selection import (

    train_test_split,

    cross_val_score

)

from xgboost import XGBRegressor

from xgboost import plot_importance

from lightgbm import LGBMRegressor

from lightgbm import plot_importance

from sklearn.metrics import mean_squared_error

from math import sqrt
train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")

sample_submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")
# let's do the date column in correct format

train['date']=train['date'].apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))
print('train:',train.shape,'test:',test.shape,'items:',items.shape,'item_categories:',item_categories.shape,'shop:',shops.shape)
#add information about category

train = train.join(items, on='item_id', rsuffix='_').drop(['item_id_', 'item_name'], axis=1)
train.head()
train[train['item_price']<=0]
train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)]
train.loc[train.item_price<0,'item_price'] = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)].item_price.mean()
train_monthly = train.sort_values('date').groupby(['date_block_num', 'shop_id','item_category_id', 'item_id'], as_index=False)
train_monthly = train_monthly.agg({'item_price':['median', 'mean'], 'item_cnt_day':['sum', 'count']})
train_monthly.head(3)
train_monthly.columns = ['date_block_num', 'shop_id', 'item_category_id','item_id', 'item_price_median', 'item_price_mean', 'item_cnt', 'transactions']
train_monthly.head(3)
train_monthly_by_category = train_monthly.groupby(['date_block_num','shop_id', 'item_category_id'], as_index=False)
train_monthly_by_category = train_monthly_by_category.agg({'item_price_median':['mean'], 'item_price_mean':['mean'],'item_cnt':['sum', 'mean'], 'transactions': ['mean']})
train_monthly_by_category.head()
train_cat_no_shop = train_monthly.groupby(['date_block_num', 'item_category_id'], as_index=False)
train_cat_no_shop = train_cat_no_shop.agg({'item_price_median':['mean'], 'item_price_mean':['mean'],'item_cnt':['sum', 'mean'], 'transactions': ['mean']})
train_cat_no_shop.head(3)
train_monthly['year'] = train_monthly['date_block_num'].apply(lambda x: ((x//12+2013)))

train_monthly['month'] = train_monthly['date_block_num'].apply(lambda x: (x%12+1))
train_monthly_by_category['year'] = train_monthly_by_category['date_block_num'].apply(lambda x: ((x//12+2013)))

train_monthly_by_category['month'] = train_monthly_by_category['date_block_num'].apply(lambda x: (x%12+1))
train_cat_no_shop['year'] = train_cat_no_shop['date_block_num'].apply(lambda x: ((x//12+2013)))

train_cat_no_shop['month'] = train_cat_no_shop['date_block_num'].apply(lambda x: (x%12+1))
train_monthly_by_category.columns = ['date_block_num', 'shop_id','item_category_id','item_price_median', 'item_price_mean', 'item_cnt_sum', 'item_cnt', 'transactions', 'year', 'month']

train_cat_no_shop.columns = ['date_block_num', 'item_category_id','item_price_median', 'item_price_mean', 'item_cnt_sum', 'item_cnt', 'transactions', 'year', 'month']
train_monthly.head(3)
train_cat_no_shop.head(3)
train_monthly_by_category.head(3)
# train_monthly.query('item_cnt >= 0 and item_cnt <= 20')

# train_monthly.query('item_cnt <= 0 or item_cnt >= 20')
train_monthly.shape
train_monthly[train_monthly['item_cnt']>1000]
uniq_pairs = train_monthly.groupby(['shop_id','item_id']).size().reset_index()

uniq_pairs.shape
empty_df = pd.DataFrame(index = uniq_pairs.index, columns = ['date_block_num','shop_id','item_id'])

empty_df[['shop_id', 'item_id']] = uniq_pairs[['shop_id','item_id']]

empty_df_2 = pd.DataFrame(columns = ['date_block_num','shop_id','item_id'])

for i in range(34):

    empty_df_1 = empty_df.copy()

    empty_df_1['date_block_num'] = i

    empty_df_2 = pd.concat([empty_df_2, empty_df_1])
empty_df_2 = empty_df_2.reset_index()
empty_df_2.head()
full_train_monthly = empty_df_2.merge(train_monthly,on=['date_block_num','shop_id', 'item_id'], how='left').fillna(0).drop(['index'], axis=1)
full_train_monthly['year'] = full_train_monthly['date_block_num'].apply(lambda x: ((x//12+2013)))

full_train_monthly['month'] = full_train_monthly['date_block_num'].apply(lambda x: (x%12+1))
full_train_monthly = full_train_monthly.join(items, on='item_id', rsuffix='_').drop(['item_id_', 'item_name', 'item_category_id'], axis=1)

full_train_monthly = full_train_monthly.rename(columns={'item_category_id_':'item_category_id'})
full_train_monthly.shape
print('Table by months shape:', train_monthly.shape, 'Full table by months shape:', full_train_monthly.shape)
uniq_pairs_cat = train_monthly_by_category.groupby(['shop_id','item_category_id']).size().reset_index()
empty_df = pd.DataFrame(index = uniq_pairs_cat.index, columns = ['date_block_num','shop_id','item_category_id'])

empty_df[['shop_id', 'item_category_id']] = uniq_pairs_cat[['shop_id','item_category_id']]

empty_df_2 = pd.DataFrame(columns = ['date_block_num','shop_id','item_category_id'])

for i in range(34):

    empty_df_1 = empty_df.copy()

    empty_df_1['date_block_num'] = i

    empty_df_2 = pd.concat([empty_df_2, empty_df_1])
empty_df_2 = empty_df_2.reset_index()
# print('Table by categories shape:', train_monthly_by_category.shape, 'Full table by categories shape:', empty_df_2.shape)
full_train_monthly_by_category = empty_df_2.merge(train_monthly_by_category, 

                                      on=['date_block_num','shop_id', 'item_category_id'],how='left').fillna(0).drop(['index'], axis=1)
full_train_monthly_by_category.head(3)
uniq_cat = train_cat_no_shop['item_category_id'].unique()
empty_df = pd.DataFrame(index = uniq_cat, columns = ['date_block_num','item_category_id'])

empty_df['item_category_id'] = uniq_cat

empty_df_2 = pd.DataFrame(columns = ['date_block_num','item_category_id'])

for i in range(34):

    empty_df_1 = empty_df.copy()

    empty_df_1['date_block_num'] = i

    empty_df_2 = pd.concat([empty_df_2, empty_df_1])
empty_df_2 = empty_df_2.reset_index()
full_train_cat_no_shop = empty_df_2.merge(train_cat_no_shop, 

                                      on=['date_block_num', 'item_category_id'],how='left').fillna(0).drop(['index'], axis=1)
full_train_cat_no_shop.head()
full_train_monthly_by_category['year'] = full_train_monthly_by_category['date_block_num'].apply(lambda x: ((x//12+2013)))

full_train_monthly_by_category['month'] = full_train_monthly_by_category['date_block_num'].apply(lambda x: (x%12+1))

full_train_cat_no_shop['year'] = full_train_cat_no_shop['date_block_num'].apply(lambda x: ((x//12+2013)))

full_train_cat_no_shop['month'] = full_train_cat_no_shop['date_block_num'].apply(lambda x: (x%12+1))
full_train_monthly_by_category.head(2)
full_train_monthly['item_cnt_next_month'] = full_train_monthly.sort_values('date_block_num').groupby(['shop_id','item_id'])['item_cnt'].shift(-1)
full_train_monthly.head(3)
# full_train_monthly[(full_train_monthly.shop_id==0)&(full_train_monthly.item_id==5572)]
full_train_monthly_by_category['item_cnt_next_month'] = full_train_monthly_by_category.sort_values('date_block_num').groupby(['shop_id', 'item_category_id'])['item_cnt'].shift(-1)
full_train_cat_no_shop['item_cnt_next_month'] = full_train_cat_no_shop.sort_values('date_block_num').groupby(['item_category_id'])['item_cnt'].shift(-1)
lag_list = [1,2,3,4,5,6,11,23]



for lag in lag_list:

    ft_name = ('item_cnt_shifted%s' % lag)

    full_train_monthly[ft_name] = full_train_monthly.sort_values('date_block_num').groupby(['shop_id', 'item_id'])['item_cnt'].shift(lag)
full_train_monthly.head()
lag_list = [1,2,3,4,5,6,11,23]



for lag in lag_list:

    ft_name = ('item_cnt_shifted%s' % lag)

    full_train_monthly_by_category[ft_name] = full_train_monthly_by_category.sort_values('date_block_num').groupby(['shop_id', 'item_category_id'])['item_cnt'].shift(lag)
full_train_monthly_by_category.head()
for lag in lag_list:

    ft_name = ('item_cnt_shifted%s' % lag)

    full_train_cat_no_shop[ft_name] = full_train_cat_no_shop.sort_values('date_block_num').groupby(['item_category_id'])['item_cnt'].shift(lag)
full_train_cat_no_shop.head()
for lag in lag_list:

    ft_name = ('item_price_mean_shifted%s' % lag)

    full_train_monthly[ft_name] = full_train_monthly.sort_values('date_block_num').groupby(['shop_id','item_id'])['item_price_mean'].shift(lag)
full_train_monthly.head()
for lag in lag_list:

    ft_name = ('item_price_mean_shifted%s' % lag)

    full_train_monthly_by_category[ft_name] = full_train_monthly_by_category.sort_values('date_block_num').groupby(['shop_id','item_category_id'])['item_price_mean'].shift(lag)
full_train_monthly_by_category.head(3)
for lag in lag_list:

    ft_name = ('item_price_mean_shifted%s' % lag)

    full_train_cat_no_shop[ft_name] = full_train_cat_no_shop.sort_values('date_block_num').groupby(['item_category_id'])['item_price_mean'].shift(lag)
full_train_cat_no_shop.head(3)
full_train_monthly  = pd.concat(

          [full_train_monthly, pd.get_dummies(full_train_monthly['year'], prefix='year')],axis=1

        )

full_train_monthly  = pd.concat(

          [full_train_monthly, pd.get_dummies(full_train_monthly['month'], prefix='month')],axis=1

        )
full_train_monthly.shape
full_train_monthly_by_category = pd.concat(

          [full_train_monthly_by_category, pd.get_dummies(full_train_monthly_by_category['year'], prefix='year')],axis=1

        )

full_train_monthly_by_category = pd.concat(

          [full_train_monthly_by_category, pd.get_dummies(full_train_monthly_by_category['month'], prefix='month')],axis=1

        )
full_train_monthly_by_category.shape
full_train_cat_no_shop = pd.concat(

          [full_train_cat_no_shop, pd.get_dummies(full_train_cat_no_shop['year'], prefix='year')],axis=1

        )

full_train_cat_no_shop = pd.concat(

          [full_train_cat_no_shop, pd.get_dummies(full_train_cat_no_shop['month'], prefix='month')],axis=1

        )
full_train_cat_no_shop.shape
shop_id_test = test['shop_id'].unique()

item_id_test = test['item_id'].unique()
t_train = full_train_monthly[full_train_monthly['shop_id'].isin(shop_id_test)]

t_train.shape
t_train = t_train[t_train['item_id'].isin(item_id_test)]

t_train.shape
t = test.copy()

t = t.join(items, on='item_id', rsuffix='_').drop(['item_id_', 'item_name'], axis=1)
print('Train before reduction:',full_train_monthly.shape, 'Train after reduction:', t_train.shape)
t_train_cat = full_train_monthly_by_category.copy()
%%time

m_test = pd.merge(t, t_train[t_train.date_block_num==33], how = 'left', on=['shop_id', 'item_id'])

m_test = m_test.rename(columns={'item_category_id_x':'item_category_id'})

m_test = m_test.drop('item_category_id_y', axis=1)
m_test.head(3)
%%time

# для тех пар, которых нет в обучающей выборке t_train, берем информацию по продажам из таблицы t_train_cat

# по средним продажам в категории:



null_test = pd.merge(m_test[m_test['item_cnt'].isnull()][['ID', 'shop_id', 'item_id','item_category_id']],t_train_cat[t_train_cat.date_block_num==33],how = 'left',on = ['shop_id', 'item_category_id'])

null_test.index = null_test['ID']

for i in m_test.columns:

    m_test.loc[m_test.ID.isin(null_test.ID),i] = null_test[i]
m_test[m_test['item_cnt'].isnull()].shape
# # для тех пар, которых нет ни в обучающей выборке t_train, ни в таблице по продажам в магазине по категориям t_train_cat,

# # берем средние продажи в категории по всем магазинам:

# null_test_2 = pd.merge(m_test[m_test['item_cnt'].isnull()][['ID', 'shop_id', 'item_id','item_category_id']],

#                      full_train_cat_no_shop[full_train_cat_no_shop.date_block_num==33], 

#                                          how = 'left',

#                                          on = ['item_category_id']

#                      )

# null_test_2.index = null_test_2['ID']

# for i in m_test.columns:

#     m_test.loc[m_test.ID.isin(null_test_2.ID),

#                                       i] = null_test_2[i]
m_test.loc[m_test['item_cnt'].isnull(),'year'] = 2015

m_test.loc[m_test['item_cnt'].isnull(),'month'] = 10

m_test.loc[m_test['item_cnt'].isnull(),'month_10'] = 1

m_test.loc[m_test['item_cnt'].isnull(),'date_block_num'] = 33
m_test = m_test.fillna(0)
m_test.head()
drop_cols = ['date_block_num',

             'ID',

             'shop_id',

             'item_id',

             'item_price_median',

             'year',

             'month',

             'item_cnt_next_month',

             'item_category_id']

X_test = m_test.drop(drop_cols, axis=1)
X_test.to_csv('X_test.csv', index=False)
# t_train.to_csv('t_train.csv', index=False)
# full_train_cat_no_shop.to_csv('full_train_cat_no_shop.csv', index=False)
# full_train_monthly.to_csv('full_train_monthly.csv', index=False)
# full_train_monthly_by_category.to_csv('full_train_monthly_by_category.csv', index=False)
# X_test = pd.read_csv('X_test.csv')

# test = pd.read_csv('test.csv')

# t_train = pd.read_csv('t_train.csv')

# t_train_cat = pd.read_csv('t_train_cat.csv')

# full_train_cat_no_shop = pd.read_csv('full_train_cat_no_shop.csv')
# full_train_monthly = pd.read_csv('full_train_monthly.csv')

# full_train_monthly_by_category = pd.read_csv('full_train_monthly_by_category.csv')
%%time

train_set = t_train.query('date_block_num>=6 and date_block_num <33').copy()

train_set_cat = t_train_cat.query('date_block_num>=6 and date_block_num <33').copy()
rand_n = train_set.query('date_block_num==6').shape[0]

print(rand_n)

for_val = np.random.choice(rand_n, size=int(0.2*rand_n), replace = False)

print(for_val, for_val.shape)
# np.setxor1d: Find the set exclusive-or of two arrays.

# Return the sorted, unique values that are in only one (not both) of the input arrays.



for_fit = np.setxor1d(np.arange(rand_n), for_val)

print(for_fit, for_fit.shape)
x1 = train_set[train_set['date_block_num']==6].reset_index().iloc[for_val]

for i in range(7,33):

    x = train_set[train_set['date_block_num']==i].reset_index().iloc[for_val]

    x1 = x1.append(x)

val_data = x1.copy()
x2 = train_set[train_set['date_block_num']==6].reset_index().iloc[for_fit]

for i in range(7,33):

    xx = train_set[train_set['date_block_num']==i].reset_index().iloc[for_fit]

    x2 =  x2.append(xx)

fit_data = x2.copy()
%%time

drop_cols = ['date_block_num', 'shop_id', 'item_id', 'item_price_median', 'year', 'month', 'item_cnt_next_month','item_category_id', 'index']

X_train = fit_data.drop(drop_cols, axis=1)

Y_train = fit_data['item_cnt_next_month']

X_val = val_data.drop(drop_cols, axis=1)

Y_val = val_data['item_cnt_next_month']
def LGBReg(x_train, y_train, x_val, y_val):

    lgb_reg = LGBMRegressor(

        n_jobs=-1,

        tree_method='auto',

        learning_rate=0.02,

        max_depth=8,

        n_estimators=1000,

        colsample_bytree=0.8, 

        subsample=0.8, 

        seed=42)

    

    lgb_reg.fit(

        x_train, 

        y_train, 

        eval_metric="rmse", 

        eval_set=[(x_train, y_train), (x_val, y_val)], 

        verbose=10, 

        early_stopping_rounds = 10)

    return lgb_reg
%%time

lgb_reg_1 = LGBReg(X_train, Y_train, X_val, Y_val)
plot_importance(lgb_reg_1, figsize=(20, 20))
# importance_features = pd.DataFrame(lgb_reg_1.feature_importances_, X_train.columns).sort_values(by=[0], ascending=False)

# importance_features
lgb_test_pred = lgb_reg_1.predict(X_test).clip(0, 20)
submission37 = pd.DataFrame(test['ID'])

submission37['item_cnt_month'] = lgb_test_pred

submission37.to_csv('submission37.csv', index=False)
submission37.head()
# if we round submission:

submission38 = pd.DataFrame(test['ID'])

submission38['item_cnt_month'] = lgb_test_pred.round()

submission38.to_csv('submission38.csv', index=False)
submission38.head()
from statsmodels.regression.linear_model import OLS

import statsmodels.api as sm
x = sm.add_constant(X_train.fillna(0))
model2 = OLS(

    Y_train,

    x

).fit()

print(model2.summary())
X_test['const'] = 1
X_test.head()
y_predict_OLS = model2.predict(X_test).clip(0, 20)
submission39 = pd.DataFrame(test['ID'])

submission39['item_cnt_month'] = y_predict_OLS

submission39.to_csv('submission39.csv', index=False)
submission39.head()
rmse_train_1 = sqrt(mean_squared_error(Y_train, model2.predict(x)))

rmse_train_1
rmse_val_1 = sqrt(mean_squared_error(Y_val,model2.predict(sm.add_constant(X_val.fillna(0)))))

rmse_val_1