import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import ensemble, metrics

from sklearn import linear_model, preprocessing

import xgboost as xgb



parser = lambda date: pd.to_datetime(date, format='%d.%m.%Y')



train = pd.read_csv('../input/sales_train.csv', parse_dates=['date'], date_parser=parser)

test = pd.read_csv('../input/test.csv')

items = pd.read_csv('../input/items.csv')

item_cats = pd.read_csv('../input/item_categories.csv')

shops = pd.read_csv('../input/shops.csv')

print('train:', train.shape, 'test:', test.shape, 'items:', items.shape, 'item_cats:', item_cats.shape, 'shops:', shops.shape)
train.head()
test.head()
items.head()
item_cats.head()
shops.head()
print(train['date_block_num'].max())
print(train['item_cnt_day'].describe())
test_only = test[~test['item_id'].isin(train['item_id'].unique())]['item_id'].unique()

print('test only items:', len(test_only))
# drop duplicates

subset = ['date','date_block_num','shop_id','item_id','item_cnt_day']

print(train.duplicated(subset=subset).value_counts())

train.drop_duplicates(subset=subset, inplace=True)
# drop shops&items not in test data

test_shops = test.shop_id.unique()

test_items = test.item_id.unique()

train = train[train.shop_id.isin(test_shops)]

train = train[train.item_id.isin(test_items)]



print('train:', train.shape)
from itertools import product



# create all combinations

block_shop_combi = pd.DataFrame(list(product(np.arange(34), test_shops)), columns=['date_block_num','shop_id'])

shop_item_combi = pd.DataFrame(list(product(test_shops, test_items)), columns=['shop_id','item_id'])

all_combi = pd.merge(block_shop_combi, shop_item_combi, on=['shop_id'], how='inner')

print(len(all_combi), 34 * len(test_shops) * len(test_items))



# group by monthly

train_base = pd.merge(all_combi, train, on=['date_block_num','shop_id','item_id'], how='left')

train_base['item_cnt_day'].fillna(0, inplace=True)

train_grp = train_base.groupby(['date_block_num','shop_id','item_id'])
# summary count by month

train_monthly = pd.DataFrame(train_grp.agg({'item_cnt_day':['sum','count']})).reset_index()

train_monthly.columns = ['date_block_num','shop_id','item_id','item_cnt','item_order']

print(train_monthly[['item_cnt','item_order']].describe())

# trim count

train_monthly['item_cnt'].clip(0, 20, inplace=True)



train_monthly.head()
item_grp = item_cats['item_category_name'].apply(lambda x: str(x).split(' ')[0])

item_grp = pd.Categorical(item_grp).codes

item_cats['item_group'] = item_grp

#item_cats = item_cats.join(pd.get_dummies(item_grp, prefix='item_group', drop_first=True))



items = pd.merge(items, item_cats.loc[:,['item_category_id','item_group']], on=['item_category_id'], how='left')



city = shops.shop_name.apply(lambda x: str.replace(x, '!', '')).apply(lambda x: x.split(' ')[0])

shops['city'] = pd.Categorical(city).codes
_ = '''

from sklearn.feature_extraction.text import TfidfVectorizer



feature_cnt = 25

tfidf = TfidfVectorizer(max_df=0.6, max_features=feature_cnt, ngram_range=(1, 2))

txtFeatures = pd.DataFrame(tfidf.fit_transform(items['item_name']).toarray())

cols = txtFeatures.columns

for i in range(feature_cnt):

    items['item_name_tfidf_' + str(i)] = txtFeatures[cols[i]]

'''
# By shop,item

grp = train_monthly.groupby(['shop_id', 'item_id'])

train_shop = grp.agg({'item_cnt':['mean','median','std'],'item_order':'mean'}).reset_index()

train_shop.columns = ['shop_id','item_id','cnt_mean_shop','cnt_med_shop','cnt_std_shop','order_mean_shop']

print(train_shop[['cnt_mean_shop','cnt_med_shop','cnt_std_shop']].describe())



train_shop.head()
# By shop,item_group

train_cat_monthly = pd.merge(train_monthly, items, on=['item_id'], how='left')

grp = train_cat_monthly.groupby(['shop_id', 'item_group'])

train_shop_cat = grp.agg({'item_cnt':['mean']}).reset_index()

train_shop_cat.columns = ['shop_id','item_group','cnt_mean_shop_cat']

print(train_shop_cat.loc[:,['cnt_mean_shop_cat']].describe())



train_shop_cat.head()
# By month,shop,item At previous

train_prev = train_monthly.copy()

train_prev['date_block_num'] = train_prev['date_block_num'] + 1

train_prev.columns = ['date_block_num','shop_id','item_id','cnt_prev','order_prev']



for i in [2,12]:

    train_prev_n = train_monthly.copy()

    train_prev_n['date_block_num'] = train_prev_n['date_block_num'] + i

    train_prev_n.columns = ['date_block_num','shop_id','item_id','cnt_prev' + str(i),'order_prev' + str(i)]

    train_prev = pd.merge(train_prev, train_prev_n, on=['date_block_num','shop_id','item_id'], how='left')



train_prev.head()
# By month,shop,item_group At previous

grp = pd.merge(train_prev, items, on=['item_id'], how='left').groupby(['date_block_num','shop_id','item_group'])

train_cat_prev = grp['cnt_prev'].mean().reset_index()

train_cat_prev = train_cat_prev.rename(columns={'cnt_prev':'cnt_prev_cat'})

print(train_cat_prev.loc[:,['cnt_prev_cat']].describe())



train_cat_prev.head()
train_piv = train_monthly.pivot_table(index=['shop_id','item_id'], columns=['date_block_num'], values='item_cnt', aggfunc=np.sum, fill_value=0)

train_piv = train_piv.reset_index()

train_piv.head()
# MACD At previous

col = np.arange(34)

pivT = train_piv[col].T

ema_s = pivT.ewm(span=12).mean().T

ema_l = pivT.ewm(span=26).mean().T

macd = ema_s - ema_l

sig = macd.ewm(span=9).mean()



ema_list = []

for c in col:

  sub_ema = pd.concat([train_piv.loc[:,['shop_id','item_id']],

      pd.DataFrame(ema_s.loc[:,c]).rename(columns={c:'cnt_ema_s_prev'}),

      pd.DataFrame(ema_l.loc[:,c]).rename(columns={c:'cnt_ema_l_prev'}),

      pd.DataFrame(macd.loc[:,c]).rename(columns={c:'cnt_macd_prev'}),

      pd.DataFrame(sig.loc[:,c]).rename(columns={c:'cnt_sig_prev'})], axis=1)

  sub_ema['date_block_num'] = c + 1

  ema_list.append(sub_ema)

    

train_ema_prev = pd.concat(ema_list)

train_ema_prev.head()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))



train_monthly.groupby(['date_block_num']).sum().reset_index()['item_cnt'].plot(ax=ax[0])

train_cat_monthly = pd.merge(train_monthly, items, on=['item_id'], how='left')

train_cat_monthly.pivot_table(index=['date_block_num'], columns=['item_group'], values='item_cnt', aggfunc=np.sum, fill_value=0).plot(ax=ax[1], legend=False)
# Price mean by month,shop,item

train_price = train_grp['item_price'].mean().reset_index()

price = train_price[~train_price['item_price'].isnull()]



# last price by shop,item

last_price = price.drop_duplicates(subset=['shop_id', 'item_id'], keep='last').drop(['date_block_num'], axis=1)



# null price by shop,item

'''

mean_price = price.groupby(['item_id'])['item_price'].mean().reset_index()

result_price = pd.merge(test, mean_price, on=['item_id'], how='left').drop('ID', axis=1)

pred_price_set = result_price[result_price['item_price'].isnull()]

'''

uitem = price['item_id'].unique()

pred_price_set = test[~test['item_id'].isin(uitem)].drop('ID', axis=1)
_ = '''

'''

if len(pred_price_set) > 0:

    train_price_set = pd.merge(price, items, on=['item_id'], how='inner')

    pred_price_set = pd.merge(pred_price_set, items, on=['item_id'], how='inner').drop(['item_name'], axis=1)

    reg = ensemble.ExtraTreesRegressor(n_estimators=25, n_jobs=-1, max_depth=15, random_state=42)

    reg.fit(train_price_set[pred_price_set.columns], train_price_set['item_price'])

    pred_price_set['item_price'] = reg.predict(pred_price_set)



test_price = pd.concat([last_price, pred_price_set], join='inner')

test_price.head()
price_max = price.groupby(['item_id']).max()['item_price'].reset_index()

price_max.rename(columns={'item_price':'item_max_price'}, inplace=True)

price_max.head()
train_price_a = pd.merge(price, price_max, on=['item_id'], how='left')

train_price_a['discount_rate'] = 1 - (train_price_a['item_price'] / train_price_a['item_max_price'])

train_price_a.drop('item_max_price', axis=1, inplace=True)

train_price_a.head()
test_price_a = pd.merge(test_price, price_max, on=['item_id'], how='left')

test_price_a.loc[test_price_a['item_max_price'].isnull(), 'item_max_price'] = test_price_a['item_price']

test_price_a['discount_rate'] = 1 - (test_price_a['item_price'] / test_price_a['item_max_price'])

test_price_a.drop('item_max_price', axis=1, inplace=True)

test_price_a.head()
def mergeFeature(source): 

  d = source

  d = pd.merge(d, items, on=['item_id'], how='left').drop('item_group', axis=1)

  d = pd.merge(d, item_cats, on=['item_category_id'], how='left')

  d = pd.merge(d, shops, on=['shop_id'], how='left')



  d = pd.merge(d, train_shop, on=['shop_id','item_id'], how='left')

  d = pd.merge(d, train_shop_cat, on=['shop_id','item_group'], how='left')

  d = pd.merge(d, train_prev, on=['date_block_num','shop_id','item_id'], how='left')

  d = pd.merge(d, train_cat_prev, on=['date_block_num','shop_id','item_group'], how='left')

  d = pd.merge(d, train_ema_prev, on=['date_block_num','shop_id','item_id'], how='left')

  

  d['month'] = d['date_block_num'] % 12

  days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])

  d['days'] = d['month'].map(days).astype(np.int8)

  

  d.drop(['shop_id','shop_name','item_id','item_name','item_category_id','item_category_name','item_group'], axis=1, inplace=True)

  d.fillna(0.0, inplace=True)

  return d
train_set = train_monthly[train_monthly['date_block_num'] >= 12]



train_set = pd.merge(train_set, train_price_a, on=['date_block_num','shop_id','item_id'], how='left')

train_set = mergeFeature(train_set)



train_set = train_set.join(pd.DataFrame(train_set.pop('item_order'))) # move to last column



X_train = train_set.drop(['item_cnt'], axis=1)

Y_train = train_set['item_cnt']

X_train.head()
test_set = test.copy()

test_set['date_block_num'] = 34



test_set = pd.merge(test_set, test_price_a, on=['shop_id','item_id'], how='left')

test_set = mergeFeature(test_set)



test_set['item_order'] = test_set['order_prev']

test_set.loc[test_set['item_order'] == 0, 'item_order'] = 1



X_test = test_set.drop(['ID'], axis=1)

X_test.head()
#reg = ensemble.ExtraTreesRegressor(n_estimators=25, n_jobs=-1, max_depth=15, random_state=42)

reg = ensemble.GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3, 

                                            max_features='sqrt', loss='huber', random_state=42)

#reg = xgb.XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.05, subsample=0.6, colsample_bytree=0.6)

#reg = xgb.XGBRegressor(n_estimators=25, max_depth=12, learning_rate=0.1, subsample=1, colsample_bytree=0.9, random_state=42, eval_metric='rmse')
#train_set.corr()['item_cnt'].sort_values(ascending=False)
_ = '''

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import ShuffleSplit



reg = ensemble.ExtraTreesRegressor(n_estimators=25, n_jobs=-1, max_depth=15, random_state=42)



#pred = cross_val_predict(reg, X_train, Y_train, cv=6)

#print('RMSE:', np.sqrt(metrics.mean_squared_error(np.log1p(Y_train.clip(0.,20.)), np.log1p(pred.clip(0.,20.)))))



ss = ShuffleSplit(n_splits=1, test_size=0.03, random_state=42)

for train_index, test_index in ss.split(X_train):

    cnt_sub_scaler = preprocessing.StandardScaler();

    reg.fit(X_train.iloc[train_index], cnt_sub_scaler.fit_transform(Y_train.iloc[train_index].clip(0.,20.).values.reshape(-1, 1)).ravel())

    pred_cnt = cnt_sub_scaler.inverse_transform(reg.predict(X_train.iloc[test_index]))

    

    #reg.fit(X_train.iloc[train_index], Y_train.iloc[train_index].clip(0.,20.))

    #pred_cnt = reg.predict(X_train.iloc[test_index])

    

    print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_train.iloc[test_index].clip(0.,20.), pred_cnt.clip(0.,20.)))) #1.38235624903

'''
_ = '''

'''

x1 = train_set[train_set['date_block_num'] < 33]

y1 = x1['item_cnt']

x1 = x1.drop(['item_cnt'], axis=1)



x2 = train_set[train_set['date_block_num'] == 33]

y2 = x2['item_cnt']

x2 = x2.drop(['item_cnt'], axis=1)



reg.fit(x1, y1)

pred_cnt = reg.predict(x2)

print('RMSE:', np.sqrt(metrics.mean_squared_error(y2.clip(0.,20.), pred_cnt.clip(0.,20.)))) #0.20783645197081926

#reg.fit(x1, np.log1p(y1))

#pred_cnt = reg.predict(x2)

#print('RMSE:', np.sqrt(metrics.mean_squared_error(y2.clip(0.,20.), np.expm1(pred_cnt).clip(0.,20.))))



col = [c for c in train_set.columns if c not in ['item_cnt']]

feature_imp = pd.DataFrame(reg.feature_importances_, index=col, columns=["importance"])

feature_imp.sort_values("importance", ascending=False).head(30)
assert(X_train.columns.isin(X_test.columns).all())



reg.fit(X_train, Y_train)

pred_cnt = reg.predict(X_test)



result = pd.DataFrame({

    "ID": test["ID"],

    "item_cnt_month": pred_cnt.clip(0. ,20.)

})

result.to_csv("submission.csv", index=False)
print(len(pred_cnt[pred_cnt > 20]))

result.head(30)