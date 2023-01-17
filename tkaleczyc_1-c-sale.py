import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from math import sqrt

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from matplotlib import style
style.use('seaborn')

import os
print(os.listdir("../input"))

%matplotlib inline
pd.options.display.float_format = '{:,.2f}'.format
PATH = '../input/'
sales = pd.read_csv(PATH + 'sales_train.csv', low_memory=False); sales.tail()
sales.describe(include="all")
items = pd.read_csv(PATH + 'items.csv', low_memory=False, index_col=1); items.head()
item_cat = pd.read_csv(PATH + 'item_categories.csv', low_memory=False, index_col=1); item_cat.head()
shops = pd.read_csv(PATH + 'shops.csv', low_memory=False, index_col=1); shops.head()
test = pd.read_csv(PATH + 'test.csv', low_memory=False, index_col=0); test.head()
def assign_key(df): df['key'] = df['item_id'].astype('int32').map(str) + "_" + df['shop_id'].astype('int32').map(str)
    
assign_key(sales)
assign_key(test)
test['key'][~test['key'].isin(sales['key'])].shape[0] / test['key'].shape[0]
test['item_id'][~test['item_id'].isin(sales['item_id'])].drop_duplicates().shape[0]
test['shop_id'][~test['shop_id'].isin(sales['shop_id'])].shape[0]
sales[['date_block_num', 'item_cnt_day']].groupby('date_block_num').sum().plot(figsize=(9, 6))
plt.show()
df = sales.copy()
test_set = test.copy()
df = df.merge(items[['item_category_id']], how='left', on='item_id')
test_set = test_set.merge(items[['item_category_id']], how='left', on='item_id')
test_set.index.rename('ID', inplace=True)
sales.shape[0] == df.shape[0], test_set.shape[0] == test.shape[0]
df.columns, test.columns
df.isna().sum(), test.isna().sum()
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
attr = ['year', 'month', 'week', 'day']

for a in attr: df[a] = getattr(df['date'].dt, a)
df.columns
d = df[['date_block_num', 'day']].groupby(['date_block_num']).max()
di = dict([(d.index[i], d.values[i][0]) for i in range(len(d))])
df['day_count'] = df['date_block_num'].map(di)
df.loc[((df['week'] == 1) & (df['day'] > 7)), 'week'] = 53
df['week_block_num'] = (df['year'] - df['year'].min()) * 52 + df['week']

wmax = df[['date_block_num', 'week_block_num']].groupby(['date_block_num']).max()
wmin = df[['date_block_num', 'week_block_num']].groupby(['date_block_num']).min()
di = dict([(wmax.index[i], wmax.values[i][0] - wmin.values[i][0]) for i in range(len(wmax))])
df['week_count'] = df['date_block_num'].map(di)
df['qtr'] = df['month'].map(dict([[i, int(i // 3.1 + 1)] for i in df['month'].unique()]))
df['x_mas'] = df['month'].map(dict([[i, i // 12] for i in df['month'].unique()]))
df.columns
to_group = ['date_block_num', 'shop_id', 'item_id', 'key', 'item_category_id',
            'year', 'month', 'day_count', 'week_count', 'qtr', 'x_mas']
to_mean = ['item_price']
to_sum = ['item_cnt_day']
da = df[to_group].copy()
da = pd.concat([da,
                df[to_group + to_mean].groupby(to_group).transform('mean'),
                df[to_group + to_sum].groupby(to_group).transform('sum'),
               ], axis=1).drop_duplicates()
da.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
da.head()
da.describe(include='all')
test_set.columns
avg_prices = pd.concat((da[['date_block_num', 'item_id']], da[['date_block_num', 'item_id','item_price']]
                        .groupby(['date_block_num', 'item_id'])
                        .transform('mean')), axis=1).sort_values(by=['date_block_num', 'item_id']).drop_duplicates().sort_values(by=['item_id', 'date_block_num'], ascending=False)

for m in da['date_block_num'].unique():
    mask = ~test_set['key'].isin(da['key'][da['date_block_num'] == m])
    mask2 = (da['date_block_num'] == m)
    
    md = test_set[['key', 'shop_id', 'item_id', 'item_category_id']][mask]
    md['date_block_num'] = m
    for col in ['year', 'month', 'day_count', 'week_count', 'qtr', 'x_mas']: md[col] = da[col][mask2].max()
    
    avg = avg_prices[avg_prices['date_block_num'] <= m].drop(columns='date_block_num').groupby('item_id').mean()
    md['item_price'] = md['item_id'].map(dict(zip(avg.index, avg.values[:, 0]))).fillna(0)
    md['item_cnt_month'] = 0
    
    da = da.append(md, ignore_index=True, sort=True)
s_ch = sales.copy()
s_ch["orig_sales"] = s_ch["item_price"] * s_ch["item_cnt_day"]
s_ch["orig_item_count"] = s_ch["item_cnt_day"]

a_ch = da.copy()
a_ch["agg_sales"] = a_ch["item_price"] * a_ch["item_cnt_month"]
a_ch["agg_item_count"] = a_ch["item_cnt_month"]

fig = plt.figure(figsize=(15, 10))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))

s_ch[["date_block_num", "orig_sales"]].groupby("date_block_num").agg([sum]).plot(legend=True, ax=ax1, title="Total Sales Value")
s_ch[["date_block_num", "orig_item_count"]].groupby("date_block_num").agg([sum]).plot(legend=True, ax=ax2, title="Items Sold")

a_ch[["date_block_num", "agg_sales"]].groupby("date_block_num").agg([sum]).plot(legend=True, ax=ax1, color='r')
a_ch[["date_block_num", "agg_item_count"]].groupby("date_block_num").agg([sum]).plot(legend=True, color='r', ax=ax2)


plt.tight_layout()
plt.show()
da['prev_month'] = da['date_block_num'].map(dict([[i, i-1] if i > 0 else [i, np.nan] for i in df['date_block_num'].unique()]))
da['prev_qtr'] = da['date_block_num'].map(dict([[i, i-3] if i > 2 else [i, np.nan] for i in df['date_block_num'].unique()]))
da['prev_year'] = da['date_block_num'].map(dict([[i, i-12] if i > 11 else [i, np.nan] for i in df['date_block_num'].unique()]))
di = dict(zip(da['date_block_num'].astype('float32').map(str) + da['key'].values, da['item_cnt_month']))

def prev_sales(prevsale, prevperiod): 
    da[prevsale] = (da[prevperiod].map(str) + da['key']).map(di).fillna(0)
    da.drop(columns=prevperiod, inplace=True)

prev_sales('item_cnt_prevm', 'prev_month')
prev_sales('item_cnt_prevq', 'prev_qtr')
prev_sales('item_cnt_prevy', 'prev_year')
da[['date_block_num','item_cnt_month', 'item_cnt_prevm', 'item_cnt_prevq', 'item_cnt_prevy']][da['date_block_num'] > 11].groupby(['date_block_num']).sum().plot(figsize=(10, 6))
plt.show()
da[['date_block_num','item_cnt_month', 'item_cnt_prevm', 'item_cnt_prevq', 'item_cnt_prevy']][(da['key'].isin(test['key'])) & (da['date_block_num'] > 11)].groupby(['date_block_num']).sum().plot(figsize=(10, 6))
plt.show()
price_map = dict(da[['key', 'item_price']][da['date_block_num'] == da['date_block_num'].max()].values)
test_set['item_price'] = test_set['key'].map(price_map).fillna(0)
test_set['date_block_num'] = da['date_block_num'].max() + 1
test_set['year'] = 2015
test_set['month'] = 11
test_set['day_count'] = 30
test_set['week_count'] = 4
test_set['qtr'] = 4
test_set['x_mas'] = 0
lags = [['item_cnt_prevm', 0], ['item_cnt_prevq', 2], ['item_cnt_prevy', 11]]
for l in lags: test_set[l[0]] = test_set['key'].map(dict(da[['key', 'item_cnt_month']][(da['date_block_num'] == da['date_block_num'].max() - l[1])].values))
da[['year', 'month']][da['year']==da['year'].max()].max()
to_val = (da['date_block_num'] == da['date_block_num'].max())
to_train = (da['date_block_num'] > da['date_block_num'].max() - 12) & (da['date_block_num'] != da['date_block_num'].max())

X_train = da[to_train].drop(columns=['item_cnt_month', 'date_block_num', 'key'])
X_val = da[to_val].drop(columns=['item_cnt_month', 'date_block_num', 'key'])
da['item_cnt_month'].skew()
y_train = np.log1p(da['item_cnt_month'][to_train].clip(0., 20.))
y_val = np.log1p(da['item_cnt_month'][to_val].clip(0., 20.))
X_train.shape, y_train.shape, X_val.shape, y_val.shape, da.shape
def RMSE(targ, pred): return np.sqrt(np.mean((np.expm1(targ) - np.expm1(pred))**2))
from sklearn.ensemble import RandomForestRegressor
def review(X_train, X_val):
    m.fit(X_train, y_train)
    preds = m.predict(X_val)
    print("-"*30, f'''
    Training score: {m.score(X_train, y_train)*100:.2f}%
    Validation score: {m.score(X_val, y_val)*100:.2f}%
    Out-of-Bag score: {m.oob_score_*100:.2f}%
    RMSE: {RMSE(y_val, preds):.4f}
    ''')
%%time
m = RandomForestRegressor(n_estimators=50, max_features=0.85, min_samples_leaf=5,
                          n_jobs=-1, oob_score=True)

review(X_train, X_val)
def f_i(X_train, X_val, use_RMSE=False):
    global FI
    accs = []
    if use_RMSE:
        targ = RMSE(m.predict(X_train), y_train)
    else:
        targ = m.score(X_train, y_train) 
    num_features = 15

    for c in X_train.columns:
        X = X_train.copy()
        X[c] = X[[c]].sample(frac=1).set_index(X.index)[c]  # random shuffle of one column
        if use_RMSE: 
            accs.append(RMSE(m.predict(X), y_train) - targ)
        else: 
            accs.append(targ - m.score(X, y_train))


    FI = sorted([[c, float(a)] for c, a in zip(X.columns, accs)], key=lambda x: x[1], reverse=True)[:num_features]
    pd.DataFrame({'Score loss': [FI[i][1] for i in range(len(FI))], 'Features': [FI[i][0] for i in range(len(FI))]}).set_index('Features').sort_values(by='Score loss', ascending=True).plot.barh()
    plt.show()
f_i(X_train, X_val, use_RMSE=True)
top = -2
selected = [FI[i][0] for i in range(len(FI))][:top]
Xt = X_train[selected].copy()
Xv = X_val[selected].copy()
%%time
m = RandomForestRegressor(n_estimators=50, max_features=0.85, min_samples_leaf=5,
                          n_jobs=-1, oob_score=True)
review(Xt, Xv)
f_i(Xt, Xv, use_RMSE=True)
tp = da[['date_block_num', 'item_cnt_month']][to_train]
tp['item_cnt_month'].clip(0., 20., inplace=True)
tp['preds'] = np.expm1(m.predict(Xt))
pd.concat((tp['date_block_num'], tp.groupby('date_block_num').transform('sum')), axis=1).drop_duplicates().append(
    {'date_block_num': 33,'item_cnt_month': da['item_cnt_month'][da['date_block_num'] == da['date_block_num'].max()]
     .clip(0., 20.).sum(), 'preds': np.expm1(m.predict(Xv)).sum()}, ignore_index=True).groupby('date_block_num').sum().plot()
plt.show()
mask = (da['date_block_num'] > da['date_block_num'].max() - 12)
X_train = da[selected][mask]
# X_train = da.drop(columns=['item_cnt_month', 'date_block_num', 'key'])[mask]
y_train = np.log1p(da['item_cnt_month'].clip(0., 20.))[mask]
%%time
m = RandomForestRegressor(n_estimators=50, max_features=0.85, min_samples_leaf=5,
                          n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print("-"*30, f'''
Training score: {m.score(X_train, y_train)*100:.2f}%
RMSE: {RMSE(m.predict(X_train), y_train):.4f}
''')
X_test = test_set[selected]
# X_test = test_set.drop(columns=['date_block_num', 'key'])
y_test = np.expm1(m.predict(X_test))
tp = da[['date_block_num']][mask]
tp['preds'] = np.expm1(m.predict(X_train))
tp = tp.groupby('date_block_num').transform('sum').drop_duplicates()
tp.append({'preds': sum(y_test)}, ignore_index=True).plot()
plt.show()
test_set['item_cnt_month'] = y_test
name = 'submission_v4.csv'
my_submission = test_set[['item_cnt_month']].clip(0., 20.)
my_submission.to_csv(name)
pd.read_csv(name).head()
