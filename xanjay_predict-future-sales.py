import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.plotting import scatter_matrix

import seaborn as sns # plotting

import matplotlib.pyplot as plt # plotting



%matplotlib inline # use inline backend



from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from itertools import product # create cartesian product



from xgboost import XGBRegressor

from xgboost import plot_importance



from datetime import datetime

import time

import pickle

import gc # garbage collection: used to free resources (cpu, ram, gpu)
item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")

sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

sample_submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")
sample_submission.head()
test.head()
#date vs item_count plot

# scatter_matrix(sales_train[['date_block_num','item_cnt_day']], alpha=0.2, figsize=(10, 10))
# -ve count=item returned

print("Min Item Count:",sales_train['item_cnt_day'].min(),"Max Item Count:",sales_train['item_cnt_day'].max())

plt.scatter(sales_train['date_block_num'], sales_train['item_cnt_day'])
plt.figure(figsize=(15,8))

sns.heatmap(sales_train[:1000000].corr())
print("Before:",sales_train.shape)

print(sales_train[sales_train.duplicated()])

sales_train = sales_train.drop_duplicates()

print("After:",sales_train.shape)
print("Total shops: ",shops.shop_id.nunique())

print("Shops in training set: ",sales_train.shop_id.nunique())

print("Shops in test set: ",test.shop_id.nunique())

print("Total Items: ",items.item_id.nunique())

print("Items in training set: ",sales_train.item_id.nunique())

print("Items in test set: ",test.item_id.nunique())
### shop and item not in training set:

# shops not in training set

print(test[~test.shop_id.isin(sales_train.shop_id.unique())]['shop_id'].nunique())

# items not in training set

print(test[~test.item_id.isin(sales_train.item_id.unique())]['item_id'].nunique())
shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])

shops.loc[shops.city=='!Якутск','city'] = 'Якутск'
shops['city_code'] = LabelEncoder().fit_transform(shops['city'])

shops = shops[['shop_id','city_code']]
# Remove shops and items from training set which are absent in test set

# can be skipped - removes 10 million rows

# test_shops = test.shop_id.unique()

# train = sales_train[sales_train.shop_id.isin(test_shops)]

# test_items = test.item_id.unique()

# train = train[train.item_id.isin(test_items)].reset_index(drop=True)

# OR

train = sales_train.copy()
# count: not-null, size=all, nunique= distinct

train.agg(['count','size','nunique'])
print(sales_train.shape)

print(train.shape)
# There is one item with price below zero. Fill it with median.

train[train.item_price<0]
median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()

train.loc[train.item_price<0, 'item_price'] = median
plt.figure(figsize=(10,4))

plt.boxplot(train.item_cnt_day,vert=False)

plt.xlabel("item_cnt_month")

plt.figure(figsize=(10,4))

plt.boxplot(train.item_price,vert=False)

plt.xlabel("item_price")
train = train[train.item_price<100000]

train = train[train.item_cnt_day<1001]
train['item_cnt_day'] = train.item_cnt_day.clip(0,20)
train['revenue'] = train['item_price'] * train['item_cnt_day']

train = train.drop(['item_price'],axis=1)

train.head()
# cols = ['date_block_num','shop_id','item_id']

train = train.groupby(['date_block_num','shop_id','item_id']).sum()

train = train.rename({'item_cnt_day':'item_cnt_month','revenue':'revenue_month'}, axis=1)

train.reset_index(inplace=True)
train.head()
matrix = []

cols = ['date_block_num','shop_id','item_id']

for i in range(34):

    sales = train[train.date_block_num==i]

    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='uint16'))

    

matrix = pd.DataFrame(np.vstack(matrix), columns=cols)

matrix['date_block_num'] = matrix['date_block_num'].astype(np.uint8)

matrix['shop_id'] = matrix['shop_id'].astype(np.uint8)

matrix['item_id'] = matrix['item_id'].astype(np.uint16)

matrix.sort_values(cols,inplace=True)
matrix = pd.merge(matrix, train, on=cols, how='left')

matrix.shape
train = matrix
train.head()
test1 = test.copy()

test1['date_block_num'] = 34

test1.tail()
test1 = test1[['date_block_num','shop_id','item_id']]

combined = train.append(test1,ignore_index=True, sort=False)

combined.tail()
combined.fillna(0,inplace=True) # 34

combined.tail()
combined.describe()
# idx = np.random.permutation(range(10000))[:1000]

# idx.sort()

train123 = combined.copy()#train.iloc[idx]

# train.iloc[idx[:1000]]

train123.head()
train123.info()
train123 = train123.astype({"date_block_num":np.uint8, "shop_id":np.uint8, "item_id":np.uint16,

                              "item_cnt_month":np.float32, "revenue_month":np.float32})
# add monthly average sales_count by shop

group = train123.groupby(['date_block_num','shop_id']).agg({'item_cnt_month':['mean']})

group.columns = ['shop_cnt_month_avg']

group.reset_index(inplace=True)

train123 = pd.merge(train123,group,on=['date_block_num','shop_id'],how='left')
# add monthly average sales_count by item

group = train123.groupby(['date_block_num','item_id']).agg({'item_cnt_month':['mean']})

group.columns = ['item_cnt_month_avg']

group.reset_index(inplace=True)

train123 = pd.merge(train123, group, on=['date_block_num','item_id'],how='left')
# function to visualize datewise shop metric

def plot_shop_metric(df, metric):

    df = df.sort_values(by=['shop_id'])

    total_shops = df['shop_id'].unique()

    ncols = 2

    nrows = len(total_shops)//(ncols*5)

    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, sharex=True, sharey=True , figsize=(16,20))

    axes[0][0].set_xlim((0, 34))

    axes[0][0].set_ylim((df[metric].min(), df[metric].max()+2)) # (0,10)

    count = 0

    for row in range(nrows):

        for col in range(ncols):

            if count >= len(total_shops):

                break

            sns.pointplot(x='date_block_num',y=metric,hue='shop_id',

                  data=df[df.shop_id.isin(total_shops[count:count+6])],

                          ax=axes[row][col])

            count += 6

plot_shop_metric(train123, "shop_cnt_month_avg")
train123.memory_usage(deep=True) / 1024 ** 2
train123.fillna(0,inplace=True)

train123.describe()
def lag_feature(df, lags, col):

    tmp = df[['date_block_num','shop_id','item_id',col]]

    for i in lags:

        shifted = tmp.copy()

        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]

        shifted['date_block_num'] += i

        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')

    return df
lags = [1,2,3,6,12]

gc.collect()

train123 = lag_feature(train123, lags, 'item_cnt_month')

train123 = lag_feature(train123, lags, 'revenue_month')

train123 = lag_feature(train123, lags, 'shop_cnt_month_avg')

train123 = lag_feature(train123, lags, 'item_cnt_month_avg')
train123.fillna(0, inplace=True)

train123.describe()
train123.tail()
train123.shape
## add shop city_code

train123 = pd.merge(train123, shops, on='shop_id', how='left')

train123.city_code = train123.city_code.astype(np.uint8)

## add categories

train123 = pd.merge(train123, items, on='item_id', how='left').drop('item_name',axis=1)

train123.item_category_id = train123.item_category_id.astype(np.uint8)

## Add month

train123['month'] = train123['date_block_num']%12



# Number of days in a month. There are no leap years.

days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])

train123['days'] = train123['month'].map(days).astype(np.int8)

# end of year

train123["is_endof_year"] = train123.apply(lambda x: 1 if x['month']%12==11 else 0, axis=1).astype(np.uint8)

# start of year

train123["is_startof_year"] = train123.apply(lambda x: 1 if x['month']%12==0 else 0, axis=1).astype(np.uint8)
train123.info()
# drop dependent features

train123 = train123.drop(['revenue_month','shop_cnt_month_avg','item_cnt_month_avg'], axis=1)
# since 12 months lag is used for all features

train123 = train123[train123.date_block_num>11]
# fillna if any

train123.isnull().values.any()
train123.to_pickle('data3.pkl')

gc.collect() # garbage collection
!ls ../input/processed-sales-train-data
data = pd.read_pickle('../input/processed-sales-train-data/data3.pkl')
data.head()
data.info()
X_train = data[data.date_block_num<33].drop(['item_cnt_month'], axis=1)

Y_train = data[data.date_block_num<33][['item_cnt_month']]

X_valid = data[data.date_block_num==33].drop(['item_cnt_month'], axis=1)

Y_valid = data[data.date_block_num==33][['item_cnt_month']]

X_test = data[data.date_block_num==34].drop(['item_cnt_month'], axis=1)

Y_test = data[data.date_block_num==34][['item_cnt_month']]
scaler = StandardScaler()

scalery = StandardScaler()
# MinMax Scaler:

# scaler = MinMaxScaler()

# scalery = MinMaxScaler()

# OR
scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_valid = scaler.transform(X_valid)

X_test = scaler.transform(X_test)
scalery.fit(Y_train)

Y_train = scalery.transform(Y_train)

Y_valid = scalery.transform(Y_valid)

Y_test = scalery.transform(Y_test)
linear_reg = LinearRegression()

linear_reg.fit(X_train, Y_train)
y_pred = linear_reg.predict(X_valid)

rmse = np.sqrt(metrics.mean_squared_error(Y_valid, y_pred))

print('Root Mean Squared Error:', rmse)

model = linear_reg
def plot_features_linear_regression(data, model):

    columns = data.drop(['item_cnt_month'], axis=1).columns

    plt.figure(figsize=(10,16))

    y_pos = np.arange(len(columns))

    feature_imp = model.coef_.reshape(-1)

    plt.barh(y=y_pos, width=feature_imp)

    plt.yticks(y_pos, columns)

    plt.show()
plot_features_linear_regression(data, linear_reg)
ts = time.time()

from sklearn.svm import LinearSVR

lin_svr = LinearSVR() # epsilon = width of street

lin_svr.fit(X_train, Y_train.reshape(-1))

time.time() - ts
model = lin_svr
model = XGBRegressor()

model.fit(

    X_train, 

    Y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 

    verbose=True, 

    early_stopping_rounds = 3)
# for xgboost

def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)

plot_features(model, (10,14))

# predict on test data

Y_pred = model.predict(X_test)

Y_pred = scalery.inverse_transform(Y_pred).clip(0, 20)
submission = pd.DataFrame({"ID": test.ID, "item_cnt_month":Y_pred.reshape(-1)})
submission.to_csv("submission.csv", index=False)