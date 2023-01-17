# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
path = '/kaggle/input/m5-forecasting-accuracy'

train_sales = pd.read_csv(f'{path}/sales_train_validation.csv')

calendar = pd.read_csv(f'{path}/calendar.csv')

submission = pd.read_csv(f'{path}/sample_submission.csv')

sell_prices = pd.read_csv(f'{path}/sell_prices.csv')

train_sales.shape,calendar.shape,submission.shape,sell_prices.shape
train_sales.head()
train_sales['item_id'].value_counts()
calendar.head()
submission.head()
sell_prices.head()
print(train_sales['item_id'].value_counts())

train_sales.loc[train_sales['item_id']=='HOBBIES_1_001',:]
train_sales['cat_id'].value_counts()
# 价格信息：每个item_id对应的价格是波动的

tmp = sell_prices.loc[sell_prices['item_id']=='HOBBIES_1_001',:]

print(tmp['sell_price'].value_counts())

tmp
fig = plt.figure(figsize=(16,6))

fig.set(alpha=0.2) 



plt.subplot2grid((1,2),(0,0)) 

# 查看每个商店的价格波动



item_prices = sell_prices.loc[sell_prices['item_id'].isin(['HOBBIES_1_001','FOODS_1_016'])]    

for i,item_id in enumerate(['HOBBIES_1_001','FOODS_1_016']):

    plt.subplot2grid((1,2),(0,i))

    tmp = item_prices[item_prices['item_id']==item_id]

    for i,store_id in enumerate(tmp['store_id'].unique()):

        store_sell_price = tmp.loc[tmp['store_id'] == store_id]

        plt.plot(store_sell_price['wm_yr_wk'].values,store_sell_price['sell_price'].values,label=store_id)

        

    plt.title(f'{item_id} sell_price trend')

    plt.legend()
# 查看该商品下所有类目价格

ca_1_prices = sell_prices.loc[sell_prices['store_id'] == 'CA_1']

ca_1_prices['dept_id'] = ca_1_prices['item_id'].apply(lambda x: x[:-4])



plt.figure(figsize=(12, 6))

for dept_id in ca_1_prices['dept_id'].unique():

    small_df = ca_1_prices.loc[ca_1_prices['dept_id'] == dept_id]

    grouped = small_df.groupby(['wm_yr_wk'])['sell_price'].mean()

    plt.plot(grouped.index, grouped.values, label=dept_id)

plt.legend(loc=(1.0, 0.5))

plt.title('CA_1 mean sell prices by dept');
ca_prices = sell_prices.loc[sell_prices['store_id'].str.contains('CA')]

plt.figure(figsize=(12, 6))

for d in ca_prices['store_id'].unique():

    small_df = ca_prices.loc[ca_prices['store_id'] == d]

    grouped = small_df.groupby(['wm_yr_wk'])['sell_price'].mean()

    plt.plot(grouped.index, grouped.values, label=d)

plt.legend(loc=(1.0, 0.5))

plt.title('Mean sell prices by store in CA');
# 销量信息：不同商品信息90天日均存在很大的波动

fig = plt.figure(figsize=(16,16))

fig.set(alpha=0.2) 

# 设置子图位置



plt.subplot2grid((3,1),(0,0)) 

for i in range(10):

    plt.plot(train_sales.loc[train_sales['item_id']=='HOBBIES_1_001'].iloc[i,6:].rolling(90).mean().values,\

            label=train_sales.loc[train_sales['item_id']=='HOBBIES_1_001'].iloc[i,5])

    plt.title('HOBBIES_1_001 sales, rolling mean 90 days')

    plt.legend();



plt.subplot2grid((3,1),(1,0))

for j in range(10):

    plt.plot(train_sales.loc[train_sales['item_id']=='FOODS_1_016'].iloc[i,6:].rolling(90).mean().values,\

            label=train_sales.loc[train_sales['item_id']=='FOODS_1_016'].iloc[i,5])

    plt.title('FOODS_1_016 sales, rolling mean 90 days')

    plt.legend();



plt.subplot2grid((3,1),(2,0))

tmp= train_sales[train_sales['item_id'].isin(['FOODS_1_016','HOBBIES_1_001'])].groupby('item_id',as_index=False).mean()

for i in ['FOODS_1_016','HOBBIES_1_001']:

    plt.plot(tmp.loc[tmp['item_id']==i].iloc[0,1:].rolling(90).mean().values,label=i)

plt.title('item_id mean trend')

plt.legend();
# 具体到某个店

cat_1_sales = train_sales.loc[train_sales['store_id'] == 'CA_1']

pd.crosstab(cat_1_sales['cat_id'],cat_1_sales['dept_id'])
dept_sum = cat_1_sales.groupby('dept_id').sum()

plt.figure(figsize=(16,12))

for i,row in dept_sum.iterrows():

    plt.plot(row.values,label=i)

plt.title(f'CA1 sales by dept')

plt.legend()

plt.show()
fig, ax = plt.subplots(4,3,figsize=(16,22))

plt.subplots_adjust(wspace=0.2,hspace=0.8)

for i, var in enumerate(["year", "weekday", "month", "event_name_1", "event_name_2", 

                         "event_type_1", "event_type_2", "snap_CA", "snap_TX", "snap_WI"]):

    i += 1

    plt.subplot(4,3,i)

    g = sns.countplot(calendar[var])

    g.set_xticklabels(g.get_xticklabels(), rotation=45)

    g.set_title(var)