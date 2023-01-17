import pandas as pd
from datetime import datetime
train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
train['Date'] = pd.to_datetime(train['date'])
train.set_index('Date', inplace=True)
train.sort_index(inplace=True)
train.drop('date', axis=1, inplace=True)
train.head()
gb = train.groupby(['date_block_num', 'shop_id'])
by_shop = train.groupby('shop_id')
gb['item_price'].mean()
by_shop = train.groupby('shop_id')
t = train[train['shop_id'] < 10]
sns.violinplot(x='shop_id', y='item_price', data=t)
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv', index_col='ID')
test.head()
gb = train.groupby(['shop_id', 'item_id'])
avg_item_cnt = gb['item_cnt_day'].mean()
df = pd.DataFrame(avg_item_cnt, index=avg_item_cnt.index)
df.head()
def f(x):
    try:
        return avg_item_cnt.loc[(x.shop_id, x.item_id)]
    except:
        return 0
test['item_cnt_month'] = test[['shop_id', 'item_id']].apply(f, axis=1)
test
s = test['item_cnt_month']
import numpy as np

s = np.clip(s, 0, 20)
s.to_csv('/kaggle/working/s.csv')