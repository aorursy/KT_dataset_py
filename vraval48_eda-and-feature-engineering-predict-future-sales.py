import pandas as pd
import sklearn 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

from subprocess import check_output
train_raw_data = pd.read_csv('../input/sales_train.csv')
test_raw_data = pd.read_csv('../input/test.csv')
items_data = pd.read_csv('../input/items.csv')
item_categories_data = pd.read_csv('../input/item_categories.csv')
shops_data = pd.read_csv('../input/shops.csv')
train_raw_data.head()
test_raw_data.head()
items_data.head()
item_categories_data.head()
shops_data.head()
print(train_raw_data.shape)
print(test_raw_data.shape)
n_train = train_raw_data.shape[0]
n_test = test_raw_data.shape[0]
train_data = train_raw_data.set_index('item_id').join(items_data.set_index('item_id'), how='left').reset_index()
train_data = train_data.drop(['item_name'], axis=1)
train_data.loc[:, 'date'] = train_data['date'].map(lambda d: datetime.strptime(d, '%d.%m.%Y')).values
train_data.head(10)
n_ids = train_data.groupby('item_id')['shop_id'].count()
n_ids.plot()
n_cats = train_data.groupby('item_category_id')['item_id'].count()
plt.bar(n_cats.index, n_cats.values)
n_items_month = train_data.groupby(['date_block_num'])['item_id'].count()
n_items_month.plot(kind='bar')
unique_items_shops = train_data.groupby('shop_id')['item_id'].nunique()
unique_items_shops.plot(kind='bar')
def has_null(data):
    columns = data.columns
    nulls = []
    for column in data.columns:
        nulls.append(data[column].isnull().sum())
        
    null_df = pd.DataFrame({'column': columns, 'total null values': nulls})
    print(null_df)
has_null(train_data)
print('min item id {}'.format(train_data['item_id'].min()))
print('max item id {}'.format(train_data['item_id'].max()))
item_id_vc = train_data['item_id'].value_counts()
print(item_id_vc)
plt.plot(item_id_vc.index, item_id_vc, 'b.')
train_data_20949 = train_data.loc[train_data['item_id'] == 20949]
train_data_20949
train_data_20949['date_block_num'].value_counts()
itemid_shopid_month = train_data.groupby(['item_id', 'shop_id', 'date_block_num'])
train_data['item_cnt_day'].describe()
train_data.loc[train_data['item_cnt_day'] > 500]
train_data.loc[train_data['item_cnt_day'] < -10]
n_itemid_shopid_month = itemid_shopid_month['item_cnt_day'].count()
n_itemid_shopid_month.describe()
itemid_shopid_month_pos = train_data.loc[train_data['item_cnt_day'] > 0].groupby(['item_id', 'shop_id', 'date_block_num'])
train_data.loc[train_data['item_cnt_day'] > 0, ['item_id', 'shop_id', 'date_block_num', 'item_cnt_day']]
n_itemid_shopid_month_pos = itemid_shopid_month_pos['item_cnt_day'].count()
n_itemid_shopid_month_pos.describe()
n_itemid_shopid_month_pos
tot_itemid_shopid_month_pos = itemid_shopid_month_pos['item_cnt_day'].sum()
tot_itemid_shopid_month_pos.describe()
tot_itemid_shopid_month_pos
print(train_data['item_id'].describe())
print(train_data['shop_id'].describe())
print(test_raw_data['item_id'].describe())
print(test_raw_data['shop_id'].describe())
def find_unique_item_shops(data):
    item_shop_ids = set()
    for index, row in data.iterrows():
        item_shop = (row['item_id'], row['shop_id'], )
        if item_shop not in item_shop_ids:
            item_shop_ids.add(item_shop)

    return item_shop_ids
train_item_shop_ids = find_unique_item_shops(train_data)
test_item_shop_ids = find_unique_item_shops(test_raw_data)
len(train_item_shop_ids)
len(test_item_shop_ids)
cnt = 0
for item_shop_id in test_item_shop_ids:
    if item_shop_id not in train_item_shop_ids:
        print(item_shop_id)
        cnt += 1

print(cnt)
tot_itemid_shopid_month_pos.sort_index(inplace=True)
tot_itemid_shopid_month_pos_df = tot_itemid_shopid_month_pos.reset_index()
def get_33_sales(r): 
    try:
        return tot_itemid_shopid_month_pos.loc[(r['item_id'], r['shop_id'], 33)]
    except pd.core.indexing.IndexingError:
        return 0
    except KeyError:
        return 0

pred_prev_value_bench = test_raw_data.apply(get_33_sales, axis=1)
pred_prev_value_bench.describe()
pred_prev_value_bench_clipped = pred_prev_value_bench.clip(0, 20)
prev_value_bench_df = pd.DataFrame()
prev_value_bench_df['ID'] = test_raw_data['ID']
prev_value_bench_df['item_cnt_month'] = pred_prev_value_bench_clipped
prev_value_bench_df.to_csv('submission1.csv', header=True, index=False)
print(check_output(["ls", "../working"]).decode("utf8"))
# train_data['week_day'] = train_data['date'].dt.weekday
train_data_final = pd.DataFrame(train_data[train_data['item_cnt_day'] > 0].groupby(['date_block_num', 'item_id', 'shop_id'])['item_cnt_day'].sum())
train_data_final.columns = ['item_cnt_month']
returned_items_df = pd.DataFrame(train_data[train_data['item_cnt_day'] < 0].groupby(['date_block_num', 'item_id', 'shop_id'])['item_cnt_day'].sum())
returned_items_df.columns = ['item_cnt_rtn_last_month']
returned_items_df.reset_index(inplace=True)
returned_items_df['item_cnt_rtn_last_month'] *= -1
returned_items_df['date_block_num'] += 1
train_data_final = train_data_final.join(returned_items_df.set_index(['date_block_num', 'item_id', 'shop_id']), how='left')
train_data_final.loc[:, 'item_cnt_rtn_last_month'] = train_data_final.loc[:, 'item_cnt_rtn_last_month'].fillna(0)
train_data_final.sort_index(inplace=True)
train_data_final.loc[:, 'first_month'] = 0
train_data_final.loc[(0), 'first_month'] = 1
total_items_df = pd.DataFrame(train_data[train_data['item_cnt_day'] > 0].groupby(['date_block_num', 'item_id', 'shop_id'])['item_cnt_day'].sum())
total_items_df.columns = ['item_cnt_sold_last_month']
total_items_df.reset_index(inplace=True)
total_items_df['date_block_num'] += 1
train_data_final = train_data_final.join(total_items_df.set_index(['date_block_num', 'item_id', 'shop_id']), how='left')
train_data_final.loc[:, 'item_cnt_sold_last_month'].fillna(0, inplace=True)
items_sold_year_ago_df = train_data_final.reset_index()
items_sold_year_ago_df['date_block_num'] += 12
items_sold_year_ago_df.drop(items_sold_year_ago_df[items_sold_year_ago_df['date_block_num'] > 34].index, axis=0, inplace=True)
items_sold_year_ago_df.drop(['item_cnt_rtn_last_month', 'first_month', 'item_cnt_sold_last_month'], axis=1, inplace=True)
items_sold_year_ago_df.rename(columns={'item_cnt_month': 'item_cnt_prev_year'}, inplace=True)
train_data_final = train_data_final.join(items_sold_year_ago_df.set_index(['date_block_num', 'item_id', 'shop_id']), how='left')
train_data_final.fillna(0, inplace=True)
train_data_final
