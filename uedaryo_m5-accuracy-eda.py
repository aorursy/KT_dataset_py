# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DIR = '../input/m5-forecasting-accuracy'
calendar = pd.read_csv(f'{DIR}/calendar.csv')
selling_prices = pd.read_csv(f'{DIR}/sell_prices.csv')
sample_submission = pd.read_csv(f'{DIR}/sample_submission.csv')
sales_train_val = pd.read_csv(f'{DIR}/sales_train_validation.csv')
sales_train_ev = pd.read_csv(f'{DIR}/sales_train_evaluation.csv')
calendar[1:30]
calendar.tail()
calendar['event_name_1'].unique()
calendar['event_type_1'].unique()
calendar['event_name_2'].unique()
calendar['event_type_2'].unique()
selling_prices.head()
selling_prices.shape
sample_submission.head()
sample_submission.shape
sales_train_val.head()
sales_train_ev.head()
import matplotlib.pyplot as plt
## 日付のカラム
d_cols = [c for c in sales_train_val.columns if 'd_' in c]
## 'HOBBIES_1_001_CA_1_validation'のみ取り出す
## グラフ化するためにset_indexでidをindexにし、日付のカラムのみ選択し、転置
sales_train_val.loc[sales_train_val['id'] == 'HOBBIES_1_200_CA_3_validation'] \
    .set_index('id')[d_cols] \
    .T \
    .plot(figsize=(15, 5))
sales_train_val.loc[sales_train_val['id'] == 'FOODS_1_001_TX_1_validation'] \
    .set_index('id')[d_cols] \
    .T \
    .plot(figsize=(15, 5))
## 上と同様に転置
example = sales_train_val.loc[sales_train_val['id'] == 'HOBBIES_1_200_CA_3_validation'].set_index('id')[d_cols].T
example
## indexを振り直し、ジョインするためにカラム名を変更
example = example.reset_index().rename(columns={'index': 'd'})
example
## カラムdをキーにジョイン
example = example.merge(calendar, how='left', validate='1:1')
example
## グラフ化するためにdateをindexに指定し、販売個数をプロット
example.set_index('date')['HOBBIES_1_200_CA_3_validation'].plot(figsize=(15, 5))
## 週、月、年ごとにgroupbyし、販売個数の平均値を可視化する
example.groupby('wday').mean()['HOBBIES_1_200_CA_3_validation']

## 週や月で平均すると2011〜2012の0に引っ張られているが、それ以降の傾向は把握できそう
### グラフ表示
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3))
example.groupby('wday').mean()['HOBBIES_1_200_CA_3_validation'].plot(ax=ax1)
example.groupby('month').mean()['HOBBIES_1_200_CA_3_validation'].plot(ax=ax2)
example.groupby('year').mean()['HOBBIES_1_200_CA_3_validation'].plot(ax=ax3)
## 商品価格のテーブルの可視化
selling_prices = pd.read_csv(f'{DIR}/sell_prices.csv')
selling_prices.head()
## _で分割し、categoryカラムを生成。expand=Trueを指定するとDataFrame型としてカラムに分割される。
selling_prices['category'] = selling_prices['item_id'].str.split('_',expand=True)[0]

selling_prices.head()
selling_prices['region'] = selling_prices['store_id'].str.split('_',expand=True)[0]
selling_prices.groupby('category')
selling_prices['region'].unique()
## カテゴリ、地域ごとにヒストグラムを表示

for region in selling_prices['region'].unique():
    i = 0 
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for category, df in selling_prices[selling_prices['region'] == region].groupby('category'):
        ax = df['sell_price'].apply(np.log1p) \
            .plot(kind='hist',bins=20,title=f'Distribution of {category} prices in {region}',ax=axs[i])
        ax.set_xlabel('Log(price)')
        i += 1
    plt.tight_layout()
## 全ての商品に対して日付をジョインしたテーブルを作成
## indexを日付にしておく
past_sales = sales_train_val.set_index('id')[d_cols] \
    .T \
    .merge(calendar.set_index('d')['date'],
           left_index=True,
           right_index=True,
            validate='1:1') \
    .set_index('date')
past_sales
sales_train_val['cat_id'].unique()
##　カテゴリごとの販売個数のトレンド
for category in sales_train_val['cat_id'].unique():
    ## カラム選択
    items_col = [c for c in past_sales.columns if category in c]
    ## 行の販売個数を足し合わせる
    past_sales[items_col] \
        .sum(axis=1)\
    .plot(figsize=(15, 5))
plt.legend(sales_train_val['cat_id'].unique())
plt.show()
##　1ヶ月移動平均
for category in sales_train_val['cat_id'].unique():
    ## カラム選択
    items_col = [c for c in past_sales.columns if category in c]
    ## 行の販売個数を足し合わせる
    ## 60日の移動平均を指定
    past_sales[items_col] \
        .sum(axis=1)\
        .rolling(30).mean() \
    .plot(figsize=(15, 5))
plt.legend(sales_train_val['cat_id'].unique())
plt.show()

selling_prices['store_id'].unique()
## 30日移動平均
for store in selling_prices['store_id'].unique():
    store_items = [c for c in past_sales.columns if store in c]
    past_sales[store_items] \
        .sum(axis=1) \
        .rolling(30).mean() \
        .plot(figsize=(15, 5),title=store)
plt.legend(selling_prices['store_id'].unique())
## 30日移動平均,x軸を共有
fig, axes = plt.subplots(5, 2, figsize=(20, 10), sharex=True)
## 1次元配列に
axes = axes.flatten()
i = 0 
for store in selling_prices['store_id'].unique():
    store_items = [c for c in past_sales.columns if store in c]
    past_sales[store_items] \
        .sum(axis=1) \
        .rolling(30).mean() \
        .plot(figsize=(15, 5),title=store, ax=axes[i])
    i += 1
plt.show()
## カリフォルニア
snap_ca_list = calendar[calendar['snap_CA'] == 1]['date'].tolist()
values = []
labels = np.array(['Hobbies', 'Hobbies not','Household', 'Household not', 'Foods', 'Foods not'])

## カテゴリごとの販売個数平均
for category in sales_train_val['cat_id'].unique():
    ## カラム選択
    items_col = [c for c in past_sales.columns if category in c and 'CA' in c ]
    values.append(past_sales.query('index in @snap_ca_list')[items_col].sum(axis=1).mean(axis=0))
    values.append(past_sales.query('index not in @snap_ca_list')[items_col].sum(axis=1).mean(axis=0))

fig = plt.figure(figsize=(8.0, 6.0))
plt.bar(np.arange(len(labels)),values, tick_label=labels)
plt.show()
print(values)
## テキサス
snap_tx_list = calendar[calendar['snap_TX'] == 1]['date'].tolist()
values = []
labels = np.array(['Hobbies', 'Hobbies not','Household', 'Household not', 'Foods', 'Foods not'])

## カテゴリごとの販売個数平均
for category in sales_train_val['cat_id'].unique():
    ## カラム選択
    items_col = [c for c in past_sales.columns if category in c and 'TX' in c ]
    values.append(past_sales.query('index in @snap_tx_list')[items_col].sum(axis=1).mean(axis=0))
    values.append(past_sales.query('index not in @snap_tx_list')[items_col].sum(axis=1).mean(axis=0))

fig = plt.figure(figsize=(8.0, 6.0))
plt.bar(np.arange(len(labels)),values, tick_label=labels)
plt.show()
print(values)
## カリフォルニア
snap_wi_list = calendar[calendar['snap_WI'] == 1]['date'].tolist()
values = []
labels = np.array(['Hobbies', 'Hobbies not','Household', 'Household not', 'Foods', 'Foods not'])

## カテゴリごとの販売個数平均
for category in sales_train_val['cat_id'].unique():
    ## カラム選択
    items_col = [c for c in past_sales.columns if category in c and 'CA' in c ]
    values.append(past_sales.query('index in @snap_wi_list')[items_col].sum(axis=1).mean(axis=0))
    values.append(past_sales.query('index not in @snap_wi_list')[items_col].sum(axis=1).mean(axis=0))

fig = plt.figure(figsize=(8.0, 6.0))
plt.bar(np.arange(len(labels)),values, tick_label=labels)
plt.show()
print(values)