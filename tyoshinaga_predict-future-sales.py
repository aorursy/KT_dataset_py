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
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

# set index to ID to avoid droping it later

test  = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv').set_index('ID')
#処理に影響のない警告を非表示に

import warnings

warnings.filterwarnings('ignore')
items
shops
cats
train.head()
train.shape
max_item_cnt = train['item_cnt_day'].max()

min_item_cnt = train['item_cnt_day'].min()

print('最大売上個数：'+ str(int(max_item_cnt)))

print('最小売上個数：'+ str(int(min_item_cnt)))
df_date_cnt = train.groupby('date').count()['item_cnt_day']

df_date_cnt
date_ser = train['date']

date_list = list(date_ser)
year_list = []

month_block_list = []

day_list = []

month_block_day_list = []

for i in date_list:

    s = i.split('.')

    day_list.append(s[1])

    month_block_list.append(s[0])

    year_list.append(s[2])

    month_block_day_list.append(s[0]+s[1])
month_list = []

for i in month_block_list:

    i = int(i)

    if i <= 12:

        month_list.append(i)

    elif 13 <= i <= 24:

        s = i - 12

        month_list.append(s)

    elif 25 <= i <= 36:

        s = i - 24

        month_list.append(s)
item_price_ser = train['item_price']

item_price_list = list(item_price_ser)



item_cnt_day_ser = train['item_cnt_day']

item_cnt_day_list = list(item_cnt_day_ser)
sales_list = []

for i,v in zip(item_price_list,item_cnt_day_list):

    s = i * v

    sales_list.append(s)
year_ser = pd.Series(year_list)

month_block_ser = pd.Series(month_block_list)

day_ser = pd.Series(day_list)

month_block_day_ser = pd.Series(month_block_day_list)

month_ser = pd.Series(month_list)

sales_ser = pd.Series(sales_list)
df = pd.concat([train,year_ser],axis=1)

df = pd.concat([df,month_ser],axis=1)

df = pd.concat([df,day_ser],axis=1)

df = pd.concat([df,month_block_ser],axis=1)

df = pd.concat([df,month_block_day_ser],axis=1)

df = pd.concat([df,sales_ser],axis=1)
feature = ['date','date_block_num','shop_id','item_id','item_price','item_cnt_day','year','month','day','month_block','month_block+day','sales']
df.columns = feature
year_concat = list(df['year'])

month_concat = list(df['month'])

day_concat = list(df['day'])
import datetime

concat_ymd = []

for i,v,c in zip (year_concat,month_concat,day_concat):

    s = datetime.date(int(i),int(v),int(c))

    concat_ymd.append(s)

ser_ymd = pd.Series(concat_ymd)
df = pd.concat([df,ser_ymd],axis=1)
feature = ['date','date_block_num','shop_id','item_id','item_price','item_cnt_day','year','month','day','month_block','month_block+day','sales','datetime']
df.columns = feature

df
date_year_month = []

for i in aaa:

    y = i.year

    m = i.month

    ym = str(y) + '-' + str(m)

    date_year_month.append(ym)

date_year_month

ser_date_year_month = pd.Series(date_year_month)

df = pd.concat([df,ser_date_year_month],axis=1)
feature = ['date','date_block_num','shop_id','item_id','item_price','item_cnt_day','year','month','day','month_block','month_block+day','sales','datetime','year_month']

df.columns = feature

df
items_df = pd.merge(items,cats,on='item_category_id')

merge_df = pd.merge(df,items_df,on='item_id')

df = pd.merge(merge_df,shops,on='shop_id')

df
df.isnull().any(axis=0)
df.describe()
df.groupby('year').count()['item_cnt_day']
pd.options.display.float_format = '{:.20g}'.format

pd.options.display.float_format = '{:.2f}'.format
df.groupby('year').sum()['sales']
pd.options.display.float_format = '{:.0f}'.format

pd.pivot_table(df, index='year',columns='month',values='item_cnt_day',aggfunc=sum)
pd.pivot_table(df, index='year',columns='month',values='sales',aggfunc=sum)
#df = df.set_index('datetime')

#df
df_datesales = df.groupby('date_block_num').sum()['sales']

df_datesales
import matplotlib.pyplot as plt

import matplotlib as mpl

plt.figure(figsize=(24,12))

plt.plot(df_datesales)
a = range(0,34)

b = list(df['year_month'])
year_month_block = {}

for i,v in zip (a,b):

    year_month_block[i] = v

year_month_block