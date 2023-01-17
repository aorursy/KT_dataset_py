import pandas as pd

import numpy as np

import seaborn as sns

import datetime
sales = pd.read_csv('../input/sales-data/sales_train_v2.csv')

test = pd.read_csv('../input/sales-data/test.csv')

sample_submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

sales.head(1)
items.head(1)
item_categories.head(1)
shops.head(1)
df1 = pd.merge(sales,items, on='item_id')

df2 = pd.merge(df1,item_categories, on='item_category_id')

df = pd.merge(df2,shops, on='shop_id')

df.head()
df.shape
df.dtypes
df.isnull().sum()