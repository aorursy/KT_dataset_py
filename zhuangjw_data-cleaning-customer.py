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
df_cus = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_customers_dataset.csv')

df_cus
df_cus['customer_id'].unique().shape
df_cus['customer_unique_id'].unique().shape
df_cus['customer_zip_code_prefix'].unique().shape
df_geo = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_geolocation_dataset.csv')

df_geo
df_geo['geolocation_zip_code_prefix'].unique().shape
df_geo.drop_duplicates('geolocation_zip_code_prefix')
set_cuscode = set(df_cus['customer_zip_code_prefix'].unique())

set_geocode = set(df_geo['geolocation_zip_code_prefix'].unique())



set_cuscode.issubset(set_geocode), len(set_cuscode - set_geocode)
%%time

df_merge_geo = df_cus.merge(

    df_geo.drop_duplicates(

        'geolocation_zip_code_prefix').rename(

        columns={"geolocation_zip_code_prefix": "customer_zip_code_prefix"}

    ), 

    on="customer_zip_code_prefix", how='left'

)
df_merge_geo
df_merge_geo.shape[0] == df_cus.shape[0]  # same as original customer
df_ord = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_orders_dataset.csv')

df_ord
df_ord['customer_id'].unique().shape
df_merge_order = df_merge_geo.merge(

    df_ord, on="customer_id", how='left'

)

df_merge_order
order_time = pd.to_datetime(df_merge_order['order_purchase_timestamp'])

order_time
df_merge_order['order_purchase_timestamp'] = order_time
order_time.min(), order_time.max()
from datetime import datetime

order_time[(datetime(2018, 1, 1) < order_time) & (order_time < datetime(2018, 2, 1))]
df_merge_order[(datetime(2018, 1, 1) < order_time) & (order_time < datetime(2018, 2, 1))]