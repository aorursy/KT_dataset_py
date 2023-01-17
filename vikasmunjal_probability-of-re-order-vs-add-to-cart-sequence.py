import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data plotting

%matplotlib inline 
aisle_df = pd.read_csv('../input/aisles.csv')

departments_df = pd.read_csv('../input/departments.csv')

prior_df = pd.read_csv('../input/order_products__prior.csv')

train_df = pd.read_csv('../input/order_products__train.csv')

orders_df = pd.read_csv('../input/orders.csv')

products_df = pd.read_csv('../input/products.csv')
train_df.head()

temp = train_df.groupby(['add_to_cart_order','reordered'])

temp=temp.size().unstack()

temp =temp.head(30)

temp.plot(kind = "bar", rot = 45, stacked = False )
temp = prior_df[prior_df.reordered == 1]

temp = temp.product_id.value_counts()

temp = temp.head(30)

temp.plot(kind = "bar", rot = 45, stacked = False)