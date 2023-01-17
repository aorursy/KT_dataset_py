import numpy as np

import pandas as pd 

import matplotlib.pyplot as pl

import seaborn as sns

%matplotlib inline



# Read in files

aisles = pd.read_csv('../input/aisles.csv')

departments =  pd.read_csv('../input/departments.csv')

order_products_prior =  pd.read_csv('../input/order_products__prior.csv')

order_products_train =  pd.read_csv('../input/order_products__train.csv')

products = pd.read_csv('../input/products.csv')

orders = pd.read_csv('../input/orders.csv')
# Filter for only Prior data in the orders

orders_prior = orders.loc[orders['eval_set'] == 'prior']
# Plot the days that people most likely order

orders_day = orders_prior.groupby(['order_dow']).size()

orders_by_day = orders_day.plot(kind='bar',title="Total Sales by Day")

orders_by_day.set_xlabel("Day")

orders_by_day.set_ylabel("No. Orders")
# Plot the hours people most likley order

orders_hour = orders_prior.groupby(['order_hour_of_day']).size()

orders_by_hour = orders_hour.plot(kind='bar',title="Total Sales by Hour")

orders_by_hour.set_xlabel("Hour")

orders_by_hour.set_ylabel("No. Orders")
# Plot time to reorder

orders_reorder = orders_prior.groupby(['days_since_prior_order']).size()

orders_by_reorder = orders_reorder.plot(kind='bar',title="Days to Reorder")

orders_by_reorder.set_xlabel("Days")

orders_by_reorder.set_ylabel("No. Orders")
# Get the distibution number of products that are ordered

products_order = order_products_prior.groupby(["order_id"]).size()

products_by_order = sns.violinplot(x=products_order,scale="width")

products_order.mean()
# Merge orders and products 

orders_prior_by_products = order_products_prior.merge(orders_prior)[['order_id','product_id','reordered']]

orders_grouped_product = orders_prior_by_products.groupby(["product_id"])["reordered"].aggregate({'Re-orders': 'count'}).reset_index()

orders_grouped_products = orders_grouped_product.merge(products).sort_values(by=['Re-orders'], ascending=False)

orders_grouped_products.head(20)
