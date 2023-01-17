import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pickle

import re

from IPython.core.display import display

from tqdm import tqdm_notebook as tqdm



%matplotlib inline

pd.options.mode.chained_assignment = None  # default='warn'
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
order_products_train_df = pd.read_csv("../input/order_products__train.csv")

order_products_prior_df = pd.read_csv("../input/order_products__prior.csv")

orders_df = pd.read_csv("../input/orders.csv")

products_df = pd.read_csv("../input/products.csv")

aisles_df = pd.read_csv("../input/aisles.csv")

departments_df = pd.read_csv("../input/departments.csv")
df_train = pd.merge(order_products_train_df, orders_df, how='left', on='order_id')

df_train = pd.merge(df_train, products_df, how='left', on='product_id')

df_train = pd.merge(df_train, aisles_df, how='left', on='aisle_id')

df_train = pd.merge(df_train, departments_df, how='left', on='department_id')

#df_train.to_csv('../input/df_train.csv', index=False) # if you want to use my feature, plz comment out.

df_train.head()
df_prior = pd.merge(order_products_prior_df, orders_df, how='left', on='order_id').head(10000) 

# if you want to use my feature, plz remove the ".head(10000)".

df_prior = pd.merge(df_prior, products_df, how='left', on='product_id')

df_prior = pd.merge(df_prior, aisles_df, how='left', on='aisle_id')

df_prior = pd.merge(df_prior, departments_df, how='left', on='department_id')

#df_prior.to_csv('../input/df_prior.csv', index=False) # if you want to use my feature, plz comment out.

df_prior.head()