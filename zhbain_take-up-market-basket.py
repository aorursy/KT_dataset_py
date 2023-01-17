import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set(style='whitegrid', context='notebook', palette='Set2')

sns.despine()



import os, gc, psutil

process = psutil.Process(os.getpid())

import warnings

warnings.filterwarnings('ignore')



#load inputFiles



depart = pd.read_csv('../input/departments.csv', engine='c',

							 dtype={'department_id':np.int8})   

aisles = pd.read_csv('../input/aisles.csv', engine='c',

						dtype={'aisle_id':np.uint8})

products = pd.read_csv('../input/products.csv', engine='c',

                       	  dtype={'product_id':np.uint16,

                                 'aisle_id':np.uint8,

                                 'department_id':np.int8})

orders = pd.read_csv('../input/orders.csv', engine='c',

                        dtype={'order_id':np.int32,

                               'user_id':np.int32,

                               'order_number':np.int8,

                               'order_dow':np.int8,

                               'order_hour_of_day':np.int8})

op_train = pd.read_csv('../input/order_products__train.csv', engine='c',

                          dtype={'order_id':np.int32, 

                                 'product_id':np.uint16, 

                                 'add_to_cart_order':np.int8, 

                                 'reordered':np.int8})

op_prior = pd.read_csv('../input/order_products__prior.csv', engine='c',

                          dtype={'order_id':np.int32, 

                                 'product_id':np.uint16, 

                                 'add_to_cart_order':np.int8, 

                                 'reordered':np.int8})





print('Memory usage ~ {:.2f} GB'.format(process.memory_info().rss / (float(1024**3))))
depart.info()

print('-'*40)

aisles.info()

print('-'*40)

products.info()

print('-'*40)

orders.info()

print('-'*40)

op_train.info()

print('-'*40)

op_prior.info()
orders.isnull().sum()
orders[orders['days_since_prior_order'].isnull()]
op_prior.isnull().sum()
print('Memory usage ~ {:.2f} GB'.format(process.memory_info().rss / (float(1024**3))))
orders.describe()
depart.head()
aisles.head()
products.head()
orders.head()
op_train.head()
op_prior.head()
gc.collect()
orders_test = orders[orders['eval_set']=='test']

orders_test.info()
orders_train = orders[orders['eval_set']=='train']
op_train = pd.merge(op_train, orders, on='order_id')

op_train.info()
op_prior = pd.merge(op_prior, orders, on='order_id')

op_prior.info()
del orders
gc.collect()

print('Memory usage ~ {:.2f} GB'.format(process.memory_info().rss / (float(1024**3))))
op_prior_train = op_prior[op_prior['user_id'].isin(orders_train['user_id'])]

op_prior_train.info()



train_y = pd.merge(op_prior_train[['user_id', 'product_id']], op_train, how='left', on=['user_id', 'product_id'])['reordered']

train_y = train_y.fillna(0).astype(np.int8)
gc.collect()

print('Memory usage ~ {:.2f} GB'.format(process.memory_info().rss / (float(1024**3))))
train_X = pd.merge(op_prior_train, orders_train, on='user_id', suffixes=['_P', '_T'])

train_X.info()
op_prior_test = op_prior[op_prior['user_id'].isin(orders_test['user_id'])]

op_prior_test.info()

test_X = pd.merge(op_prior_test, orders_test, on='user_id', suffixes=['_P', '_T'])

test_X.info()
gc.collect()

print('Memory usage ~ {:.2f} GB'.format(process.memory_info().rss / (float(1024**3))))
train_X['y'] = pd.Series(train_y)
train_X.info()
train_X
gc.collect()

print('Memory usage ~ {:.2f} GB'.format(process.memory_info().rss / (float(1024**3))))
train_X.isnull().sum()
gc.collect()

print('Memory usage ~ {:.2f} GB'.format(process.memory_info().rss / (float(1024**3))))
train_X = pd.merge(train_X, products, how='left', on='product_id')