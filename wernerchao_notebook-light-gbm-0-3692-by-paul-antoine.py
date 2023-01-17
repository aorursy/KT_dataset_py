import numpy as np

import pandas as pd

import lightgbm as lgb

IDIR = '../input/'
print('loading prior')

priors = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={

            'order_id': np.int32,

            'product_id': np.uint16,

            'add_to_cart_order': np.int16,

            'reordered': np.int8})



print('loading train')

train = pd.read_csv(IDIR + 'order_products__train.csv', dtype={

            'order_id': np.int32,

            'product_id': np.uint16,

            'add_to_cart_order': np.int16,

            'reordered': np.int8})



print('loading orders')

orders = pd.read_csv(IDIR + 'orders.csv', dtype={

        'order_id': np.int32,

        'user_id': np.int32,

        'eval_set': 'category',

        'order_number': np.int16,

        'order_dow': np.int8,

        'order_hour_of_day': np.int8,

        'days_since_prior_order': np.float32})



print('loading products')

products = pd.read_csv(IDIR + 'products.csv', dtype={

        'product_id': np.uint16,

        'order_id': np.int32,

        'aisle_id': np.uint8,

        'department_id': np.uint8},

        usecols=['product_id', 'aisle_id', 'department_id'])



print('priors {}: {}'.format(priors.shape, ', '.join(priors.columns)))

print('orders {}: {}'.format(orders.shape, ', '.join(orders.columns)))

print('train {}: {}'.format(train.shape, ', '.join(train.columns)))
### Construct product features ###



print('computing product f')

prods = pd.DataFrame()

prods['orders'] = priors.groupby(priors.product_id).size().astype(np.int32)

prods['reorders'] = priors.groupby('product_id')['reordered'].sum().astype(np.float32)

prods['reorder_rate'] = (prods.reorders / prods.orders).astype(np.float32)

products = products.join(prods, on='product_id')

products.set_index('product_id', drop=False, inplace=True)

del prods
products.head()
print('add order info to priors')

orders.set_index('order_id', inplace=True, drop=False)

priors = priors.join(orders, on='order_id', rsuffix='_')

priors.drop('order_id_', inplace=True, axis=1)

priors.head()