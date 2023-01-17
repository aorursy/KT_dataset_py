import numpy as np

import pandas as pd

import gc

import os, psutil; process = psutil.Process(os.getpid())
# Dataframe print

from IPython.display import display, HTML

def show_dataframe(X, rows = 2):

    display(HTML(X.to_html(max_rows=rows)))

print('(Python) Memory usage ~ {:.2f} GB'.format(process.memory_info().rss / (float(1024**3))))
order_products_prior = pd.read_csv('../input/order_products__prior.csv', engine='c',

                                   dtype={'order_id':np.int32, 

                                          'product_id':np.int32, 

                                          'add_to_cart_order':np.int8, 

                                          'reordered':np.int8})



order_products_train = pd.read_csv('../input/order_products__train.csv', engine='c',

                                   dtype={'order_id':np.int32, 

                                          'product_id':np.int32, 

                                          'add_to_cart_order':np.int8, 

                                          'reordered':np.int8})

orders = pd.read_csv('../input/orders.csv', engine='c',

                     dtype={'order_id':np.int32,

                            'user_id':np.int32,

                            'order_number':np.int8,

                            'order_dow':np.int8,

                            'order_hour_of_day':np.int8

                           })

orders.days_since_prior_order.fillna(-1, inplace=True)

orders.days_since_prior_order = orders.days_since_prior_order.astype(np.int8)

products = pd.read_csv('../input/products.csv', engine='c',

                       dtype={'product_id':np.int32,

                              'aisle_id':np.int8,

                              'department_id':np.int8

                             })

aisles = pd.read_csv('../input/aisles.csv', engine='c', dtype={'aisle_id':np.int16})

departments = pd.read_csv('../input/departments.csv', engine='c', dtype={'department_id':np.int16})
show_dataframe(order_products_prior, 2); order_products_prior.info()

show_dataframe(order_products_train, 2); order_products_train.info()

show_dataframe(orders, 2); orders.info()

show_dataframe(products, 2); products.info()

show_dataframe(aisles, 2); aisles.info()

show_dataframe(departments, 2); departments.info()

gc.collect()

print('(Python) Memory usage ~ {:.2f} GB'.format(process.memory_info().rss / (float(1024**3))))
labels, levels = pd.factorize(orders.eval_set)

orders.eval_set = labels

orders.eval_set = orders.eval_set.astype(np.int8)

show_dataframe(orders, 2)

orders.info()

print('New values for \'eval_set\':', levels[0], '-', 0,  levels[1], '-', 1, levels[2], '-', 2)

print('(Python) Memory usage ~ {:.2f} GB'.format(process.memory_info().rss / (float(1024**3))))
order_products_users_prior = pd.merge(left=order_products_prior, right=orders, on='order_id', how='left')

show_dataframe(order_products_users_prior, 2)

order_products_users_prior.info()

gc.collect()

print('(Python) Memory usage ~ {:.2f} GB'.format(process.memory_info().rss / (float(1024**3))))
%%time

grouped = order_products_users_prior.groupby(['user_id', 'order_id']).product_id.apply(list)

grouped_df = pd.DataFrame(grouped)

grouped_df.columns = ['product_list']

show_dataframe(grouped_df, 10)

gc.collect()

print('(Python) Memory usage ~ {:.2f} GB'.format(process.memory_info().rss / (float(1024**3))))