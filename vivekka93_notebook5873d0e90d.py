# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Loading the data sets

aisles = pd.read_csv('../input/aisles.csv')

departments = pd.read_csv('../input/departments.csv')

order_products_prior = pd.read_csv('../input/order_products__prior.csv')

order_products_train = pd.read_csv('../input/order_products__train.csv')

orders = pd.read_csv('../input/orders.csv')

products = pd.read_csv('../input/products.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission.head(2)
sets = [('aisles', aisles), ('departments', departments), ('order_products_prior', order_products_prior), 

        ('order_products_train', order_products_train), ('orders', orders), ('products', products)]

# Observing the dimensions of data sets

for s in sets:

    print (s[0], s[1].shape)
# Observing initial entries of all data sets

for s in sets:

    print (s[0], '\n', s[1].head(2), '\n')
# Number of unique values in each column

for s in sets:

    for col in list(s[1].columns):

        print (s[0], col, s[1][col].nunique())
products = pd.merge(left = products, right = aisles, on = 'aisle_id', how = 'left')

products = pd.merge(left = products, right = departments, on = 'department_id', how = 'left')

products.head(5)
order_products_prior = pd.merge(left = order_products_prior, right = products, on = 'product_id', how = 'left')

order_products_train = pd.merge(left = order_products_train, right = products, on = 'product_id', how = 'left')

orders = pd.merge(left = orders, right = order_products_prior, on = 'order_id', how = 'left')

orders = pd.merge(left = orders, right = order_products_train, on = 'order_id', how = 'left')