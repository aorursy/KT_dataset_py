# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb



color = sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment = None  # default='warn'

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import os.path
order_products_train_df = pd.read_csv("../input/order_products__train.csv")

order_products_prior_df = pd.read_csv("../input/order_products__prior.csv")

orders_df = pd.read_csv("../input/orders.csv")

products_df = pd.read_csv("../input/products.csv")

aisles_df = pd.read_csv("../input/aisles.csv")

departments_df = pd.read_csv("../input/departments.csv")
joined_all = order_products_train_df.merge(products_df).merge(aisles_df).merge(departments_df)

#joined_all.size()

joined_all2 = pd.DataFrame(joined_all.iloc[:10000])
#features = ['reordered','add_to_cart_order','aisle_id','department_id']

features = ['reordered','add_to_cart_order']

f = joined_all2[features].values

# response = ['order_id','product_id']

response = ['product_id']

labels = joined_all2[response].values

d_train = lgb.Dataset(f,label = labels)

params = {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': {'binary_logloss'},

    'num_leaves': 96,

    'max_depth': 10,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.95,

    'bagging_freq': 5

}

ROUNDS = 100
bst = lgb.train(params,d_train,ROUNDS)

lgb.plot_importance(bst, figsize=(9,20))
bst.feature_importance()