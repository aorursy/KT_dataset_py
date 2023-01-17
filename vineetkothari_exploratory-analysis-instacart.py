# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.plotly as py

from functools import reduce

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
pd_aisles = pd.read_csv("../input/aisles.csv")

pd_departments = pd.read_csv("../input/departments.csv")

pd_product_prior_ord = pd.read_csv("../input/order_products__prior.csv")

pd_products_train = pd.read_csv("../input/order_products__train.csv")

pd_orders = pd.read_csv("../input/orders.csv")

pd_products = pd.read_csv("../input/products.csv")
pd_orders.head()
fig, ax = plt.subplots()

# the histogram of the data

n, bins, patches = ax.hist(pd_orders.order_hour_of_day,bins='auto',normed=2,facecolor='g')

ax.set_ylabel('Count')

ax.set_xlabel('No of Hours')

ax.set_title(r'Histogram')

# Tweak spacing to prevent clipping of ylabel

fig.tight_layout()

plt.show()   #plot shows people order on 8-8:30 with highest frequecy.
fig, ax = plt.subplots()

# the histogram of the data

n, bins, patches = ax.hist(pd_orders.order_dow,bins='auto',normed=1,facecolor='b') 

ax.set_ylabel('Count')

ax.set_xlabel('Days of week')

ax.set_title(r'Histogram')

# Tweak spacing to prevent clipping of ylabel

fig.tight_layout()

plt.show()   #plot shows most orders are on 0 and 1 days of weekend.
fig, ax = plt.subplots()

#there are Nan values so datacleaning

pd_orders.days_since_prior_order = pd_orders.days_since_prior_order.fillna(round(pd_orders.days_since_prior_order.median()))

# the histogram of the data

n, bins, patches = ax.hist(pd_orders.days_since_prior_order,bins='auto',normed=1,facecolor='y') 

ax.set_ylabel('Count')

ax.set_xlabel('Preorder')

ax.set_title(r'Histogram')

plt.setp(patches[len(patches) - 1], 'facecolor', 'g')

# Tweak spacing to prevent clipping of ylabel

fig.tight_layout()

plt.show()   #plot shows higest order after 30 days .(not sure please clearify me if wrong)
#pd_orders.loc[pd_orders['eval_set'] == "prior"].order_number.value_counts()[:3]    1 2 and 3 are 

                                                                               #coming most freq

x = pd_orders.loc[pd_orders['eval_set'] == "prior"].order_number.value_counts()

plt.plot(x,color='m')

plt.show()
#Find the maximum preorder it is 3

fig, ax = plt.subplots()

n_max = x.argmax()   #find maximum value of preorder count

ax.plot(x[n_max],x[n_max],'o') #plot the subplot 

n_min = x.argmin()   #find minimum value of preorder count

ax.plot(x[n_min],x[n_min],'x') #plot the subplot

ax.set_xlabel('order numbers')

ax.set_ylabel('count numbers')

plt.show()
#pd_products_train.head()

x = pd_products_train.groupby('order_id').agg({'add_to_cart_order': [max, 'count']})

x.mode()
#pd_product_prior_ord.head()

x = pd_product_prior_ord.groupby('order_id').agg({'add_to_cart_order': [max, 'count']})

x.mode()
pd_products_train.head()
#count_product_topn = pd_products_train.groupby('product_id').product_id.value_counts().nlargest(10)

df = pd_products_train.groupby('product_id')

df_count = df.product_id.count()

df_reset_index = df_count.reset_index(name='count').sort_values(['count'], ascending=False)

topn = df_reset_index.head(11)

topn
df_topn = pd.merge(pd_products, topn, on='product_id', how='inner',sort=False) #count of bananas is the higest

df_topn.head(10)
plt.xticks(df_topn['count'], list(df_topn['product_name']),rotation='vertical', fontsize=20)

plt.plot(df_topn['count'],np.arange(11))

plt.show()
df_reordered = pd_products_train.groupby('reordered').count()

df_reordered.head() #59% of the ordered items are reorders.
df_reordered.plot(kind='bar')

plt.show()
df_reordered = pd_products_train.groupby(['product_id'])

df_mean = df_reordered.reordered.mean()

df_reset_index_mean = df_mean.reset_index(name='mean')
prob_reordered = df_reset_index_mean.merge(pd_products, left_on='product_id', right_on='product_id', how='outer')

prob_reordered.head()
# Calculate the Prior : p(reordered|product_id)

prior = pd.DataFrame(pd_product_prior_ord.groupby('product_id')['reordered'].agg([('number_of_orders',len),

                                                                                  ('sum_of_reorders','sum')]))

prior['prior_p'] = (prior['sum_of_reorders']+1)/(prior['number_of_orders']+2)

prior.drop(['number_of_orders','sum_of_reorders'], axis=1, inplace=True)

print('Here is The Prior: our first guess of how probable it is that a product be reordered.')

prior.head(3)
pd_aisles.head()
pd_departments.head()
pd_product_prior_ord.head()
pd_products_train.head()
pd_orders.head()
pd_products.head()

