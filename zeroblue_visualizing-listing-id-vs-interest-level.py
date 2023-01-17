%matplotlib inline

import matplotlib.pylab as plt

import numpy as np

import pandas as pd

import seaborn as sns



train = pd.read_json('../input/train.json')

#train['listing_id'] -= train['listing_id'].min()



order = ['low', 'medium', 'high']

plt.figure(figsize=(8, 10))

plt.title("Listing ID vs Interest Level")

sns.stripplot(train['interest_level'],train['listing_id'],jitter=True, order=order)

plt.show()

train['created'] = pd.to_datetime(train['created'])

train['day_of_year'] = train['created'].dt.dayofyear



plt.figure(figsize=(13,10))

#plt.figure(figsize=(8, 10))

#plt.title("Listing ID vs Interest Level")

train['week_of_year'] = train['day_of_year'] // 7

sns.boxplot(train['week_of_year'], train['listing_id'], train['interest_level'], 

              hue_order=order)

plt.show()