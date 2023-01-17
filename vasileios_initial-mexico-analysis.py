# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
mexico = pd.read_csv('../input/mex_clean.csv', encoding='utf-8')



# remove useless columns

mexico.drop(['id', 'store_and_fwd_flag'], axis = 1, inplace=True)
mexico.head()
mexico['pickup_datetime'] = pd.to_datetime(mexico.pickup_datetime)

mexico['dropoff_datetime'] = pd.to_datetime(mexico.dropoff_datetime)
mexico['Day'] = mexico['pickup_datetime'].dt.weekday_name

mexico['Year'] = mexico['pickup_datetime'].dt.year

mexico['Month'] = mexico['pickup_datetime'].dt.month
mexico.set_index('pickup_datetime', inplace = True)
fig, ax = plt.subplots(figsize = (15, 8))

ax = sns.countplot(x = mexico['vendor_id'], data = mexico)



ncount = len(mexico)

for p in ax.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 

            ha='center', va='bottom') # set the alignment of the text

ax.set_title('Distribution of vendors in Mexico City');
fig, ax = plt.subplots(figsize = (15, 8))

ax = sns.barplot(x = 'Day', y = 'wait_sec', data = mexico, ci = False);
fig, ax = plt.subplots(figsize = (15, 8))

ax = sns.barplot(x = 'Day', y = 'wait_sec', hue = 'vendor_id', data = mexico, ci = None);
fig, ax = plt.subplots(figsize = (15, 8))

ax = sns.barplot(x = 'Day', y = 'dist_meters', data = mexico);
fig, ax = plt.subplots(figsize = (15, 8))

ax = sns.barplot(x = 'Day', y = 'dist_meters', hue = 'vendor_id', data = mexico, ci = None);
mexico['dist_meters'].resample('W').sum().plot();
mexico['trip_duration'].resample('W').sum().plot();