# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

% matplotlib inline

import tensorflow as tf

tf.reset_default_graph()

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.contrib import learn

from sklearn import cross_validation

from sklearn import preprocessing

from sklearn import metrics

from __future__ import print_function

from matplotlib.dates import YearLocator, MonthLocator, DateFormatter

import datetime



%matplotlib inline
df=pd.read_csv('../input/bitcoin_price.csv',parse_dates=['Date'])
df.head()
df.describe()
print(df.Date.max())

print(df.Date.min())
date1 = datetime.date(2013,4, 28)

date2 = datetime.date(2015, 8, 15)

years = YearLocator()   # every year

months = MonthLocator()  # every month

yearsFmt = DateFormatter('%Y')

fig, ax = plt.subplots()

ax.plot_date(df.Date,df.Close, '-')

# format the ticks

ax.xaxis.set_major_locator(years)

ax.xaxis.set_major_formatter(yearsFmt)

ax.xaxis.set_minor_locator(months)

ax.autoscale_view()