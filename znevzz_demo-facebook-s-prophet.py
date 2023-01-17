%matplotlib inline

import pandas as pd

import numpy as np

import fbprophet



fbprophet.__version__
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os



kaggle_datasets = []

path = "../input/fbprophet-sample-1/"

kaggle_datasets.extend(os.listdir(path))
df = pd.read_csv(path + kaggle_datasets[0]) 

df.describe() 

# df.head()

df.tail()
import logging

logging.getLogger('fbprophet').setLevel(logging.ERROR)

import warnings

warnings.filterwarnings("ignore")
partitioned = df[:-364]

partitioned.tail()
m = fbprophet.Prophet()

m.fit(partitioned)

future = m.make_future_dataframe(periods=366)

# m.predict(future)

future.tail()

m = fbprophet.Prophet(changepoints=['2012-01-31', '2014-01-31'])

m.fit(partitioned)

future = m.make_future_dataframe(periods=366)

# m.predict(future)

future.tail()

# Python

playoffs = pd.DataFrame({

  'holiday': 'playoff',

  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',

                        '2010-01-24', '2010-02-07', '2011-01-08',

                        '2013-01-12', '2014-01-12', '2014-01-19',

                        '2014-02-02', '2015-01-11', '2016-01-17',

                        '2016-01-24', '2016-02-07']),

  'lower_window': 0,

  'upper_window': 1,

})

superbowls = pd.DataFrame({

  'holiday': 'superbowl',

  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),

  'lower_window': 0,

  'upper_window': 1,

})

holidays = pd.concat((playoffs, superbowls))



m = fbprophet.Prophet(holidays=holidays)

m.add_country_holidays(country_name='US')



m.fit(partitioned)

future = m.make_future_dataframe(periods=366)

future.tail()
forecast = m.predict(future)

forecast.tail()
# m.plot(forecast)

from fbprophet.plot import add_changepoints_to_plot



fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
from fbprophet.plot import plot_yearly



a = plot_yearly(m)
fig = m.plot_components(forecast)
original = df.loc[-365:]

original.tail()
predicted = forecast[['ds','yhat']]

predicted.tail()
# Calculate root mean squared error.

diff = forecast.yhat - original.y

np.sqrt(np.mean(diff**2))
# !curl --get https://www.kaggle.com/znevzz
forecast.head(2)