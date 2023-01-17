import pandas as pd

import numpy

import seaborn as sns

import matplotlib.pyplot as plt

import fbprophet

import plotly

import warnings

import plotly.graph_objs as go 

import fbprophet



from datetime import datetime



plt.style.use("ggplot")

warnings.filterwarnings("ignore")
item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv',encoding='UTF-8',sep=',',

error_bad_lines=False,low_memory=False,parse_dates=True)



item_categories.head()
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv',encoding='UTF-8',sep=',',

error_bad_lines=False,low_memory=False,parse_dates=True)

items.head()

sales_train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv',encoding='UTF-8',sep=',',

error_bad_lines=False,low_memory=False,parse_dates=True)



sales_train.head()
sample_submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv',encoding='UTF-8',sep=',',

error_bad_lines=False,low_memory=False,parse_dates=True)



sample_submission.head()
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv',encoding='UTF-8',sep=',',

error_bad_lines=False,low_memory=False,parse_dates=True)



shops.head()
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv',encoding='UTF-8',sep=',',

error_bad_lines=False,low_memory=False,parse_dates=True)



test.head()
df = sales_train.iloc[:,[0,4]]

df.columns = ['Date','Price']

df.Date = pd.to_datetime(df.Date)

df = df.groupby(by=['Date'],sort=True,as_index=False).count()

df.tail()
df = df.rename(columns={'Date': 'ds', 'Price': 'y'})

fbp = fbprophet.Prophet()

fbp.fit(df)
df_forecast = fbp.make_future_dataframe(periods=4,freq='M')

df_forecast = fbp.predict(df_forecast)

fbp.plot(df_forecast, xlabel = 'Tarih', ylabel = 'Toplam Harcama')

plt.title('4 Aylık Toplam Harcama Grafiği Ve Zaman Serisi')
holiday = pd.DataFrame({

  'ds': pd.to_datetime(['2013-01-01', '2015-01-01']),

  'holiday': 'start',

  'lower_window': 0,

  'upper_window': 3,

})

holiday.head()
fbp = fbprophet.Prophet(holidays=holiday)

fbp.fit(df)

df_forecast = fbp.make_future_dataframe(periods=12,freq='M')

df_forecast2 = fbp.predict(df_forecast)

fbp.plot(df_forecast2, xlabel = 'Tarih', ylabel = 'Toplam Harcama')

plt.title('6 Aylık Toplam Harcama Grafiği Ve Zaman Serisi')
fbp.plot_components(df_forecast2)