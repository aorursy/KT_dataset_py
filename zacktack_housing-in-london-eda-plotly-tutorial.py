import numpy as np

import pandas as pd

from pandas_profiling import ProfileReport

import plotly

import plotly.express as px

from plotly.offline import iplot

import cufflinks as cf
df_month = pd.read_csv("/kaggle/input/housing-in-london/housing_in_london_monthly_variables.csv")

df_month.date = pd.to_datetime(df_month.date)
df_month.tail()
df_month.isnull().sum() / df_month.shape[0]
# ProfileReport(df_month)
#help(px.line)
px.line(df_month,x='date',y='average_price',color='area',title='Average Price by Area Trend')
kc_2019 = df_month[(df_month.area == 'kensington and chelsea') & (df_month.date.dt.year == 2019)].average_price.mean()

kc_1995 = df_month[(df_month.area == 'kensington and chelsea') & (df_month.date.dt.year == 1995)].average_price.mean()

display(round(kc_2019 - kc_1995,2), str(round((kc_2019 / kc_1995 * 100),2)) + '%')
px.line(df_month.groupby(['date','borough_flag']).average_price.mean().to_frame().reset_index(),x='date',y='average_price',color='borough_flag',title='Average Housing Price by Borough_Flag Trend')
px.line(df_month,x='date',y='houses_sold',color='area',title='Number of Houses Sold Trend')
px.line(df_month,x='date',y='no_of_crimes',color='area',title='Crime')
df_month
px.bar(df_month.groupby("borough_flag").area.nunique().reset_index(),x="borough_flag",y="area")
houses_2019 = df_month[df_month.date.dt.year == 2019].groupby("area").houses_sold.sum().reset_index()

px.bar(houses_2019,x="area",y="houses_sold")
df_month['total'] = df_month.average_price * df_month.houses_sold

total_2019 = df_month[df_month.date.dt.year == 2019].groupby('area').total.sum().reset_index()

print("Total value was %s GBP" % '{:,.2f}'.format(total_2019.total.sum()))
px.pie(total_2019,values='total',names='area')