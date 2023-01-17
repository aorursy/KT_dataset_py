import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import datetime
import matplotlib
import re
from scipy import stats

matplotlib.rcParams['figure.figsize'] = (10,5)
matplotlib.rcParams['font.size'] = 12

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import random
random.seed(1)
import time
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
print('Done!')
(market_train_orig, news_train_orig) = env.get_training_data()
market_train_df = market_train_orig.copy()
news_train_df = news_train_orig.copy()
print('Market train shape: ', market_train_df.shape)
print('News train shape: ', news_train_df.shape)
market_train_df.describe()
news_train_df.describe()
print('Check null data:')
market_train_df.isna().sum()
# *Preprocessing*: fill NAN data in the market adjusted columns with the raw value data
column_market = ['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10','returnsOpenPrevMktres10']
column_raw = ['returnsClosePrevRaw1','returnsOpenPrevRaw1','returnsClosePrevRaw10','returnsOpenPrevRaw10']
for i in range(len(column_raw)):
    market_train_df[column_market[i]] = market_train_df[column_market[i]].fillna(market_train_df[column_raw[i]])
market_train_df.head()
# plot the closing prices of 10 random selected assets
data = []
for asset in np.random.choice(market_train_df['assetName'].unique(), 10):
    asset_df = market_train_df[(market_train_df['assetName'] == asset)]
    
    data.append(go.Scatter(
        x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset_df['close'].values,
        name = asset
    ))
layout = go.Layout(dict(title = "Closing prices of 10 random assets",
                       xaxis = dict(title = 'Month'),
                       yaxis = dict(title = 'Price (USD)'),
                       ),legend=dict(
                orientation="h"))
py.iplot(dict(data=data, layout=layout), filename='basic-line')
# plot the trends of closing prices by quantiles
data = []
for i in [0.05, 0.1, 0.25, 0.5, 0.9, 0.95]:
    price_df = market_train_df.groupby('time')['close'].quantile(i).reset_index()
    
    data.append(go.Scatter(
        x = price_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = price_df['close'].values,
        name = f'{i} quantile'
    ))
layout = go.Layout(dict(title = "Trends of closing prices by quantiles",
                       xaxis = dict(title = 'Month'),
                       yaxis = dict(title = 'Price (USD)'),
                       ),legend=dict(
                     orientation="h"),
         annotations=[
             dict(
                 x='2008-09-01 22:00:00+0000',
                 y=82,
                 xref='x',
                 yref='y',
                 text='Collapse of Lehman Brothers',
                 showarrow=True,
                 font=dict(
                     family='Courier New, monospace',
                     size=16,
                     color='#ffffff'
                 ),
                 align='center',
                 arrowhead=2,
                 arrowsize=1,
                 arrowwidth=2,
                 arrowcolor='#636363',
                 ax=20,
                 ay=-30,
                 bordercolor='#c7c7c7',
                 borderwidth=2,
                 borderpad=4,
                 bgcolor='#ff7f0e',
                 opacity=0.8
             ),
             dict(
                 x='2011-08-01 22:00:00+0000',
                 y=85,
                 xref='x',
                 yref='y',
                 text='Black Monday',
                 showarrow=True,
                 font=dict(
                     family='Courier New, monospace',
                     size=16,
                     color='#ffffff'
                 ),
                 align='center',
                 arrowhead=2,
                 arrowsize=1,
                 arrowwidth=2,
                 arrowcolor='#636363',
                 ax=20,
                 ay=-30,
                 bordercolor='#c7c7c7',
                 borderwidth=2,
                 borderpad=4,
                 bgcolor='#ff7f0e',
                 opacity=0.8
             ),
             dict(
                 x='2014-10-01 22:00:00+0000',
                 y=120,
                 xref='x',
                 yref='y',
                 text='Another crisis',
                 showarrow=True,
                 font=dict(
                     family='Courier New, monospace',
                     size=16,
                     color='#ffffff'
                 ),
                 align='center',
                 arrowhead=2,
                 arrowsize=1,
                 arrowwidth=2,
                 arrowcolor='#636363',
                 ax=-20,
                 ay=-30,
                 bordercolor='#c7c7c7',
                 borderwidth=2,
                 borderpad=4,
                 bgcolor='#ff7f0e',
                 opacity=0.8
            ),
            dict(
                x='2014-10-01 22:00:00+0000',
                y=120,
                xref='x',
                yref='y',
                text='Another crisis',
                showarrow=True,
                font=dict(
                    family='Courier New, monospace',
                    size=16,
                    color='#ffffff'
                ),
                align='center',
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#636363',
                ax=-20,
                ay=-30,
                bordercolor='#c7c7c7',
                borderwidth=2,
                borderpad=4,
                bgcolor='#ff7f0e',
                opacity=0.8
            ),
            dict(
                x='2016-01-01 22:00:00+0000',
                y=120,
                xref='x',
                yref='y',
                text='Oil prices crash',
                showarrow=True,
                font=dict(
                    family='Courier New, monospace',
                    size=16,
                    color='#ffffff'
                ),
                align='center',
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#636363',
                ax=20,
                ay=-30,
                bordercolor='#c7c7c7',
                borderwidth=2,
                borderpad=4,
                bgcolor='#ff7f0e',
                opacity=0.8
            )
         ])
py.iplot(dict(data=data, layout=layout), filename='basic-line')
# plot the trend of returnsOpenNextMktres10 by quantiles
data = []
for i in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
    price_df = market_train_df.groupby('time')['returnsOpenNextMktres10'].quantile(i).reset_index()

    data.append(go.Scatter(
        x = price_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = price_df['returnsOpenNextMktres10'].values,
        name = f'{i} quantile'
    ))
layout = go.Layout(dict(title = "Trends of returnsOpenNextMktres10 by quantiles",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"),)
py.iplot(dict(data=data, layout=layout), filename='basic-line')
# plot the trend of mean of target variable-returnsOpenNextMktres10 of data since 2010 year
data = []
market_train_df = market_train_df.loc[market_train_df['time'] >= '2010-01-01 22:00:00+0000']
price_df = market_train_df.groupby('time')['returnsOpenNextMktres10'].mean().reset_index()
data.append(go.Scatter(
    x = price_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = price_df['returnsOpenNextMktres10'].values,
    name = f'{i} quantile'
))
layout = go.Layout(dict(title = "Treand of returnsOpenNextMktres10 mean",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"),)
py.iplot(dict(data=data, layout=layout), filename='basic-line')
# plot the trend of mean values of returnsClosePrevRaw1', 'returnsOpenPrevRaw1','returnsClosePrevMktres1', 'returnsOpenPrevMktres1','returnsClosePrevRaw10', 'returnsOpenPrevRaw10','returnsClosePrevMktres10', 'returnsOpenPrevMktres10','returnsOpenNextMktres10'
data = []
for col in ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
       'returnsOpenNextMktres10']:
    df = market_train_df.groupby('time')[col].mean().reset_index()
    data.append(go.Scatter(
        x = df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = df[col].values,
        name = col
    ))
layout = go.Layout(dict(title = "Trend of mean values",
                        xaxis = dict(title = 'Month'),
                        yaxis = dict(title = 'Price (USD)'),
                        ),legend=dict(orientation="h"),)
py.iplot(dict(data=data, layout=layout), filename='basic-line') 
# Sort data in chronological order
market_train_df = market_train_df.sort_values('time')
market_train_df['date'] = market_train_df['time'].dt.date

# Function to plot time series data
def plot_vs_time(data_frame, column, calculation='mean', span=10):
    if calculation == 'mean':
        group_temp = data_frame.groupby('date')[column].mean().reset_index()
    if calculation == 'count':
        group_temp = data_frame.groupby('date')[column].count().reset_index()
    if calculation == 'nunique':
        group_temp = data_frame.groupby('date')[column].nunique().reset_index()
    group_temp = group_temp.ewm(span=span).mean()
    fig = plt.figure(figsize=(10,3))
    plt.plot(group_temp['date'], group_temp[column])
    plt.xlabel('Time')
    plt.ylabel(column)
    plt.title('%s versus time' %column)
# plot the number of asset codes versus time
plot_vs_time(market_train_df, 'assetCode', 'count')
plt.title('Number of asset codes versus time')
# plot the market trade volumes by quantile
for i in [0.05, 0.25, 0.5, 0.75, 0.95]:
    price_df = market_train_df.groupby('date')['volume'].quantile(i).reset_index()
    plt.plot(price_df['date'], price_df['volume'], label='%.2f quantile' %i)
plt.legend(loc='best')
plt.xlabel('Time')
plt.ylabel('Volumes')
plt.title('Market trade volumes by quantile')
# difference between raw values and market adjusted values
column_mkt_raw_diff = []
for i in range(len(column_market)):
    this_raw = column_raw[i]
    this_market = column_market[i]
    new_column_name = 'mkt_raw_diff'+this_raw.replace('returns','').replace('Raw','')
    column_mkt_raw_diff.append(new_column_name)
    market_train_df[new_column_name] = market_train_df[this_market] - market_train_df[this_raw]
    
market_train_df[column_mkt_raw_diff].describe()
# The difference between raw return and market adjusted returns are negligible
# *Asset codes* (no. of unique asset nodes in total & no. of unique asset nodes with unknown asset name)
assetCode_df = market_train_df.groupby('assetCode')['volume'].sum().sort_values(ascending=False)
print('There are %i unique asset code' %len(assetCode_df))

unknown_name = market_train_df[market_train_df['assetName']=='Unknown']
unknown_count = unknown_name['assetCode'].value_counts().sort_values(ascending=False)
print('There are %i unique asset code with unknown asset name' %len(unknown_count))
# plot top 15 asset code by volume
unknown_count[:15].plot.barh()
plt.ylabel('assetCode')
plt.xlabel('Counts')
plt.title('Top 15 asset code with Unknown asset name')
plt.gca().invert_yaxis()
# plot top 15 asset codes with unknown asset name
unknown_count[:15].plot.barh()
plt.ylabel('assetCode')
plt.xlabel('Counts')
plt.title('Top 15 asset codes with Unknown asset name')
plt.gca().invert_yaxis()
#*Asset name*
# plot top 15 asset name by volume
assetName_Volume = market_train_df.groupby('assetName')['volume'].sum().sort_values(ascending=False)
assetName_Volume[:15].plot.barh()
plt.ylabel('assetName')
plt.xlabel('Trading volume')
plt.title('Top 15 asset name by volume')
plt.gca().invert_yaxis()
del assetName_Volume
# Correlation
columns_corr_market = ['volume', 'open', 'close','returnsClosePrevRaw1','returnsOpenPrevRaw1',\
           'returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10','returnsOpenPrevRaw10',\
           'returnsClosePrevMktres10', 'returnsOpenPrevMktres10', 'returnsOpenNextMktres10']
colormap = plt.cm.RdBu
plt.figure(figsize=(18,15))
sns.heatmap(market_train_df[columns_corr_market].astype(float).corr(), linewidths=0.1, vmax=1.0, vmin=-1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.title('Pair-wise correlation')
