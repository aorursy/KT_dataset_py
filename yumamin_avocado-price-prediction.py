# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np

import pandas as pd

import datetime as dt



import os

import pylab as pl

from pylab import rcParams

rcParams['figure.figsize'] = 12, 8

import matplotlib.pyplot as plt

import seaborn as sns

data = pd.read_csv('../input/avocado-prices/avocado.csv')



import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)

import warnings

warnings.filterwarnings("ignore", message='numpy.dtype size changed')



%matplotlib inline
df = pd.read_csv('../input/avocado-prices/avocado.csv')
df.head()
df.groupby('type').groups
PREDICTION_TYPE = 'conventional'

df = df[df.type == PREDICTION_TYPE]
df['Date'] = pd.to_datetime(df['Date'])
regions = df.groupby(df.region)

print("Total regions :", len(regions))

print("-----------")

for name, group in regions:

    print(name, " : ", len(group))
PREDICTIONG_FOR = "TotalUS"
date_price = regions.get_group(PREDICTIONG_FOR)[['Date', 'AveragePrice']].reset_index(drop=True)
date_price.plot(x='Date', y='AveragePrice', kind='line')
date_price = date_price.rename(columns={'Date':'ds', 'AveragePrice':'y'})
from fbprophet import Prophet
m = Prophet()

m.fit(date_price)
future = m.make_future_dataframe(periods=1095)

forecast = m.predict(future)
forecast.tail()
fig2 = m.plot_components(forecast)
mask = data['type'] == 'organic'

g = sns.factorplot('AveragePrice', 'region', data=data[mask],

                  hue = 'year', size = 13, aspect = 0.8,

                  palette = 'magma', join = False)
type_list = list(data.type.unique())

average_price2 = []



for i in type_list:

    x=data[data.type==i]

    average_price2.append(sum(x.AveragePrice)/len(x))

df2 = pd.DataFrame({'type_list':type_list, 'average_price':average_price2})



plt.figure(figsize=(15,10))

ax=sns.barplot(x=df2.type_list, y=df2.average_price, palette='vlag')

plt.xlabel('Type of Avocado')

plt.ylabel('Average Price')

plt.title('Average Price of Avocado According to Types')
region_list = list(data.region.unique())

average_price = []



for i in region_list:

    x=data[data.region==i]

    region_average=sum(x.AveragePrice)/len(x)

    average_price.append(region_average)

    

df1=pd.DataFrame({'region_list':region_list, 'average_price':average_price})

new_index=df1.average_price.sort_values(ascending=False).index.values

sorted_data=df1.reindex(new_index)



plt.figure(figsize=(15,10))

ax=sns.barplot(x=sorted_data.region_list, y=sorted_data.average_price, palette='rocket')



plt.xticks(rotation=90)

plt.xlabel('Region')

plt.ylabel('Average Price')

plt.title('Average Price of Avocado According to Region')
small = []

large = []

xlarge = []



for i in region_list:

    x=data[data.region == i]

    small.append(sum(x['Small Bags'])/len(x))

    large.append(sum(x['Large Bags'])/len(x))

    xlarge.append(sum(x['XLarge Bags'])/len(x))

    

df5=pd.DataFrame({'region_list':region_list, 'small':small, 'large':large, 'xlarge':xlarge})



f,ax1 = plt.subplots(figsize=(20,10))

sns.pointplot(x=region_list, y=small, data=df5, color='brown', alpha=0.7)

sns.pointplot(x=region_list, y=large, data=df5, color='green', alpha=0.7)

sns.pointplot(x=region_list, y=xlarge,data=df5, color='yellow', alpha=0.7)



plt.xticks(rotation=90)

plt.text(1, 650000, 'small bags', color='brown', fontsize=14)

plt.text(1, 625000, 'large bags', color='green', fontsize=14)

plt.text(1, 600000, 'x large bags', color='yellow', fontsize=14)



plt.xlabel('Region', color = 'blue', fontsize=14)

plt.ylabel('Values', color='blue', fontsize=14)

plt.title('Small Bags, Large Bags and X Large Bags of Each Region', color='blue', fontsize=14)

plt.grid()
filter1 = data.region!='TotalUS'

data1 = data[filter1]



region_list = list(data1.region.unique())

average_total_volume=[]



for i in region_list:

    x=data1[data1.region==i]

    average_total_volume.append(sum(x['Total Volume'])/len(x))

df3=pd.DataFrame({'region_list':region_list, 'average_total_volume':average_total_volume})



new_index=df3.average_total_volume.sort_values(ascending=False).index.values

sorted_data1 = df3.reindex(new_index)



plt.figure(figsize=(15,10))

ax=sns.barplot(x=sorted_data1.region_list, y=sorted_data1.average_total_volume,palette='deep')



plt.xticks(rotation=90)

plt.xlabel('Region')

plt.ylabel('Average of Total Volume')

plt.title('Average of Total Volume According to Region')
sns.boxplot(y='type', x='AveragePrice', data=data, palette='pink')
data = pd.read_csv('../input/avocado-prices/avocado.csv') #read to data

data = data.drop(['Unnamed: 0'], axis = 1) #drop the useless column

names = ["date", "avprice", "totalvol", "small","large","xlarge","totalbags","smallbags","largebags","xlargebags","type","year","region"] #get new column names

data = data.rename(columns=dict(zip(data.columns, names))) #rename columns

data.head()
plt.figure(figsize=(12,20))

sns.set_style('whitegrid')

sns.pointplot(x='avprice', y='region', data=data, hue='type', join=False)

plt.xticks(np.linspace(1,2,5))

plt.xlabel('Region', {'fontsize':'large'})

plt.ylabel('Avrage Price', {'fontsize':'large'})

plt.title('Type Average Price in Each Region', {'fontsize':'20'})
import datetime
dates = [datetime.datetime.strptime(ts, "%Y-%m-%d") for ts in data['date']]

dates.sort()

sorteddates = [datetime.datetime.strftime(ts, "%Y-%m-%d") for ts in dates]

data['date'] = pd.DataFrame({'date':sorteddates})

data['Year'], data['Month'],  data['Day'] = data['date'].str.split('-').str

data.head()
df = pd.read_csv('../input/avocado-prices/avocado.csv')

df.head()
df.shape
df.describe()
df.dtypes
df['Year'], df['Month'], df['Day'] = df['Date'].str.split('-').str
plt.figure(figsize=(18,10))

sns.lineplot(x="Month", y='AveragePrice', hue='type', data=df)

plt.show()
fig1 = m.plot(forecast)