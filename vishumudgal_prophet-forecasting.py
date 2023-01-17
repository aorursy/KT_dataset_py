# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

df= pd.read_csv('/kaggle/input/avocado-prices/avocado.csv')
df
# the data column should be convertedin readable form

df['Date']=pd.to_datetime(df['Date'])
df['region'].unique()
#Let's start with Scatterplot



plt.scatter(df.year,df.AveragePrice)
plt.plot(df.year,df.AveragePrice,'g--',markersize=5)
# calculating average price for different

types=df.type.unique()

avg=[]



for i in types:

    x = df[df.type == i]

    avg.append(sum(x['AveragePrice'])/len(x))

sns.barplot(types,avg,ci=110,palette = 'afmhot_r')

plt.xlabel('Types')

plt.ylabel("Average Price")

plt.title('Average price of  different types Avacado ')
reg = df.region.unique()

avgr=[]

for i in reg:

    x = df[df.region == i]

    avgr.append(sum(x['AveragePrice'])/len(x))

    



plt.figure(figsize=(22,9))

a=plt.xticks(rotation=70)

plt.xlabel('Regions')

plt.ylabel('Average_Price')

plt.title('Average price of each region')    

sns.barplot(reg,avgr,ci=150,palette ='inferno')

sns.distplot(df.year)
#price trend per year



yr = df.year.unique()

avgy=[]

for i in yr:

    x = df[df.year == i]

    avgy.append(sum(x['AveragePrice'])/len(x))

    



plt.figure(figsize=(12,9))

a=plt.xticks(rotation=70)

plt.xlabel('Year')

plt.ylabel('Average_Price')

plt.title('Price trend every trend')    

sns.lineplot(yr,avgy)

# price trend monthly for different  for alabany 

df['Date']=pd.to_datetime(df['Date'])
df['Month']=pd.DatetimeIndex(df['Date']).month
df['Month-Year'] = df['Month'].map(str)+'-'+df['year'].map(str)
z=df[['Month-Year','AveragePrice','Total Volume']]
z=z.groupby(by='Month-Year').mean()
z=z.reset_index()

plt.figure(figsize=(22,10))

plt.ylabel("AveragePrice")

_=plt.xticks(rotation=80)

plt.plot(z['Month-Year'],z['AveragePrice'])

plt.figure(figsize=(10,9))

plt.scatter(z['AveragePrice'],z['Total Volume'])

plt.xlabel('AveragePrice')

plt.ylabel('Average Volume')

plt.title('Average Proce per volume')

plt.figure(figsize=(12,8))

p=z.pivot('Month-Year','AveragePrice','Total Volume')

sns.heatmap(p,cmap='inferno')
org = df['type']=='organic'

g = sns.factorplot('AveragePrice','region',data=df[org],hue='year',size=13,palette='mako',join=False)
bag=df.groupby(by='region').sum()

bag.reset_index(inplace=True)
plt.figure(figsize=(15,9))

sns.set_style='dark grid'

plt.plot(bag['region'],bag['Small Bags'],color='y',alpha=.9)

plt.plot(bag['region'],bag['Large Bags'],color='r',alpha=.8)

plt.plot(bag['region'],bag['XLarge Bags'],color='g',alpha=.7)

_=plt.xticks(rotation=80)

plt.show()
data= pd.read_csv('/kaggle/input/avocado-prices/avocado.csv')

data['Date'] = pd.to_datetime(data['Date'])

from fbprophet import Prophet
df = data[['Date','AveragePrice']]



df = df.rename(columns={'Date':'ds', 'AveragePrice':'y'})
m = Prophet()

m.fit(df)
# our future forecast of price



future = m.make_future_dataframe(periods=365)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)
fig2=m.plot_components(forecast)
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(m, forecast)  # This returns a plotly Figure

py.iplot(fig)
#after removing outliers

future = m.make_future_dataframe(periods=1096)

forecast = m.predict(future)

fig = m.plot(forecast)
# finding the inflation in price



df['cap'] = 8.5

m = Prophet(growth='logistic')

m.fit(df)

future = m.make_future_dataframe(periods=1826)

future['cap'] = 8.5

fcst = m.predict(future)

fig = m.plot(fcst)
