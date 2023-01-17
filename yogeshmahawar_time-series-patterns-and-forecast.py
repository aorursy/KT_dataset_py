# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sb
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Reading csv using pandas
import pandas as pd 

adf = pd.read_csv('../input/avocado.csv')
print(adf.head())
adf.info()
adf.describe()
#First column uncertain: No column name given, repeatin after 0-51 counts! strange!
adf.drop(['Unnamed: 0'],axis=1,inplace=True)
adf.head()
#null check 
#adf.isnull() --> only gives binary mask 
adf.isnull().sum()

adf_US = adf[adf['region']=='TotalUS']
adf_US_organic = adf_US[adf_US['type']=='organic']
adf_US_organic.head()
plt.figure(figsize=(12,5))
plt.title("Distribution of Avg Price")
#average_price_fit = stats.norm.pdf(adf['AveragePrice'],np.mean(adf['AveragePrice']),np.std(adf['AveragePrice']))
plt.xlabel('Average Price')
plt.ylabel('Probability')
#plt.hist(adf['AveragePrice'],bins=40,color='g')
#plt.plot(adf['AveragePrice'],average_price_fit)
sb.distplot(adf_US_organic["AveragePrice"],hist=True,kde=True,rug=True,bins=100, color = 'b')

#creating month column
adf['Date'] = pd.to_datetime(adf['Date'], format='%Y-%m-%d')
adf['Month']=adf['Date'].map(lambda x: x.month)
adf = adf.sort_values(by='Date')
plt.figure(figsize=(12,3))
sb.lineplot(x="Date", y="AveragePrice",hue='year',data=adf_US_organic,palette='magma')
plt.figure(figsize=(12,3))
sb.lineplot(x="Date", y="Total Volume",hue='year',data=adf_US_organic,palette='magma',)
plt.figure(figsize=(12,3))
sb.lineplot(x="Date", y="Total Bags",hue='year',data=adf_US_organic,palette='magma')
#sb.lineplot(x="Month", y="Total Volume",hue='year',data=adf_US_organic,palette='copper')
plt.figure(figsize=(12,3))
sb.lineplot(x="Date", y="4046",hue='year',data=adf_US_organic,palette='magma',)
plt.figure(figsize=(12,3))
sb.lineplot(x="Date", y="4225",hue='year',data=adf_US_organic,palette='magma')
plt.figure(figsize=(12,3))
sb.lineplot(x="Date", y="4770",hue='year',data=adf_US_organic,palette='magma')
adf_US_organic = adf_US_organic.sort_values(by='Date')
# Valid = adf[(adf['year'] == 2017) | (adf['year'] == 2018)]
# Train = adf[(adf['year'] != 2017) & (adf['year'] != 2018)]
Train = adf_US_organic.sort_values(by='Date')

from fbprophet import Prophet
m = Prophet()
date_volume = Train.rename(columns={'Date':'ds', 'Total Volume':'y'})
m.fit(date_volume)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
n = Prophet()
date_bags = Train.rename(columns={'Date':'ds', 'Total Bags':'y'})
n.fit(date_bags)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
fig2 = m.plot(forecast)
fig2 = m.plot_components(forecast)
