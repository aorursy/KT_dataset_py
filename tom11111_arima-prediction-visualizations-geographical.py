import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

import matplotlib.ticker as ticker

from statsmodels.graphics.api import qqplot

import statsmodels

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.stattools import adfuller

import folium

from folium.plugins import HeatMap

import statsmodels as sm

import matplotlib.ticker as ticker

import statsmodels.api as sm

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_model import ARMA

from statsmodels.tsa.statespace.sarimax import SARIMAX
def csv(data):



    read_csv = pd.read_csv(data, encoding="latin-1")



    return ( read_csv )



def info (b):

    infor = b.info()



    null = b.isnull().sum().sort_values(ascending = False)



    print (infor)



    print (null)



def dropna (c):

    new = c.dropna(how = "any")

    return new



def wma (data , N):

    weights = np.arange(1,N+1)

    wma = data.rolling(N).apply(lambda element: np.dot(element, weights)/weights.sum(), raw=True)

    return wma
df = pd.read_csv('../input/crimes-in-boston/crime.csv', encoding='latin-1')



df.head()
df.info
df = df.rename(columns = {"INCIDENT_NUMBER":"NUMBER","OFFENSE_CODE":"CODE","OFFENSE_CODE_GROUP":"GROUP"})

df.columns
plt.figure(figsize=(8,8))

sns.heatmap(df.isnull())

plt.xlabel('Columns')

plt.ylabel('Rows')

plt.show()
df = df.drop(["SHOOTING","NUMBER","UCR_PART"],axis = 1)



df = dropna(df)
plt.figure(figsize=(8,8))

sns.heatmap(df.isnull())

plt.xlabel('Columns')

plt.ylabel('Rows')

plt.show()
df.index = pd.DatetimeIndex(df.OCCURRED_ON_DATE)





df[["DATE","TIME"]]=df['OCCURRED_ON_DATE'].str.split(" ",expand=True) 





dfp = df.resample('D').size().reset_index()



dfp.columns = ['date','count']







d = dfp['date']

c = dfp['count']

dfp.head()



c_wma = wma(c,30)



plt.figure(figsize = (15,12))





plt.plot(d,c,lw=1)





plt.plot(d,c_wma,lw=3)





plt.title("Number of Crime  & 30-day weighted moving average")



plt.xlabel('Date')



plt.ylabel('Number of Crime ')



plt.legend(['Number of Crime','30-days WMA'])





plt.show()




res = sm.tsa.seasonal_decompose(c,freq=365,model="additive")



trend = res.trend

seasonal = res.seasonal

residual = res.resid



fig,ax=plt.subplots(figsize = (17,12))

ax1 = fig.add_subplot(411)



ax1.plot(d,c, label='Original')

ax1.legend(loc='best')



ax2 = fig.add_subplot(412)



ax2.plot(d,trend, label='Trend')

ax2.legend(loc='best')

ax3 = fig.add_subplot(413)



ax3.plot(d,seasonal,label='Seasonality')

ax3.legend(loc='best')



ax4 = fig.add_subplot(414)



ax4.plot(d,residual, label='Residuals')

ax4.legend(loc='best')

plt.tight_layout()

train1 = dfp.copy()

train1['count']=train1['count'].diff(1)





plt.figure(figsize = (15,8))









plt.plot( train1['date'],train1['count'],label='d=1')

plt.legend(loc='best')









plt.show()


train1 = train1.dropna()





fig = plt.figure(figsize=(12,8))

 

ax1 = fig.add_subplot(211)

fig = sm.graphics.tsa.plot_acf(train1['count'], lags=15,ax=ax1)



fig.tight_layout()

 

ax2 = fig.add_subplot(212)

fig = sm.graphics.tsa.plot_pacf(train1['count'], lags=15, ax=ax2)



fig.tight_layout()

plt.show()


arma_mod = ARMA(dfp['count'],(4,1,2)).fit()





summary = (arma_mod.summary2(alpha=.05, float_format="%.8f"))

print(summary)




resid = arma_mod.resid

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(111)

fig = qqplot(resid, line='q', ax=ax, fit=True)
arma_mod.forecast()[0]
df = df.loc[df['YEAR'].isin([2017,2018])]

df.head()
plt.figure(figsize = (7,5))

sns.countplot(x = df.YEAR)

plt.show()
plt.figure(figsize = (8,5))

sns.countplot(x = df.DAY_OF_WEEK)

plt.show()
plt.figure(figsize = (8,8))

label = 'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'

size_month = []



for i in range(12):

    i+=1

    size_month.append(len(df[df["MONTH"]==i]))



plt.pie(size_month, autopct = '%1.1f%%',labels = label,shadow = True)
plt.figure(figsize = (10,5))

sns.countplot(x = df.MONTH)

plt.show()
plt.figure(figsize = (12,5))

sns.countplot(x = df.HOUR)

plt.show()


# Create basic Folium crime map

crime_map = folium.Map(location=[42.36,-71.07], 

                      zoom_start = 11.5,min_zoom = 11.5)



# Add data for heatmp 

data_heatmap = df[df.YEAR == 2017]

data_heatmap = df[['Lat','Long']]

data_heatmap = [[row['Lat'],row['Long']] for index, row in data_heatmap.iterrows()]

HeatMap(data_heatmap, radius=10).add_to(crime_map)



# Plot!

crime_map
# Street Robber

crime_map = folium.Map(location=[42.32,-71.07], 

                      zoom_start = 11.5,min_zoom = 11.5)



# Add data for heatmp 

data_heatmap = df[df.YEAR == 2017]

data_heatmap = df[['Lat','Long']]

data_heatmap = df[df['CODE'] == 301]



data_heatmap = [[row['Lat'],row['Long']] for index, row in data_heatmap.iterrows()]

HeatMap(data_heatmap, radius=10).add_to(crime_map)



# Plot!

crime_map