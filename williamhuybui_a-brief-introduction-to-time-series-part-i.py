# min_temp data

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from IPython.display import Image

import os

%matplotlib inline



temp_data = pd.read_csv("../input/min-temp/min_temp.csv")

temp_data.head().append(temp_data.tail())
#ts data

years = pd.date_range('2012-01', periods=72, freq="M")

index = pd.DatetimeIndex(years)



np.random.seed(3456)

sales= np.random.randint(-4, high=4, size=72)

bigger = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,

                   3,3,3,3,3,3,3,3,7,7,7,7,7,7,7,7,7,7,7,

                   11,11,11,11,11,11,11,11,11,11,18,18,18,

                   18,18,18,18,18,18,26,26,26,26,26,36,36,36,36,36])

data = pd.Series(sales+bigger+6, index=index)

ts=data

ts
#Replace the index by Date column

temp_data.Date = pd.to_datetime(temp_data.Date, format='%d/%m/%y') #convert to pandas timestamp type

temp_data.set_index('Date', inplace=True) 



temp_data.head(2)
#Group by month and find the average of each month



temp_monthly = temp_data.resample('MS') #Since it reduces the data, this called down sampling

month_mean = temp_monthly.mean()

month_mean.head(5)
#Adding addition row for each day and fill it with previous value

temp_bidaily= temp_data.resample('12H').asfreq()#Since it increase the data, this called up sampling



print(temp_bidaily.isnull().sum()) #There are some null here



#Fill data behind it with the following one

temp_bidaily_fill= temp_data.resample('12H').ffill() #forward filling, backward filling (bfill())

temp_bidaily_fill.head()
#Selecting and slicing time series data

#Retrieve data after 1985

temp_1985_onwards = temp_data['1985':]

temp_1985_onwards.head(2).append(temp_1985_onwards.tail(2))
temp_data.plot()

plt.show()
#Dot plots can prove to be very helpful in identifying outliers and very small patterns 

#which may not be so obvious otherwise



temp_data.plot(style=".b")

plt.show()
Image("../input/pictures/stationary.png")
rolmean = ts.rolling(window = 10, center = False).mean()

rolstd = ts.rolling(window = 10, center = False).std()

#Note that it lost a little bit in the beginning since the window use the previous info to check the future



fig = plt.figure(figsize=(12,7))

orig = plt.plot(ts, color='blue',label='Original')

mean = plt.plot(rolmean, color='red', label='Rolling Mean')

std = plt.plot(rolstd, color='black', label = 'Rolling Std')

plt.legend(loc='best')

plt.title('Rolling Mean & Standard Deviation')

plt.show(block=False)
# Use Pandas ewma() to calculate Weighted Moving Average of ts_log

exp_rolmean = data.ewm(halflife = 3).mean() #Here, 3 is 3 month period. Halflife means the decativity rate

exp_rolstd = data.ewm(halflife = 3).std()

# Plot the original data with exp weighted average

fig = plt.figure(figsize=(12,7))

plt.plot(data, color='blue',label='Original')

plt.plot(exp_rolmean, color='red', label='Exponentially Weighted Rolling Mean')

plt.plot(exp_rolstd, color='black', label='Exponentially Weighted Rolling STD')

plt.legend()

plt.title('Exponentially Weighted Rolling Mean & Standard Deviation')

plt.show()
from statsmodels.tsa.stattools import adfuller



dftest = adfuller(ts)



dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])



print ('Results of Dickey-Fuller Test:')

print(dfoutput)
fig, axs = plt.subplots(4,sharex=True,figsize=(11,7),gridspec_kw={'hspace': 0})



#Original data

axs[0].plot(ts, color='blue')



#Log transform

axs[1].plot(np.log(ts),color='red')



#Subtracting the rolling mean

rolmean = ts.rolling(window = 4).mean()

data_minus_rolmean1 = ts - rolmean #How we define "Subtracting the rolling mean"

axs[2].plot(data_minus_rolmean1,color='green')



#Subtracting the weighted rolling mean

exp_rolmean = data.ewm(halflife = 2).mean()

data_minus_rolmean2 = ts - exp_rolmean

axs[3].plot(data_minus_rolmean2,color='purple')
data_diff = data.diff(periods=1)

data_diff.head(10)



fig = plt.figure(figsize=(11,3))

plt.plot(data_diff, color='blue',label='Sales - rolling mean')

plt.legend(loc='best')

plt.title('Differenced sales series')

plt.show()
fig, axs = plt.subplots(2,sharex=True,figsize=(11,7),gridspec_kw={'hspace': 0})



#Original data temp data

axs[0].plot(temp_data, color='blue', linewidth=1)



#1-period lag

data_diff = temp_data.diff(periods=365)

axs[1].plot(data_diff, color='red', linewidth=1)
from statsmodels.tsa.seasonal import seasonal_decompose



decomposition = seasonal_decompose(ts) #model="additive" by default



# Gather the trend, seasonality and noise of decomposed object

trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid



# Plot gathered statistics

fig, axs = plt.subplots(4,sharex=True,figsize=(11,7),gridspec_kw={'hspace': 0})



axs[0].plot(ts, label='Original', color="blue") #Original data

axs[1].plot(trend, label='Trend', color="red") #Trend

axs[2].plot(seasonal,label='Seasonality', color="green") #Season

axs[3].plot(residual, label='Residuals', color="brown") #Residual



plt.show()