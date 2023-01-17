from pathlib import Path



DATA_DIR = Path("/kaggle/input")

if (DATA_DIR / "ucfai-core-sp20-time-series").exists():

    DATA_DIR /= "ucfai-core-sp20-time-series"

else:

    # You'll need to download the data from Kaggle and place it in the `data/`

    #   directory beside this notebook.

    # The data should be here: https://kaggle.com/c/ucfai-core-sp20-time-series/data

    DATA_DIR = Path("data")
# import dependencies

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima_model import ARIMA

from fbprophet import Prophet



import time

import datetime

import os

import sys



# matplotlib configuration

%matplotlib inline

plt.style.use('ggplot')
# load data for Dow Jones Index

df = pd.read_csv(DATA_DIR / 'DJI.csv', delimiter=',')

df.head()
fig, ax = plt.subplots(figsize=(20,10))

ax.plot(df['Open'], color='green', label='Dow Jones Opening Price')

ax.set_ylabel('Price')

ax.set_xlabel('Days')

ax.legend()

plt.show()

# Plot the last 100 days of Dow opening prices with date labels

fig, axs = plt.subplots(2, figsize=(20,10))



# To plot dates is a somewhat expensive operation, so we're only going to look at the past m days

m = 100

dow = df[:m]



# plot the Dow with dates on the first axes

axs[0].plot(dow['Date'], dow['Open'], color='green', label='Dow Jones Opening Price')

axs[0].set_ylabel('Price')

axs[0].set_xlabel('Days')



# plot volume using a bar graph on the second axes

axs[1].bar(dow['Date'], dow['Volume'], color='blue', alpha=0.5, label='Dow Jones Volume')



# show every nth x tick label

i = 5

plt.xticks([x for x in dow['Date']][::i])





# rotate date labels

fig.autofmt_xdate()

axs[0].format_xdata = mdates.DateFormatter('%Y-%m-%d')

axs[1].format_xdata = mdates.DateFormatter('%Y-%m-%d')



# show legend

axs[0].legend()

axs[1].legend()

plt.show()

# calculate 5 day moving average

MA5 = []



# start at the fifth index and loop through all data

for i in range(5, len(df)):

    # sum previous 5 data points and average them

    sum = 0

    for j in range(i-5, i):

        sum += df['Adj Close'][j]

    # add the average to the list

    MA5.append(sum / 5)

    



# drop rows 0 - 5

df = df[:-5]



# append the moving average to the data frame

df['MA5'] = MA5

# Let's take a look at the 5 day moving average we just calculated



fig, ax = plt.subplots(figsize=(20,10))



# zoom in a bit

plt.axis([2300, 2500, 3600, 4000])

ax.plot(df['Open'], color='blue', label='Dow Jones Open')

ax.plot(df['MA5'], color="green", label='5 Day MA')

ax.legend()

plt.show()
# define a list of averages to calculate

averages = [10, 15, 50, 100, 200, 500]



# expand our data to several different moving averages

for avg in averages:

    # easier way to calculate moving averages than using for loop method

    df['MA' + str(avg)] = df['Adj Close'].rolling(window=avg).mean()
# view the new moving averages in our data frame

df.head()
# drop rows with null values, which is up to our largest moving average

df = df.dropna()

df.head()
fig, ax = plt.subplots(figsize=(20,10))



# adjust the view

plt.axis([5000, 8000, 5000, 20000])



ax.plot(df['Adj Close'], c='green', label='Dow Adj Close')



for avg in averages:

    name = 'MA' + str(avg)

    ax.plot(df[name], label=name)



ax.legend()

plt.show()
# EWM: Exponential Weighted Functions



df['EMA5'] = df['Open'].ewm(span=5, adjust=False).mean()



fig, ax = plt.subplots(figsize=(20,10))

plt.axis([6100, 6200, 7500, 10000])



ax.plot(df['Open'], c='green', label='Dow Open')

ax.plot(df['EMA5'], c='blue', label='5 Day EMA')

ax.legend()

plt.show()
# compare EMA to SMA

fig, ax = plt.subplots(figsize=(20,10))

plt.axis([6100, 6200, 7500, 10000])

ax.plot(df['Open'], c='green', label='Dow Open')

ax.plot(df['EMA5'], c='blue', label='5 Day EMA')

ax.plot(df['MA5'], c='orange', label='5 Day SMA')

ax.legend()

plt.show()
df = pd.read_csv(DATA_DIR / 'AirPassengers.csv', delimiter=',')

df.head()
fig,ax = plt.subplots(figsize=(8,6))

ax.plot(df['Month'], df['Passengers'], color='blue', label='Passenger Volume')



# fancy way to show every nth tick label

n = 12

plt.xticks([x for x in df['Month']][::n])

    

fig.autofmt_xdate()

ax.format_xdata = mdates.DateFormatter('%Y')

ax.legend()

plt.show()
# Use the 48 Month moving average as the trend line

# note: we're using 48 here for a couple reasons

 

df['MA48'] = df['Passengers'].rolling(window=48).mean()



# plot the the trend line

fig, ax = plt.subplots(figsize=(20,10))

ax.plot(df['Month'], df['Passengers'], color='blue', label='Passenger Data')

ax.plot(df['MA48'], color='green', label='48 Month MA')



n = 6

plt.xticks([x for x in df['Month']][::n])

fig.autofmt_xdate()

ax.legend()

plt.show()
# subtract the trend line from the passenger data

df['detrend'] = df['Passengers'] - df['MA48']



# plot detrended data

fig, ax = plt.subplots(figsize=(20,10))

ax.plot(df['detrend'], color='orange', label='detrended data')

ax.set_xlabel('Month')

ax.set_ylabel('Seasonal Fluctuation')

ax.legend()

plt.show()
# create a new dataframe to hold our annual detrended data

annual = pd.DataFrame()



# iterate and chop our data into 12 month periods

i = 0

while (i * 12) < (len(df['detrend']) - 12):

    x = i * 12

    annual[i] = df['detrend'].iloc[x:x+12].reset_index(drop=True)

    i += 1



fig, ax = plt.subplots(figsize=(15,5))

fig.suptitle('Annual Seasonal Data')

for i in range(0, 11):

    ax.plot(annual[i], label='year '+str(i))





# plot each month out on a timeline

ax.set_xlabel('Month')

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

plt.xticks(ticks=[x for x in range(12)], labels=months)

ax.set_ylabel('Seasonal Fluctuation')





ax.legend()

plt.show()
# Find the average seasonal data by taking the mean from each year

annual['Seasonal Mean'] = annual.mean(axis=1)

# blue line represents seasonal trend

fig, ax = plt.subplots(figsize=(15,5))

for i in range(0, 11):

    ax.plot(annual[i], label='year '+ str(i), alpha=0.3)





ax.set_title('Detrended Seasonal Data')

ax.plot(annual['Seasonal Mean'], color='blue', label='Seasonal Mean')

ax.legend()

ax.set_ylabel('Seasonal Fluctuation')

ax.set_xlabel('Month')

plt.xticks(ticks=[x for x in range(12)], labels=months)

plt.show()
# implemented with statsmodels

# not as much fun, but a whole lot easier

from statsmodels.tsa.seasonal import seasonal_decompose



# strips time

def parser(x):

    return datetime.datetime.strptime(x, '%Y-%m')



data = pd.read_csv(DATA_DIR / 'AirPassengers.csv', delimiter=',', index_col=['Month'], parse_dates=True, squeeze=True, date_parser=parser)



# try toggling the model between additive and multiplicative!

res = seasonal_decompose(data, model='multiplicative')

pd.plotting.register_matplotlib_converters() # this fixes an error when plotting

res.plot()

plt.show()
from pandas.plotting import autocorrelation_plot



# plot autocorrelation

autocorrelation_plot(data)

plt.show()
# Isolate values

X = df['Passengers'].values



# manual test-train split

size = int(len(X) * 0.66)

train, test = X[0:size], X[size:len(X)]



# create a new list for our training data. We'll expand this list once we "see" testing values

history = [x for x in train]



# list to store just our predictions

predictions = []
# Make predictions at each point in our testing data

for t in range(len(test)):

    # since our training data is dynamically growing, we make and train a new model each iteration

    model = ARIMA(history, order=(5,1,0))

    model_fit = model.fit(disp=0)

    

    # make a prediction

    output = model_fit.forecast()

    

    # forecast method returns a tuple, take the first value

    yhat = output[0]

    

    # add prediction to our list

    predictions.append(yhat)

    

    # add our testing value back into our history

    obs = test[t]

    history.append(obs)

    print('predicted=%f, expected=%f' % (yhat, obs))

    

error = mean_squared_error(test, predictions)

print('Test MSE: %.3f' % error)



fig, ax = plt.subplots(figsize=(10,5))

ax.plot(test, color='blue', label='test data')

ax.plot(predictions, color='red', label='predictions')

ax.legend()

plt.show()
model = Prophet()

print(df)

# create a copy of the data frame and rename columns so they work nicely with fbprophet library

df = df.rename(columns={'Month':'ds', 'Passengers': 'y'})



# fit the model to our data frame

model.fit(df)



# make a new dataframe for the forecast

future = model.make_future_dataframe(periods=12, freq='M')



# make a forecast

forecast = model.predict(future)



# plot forecast

fig = model.plot(forecast)
# Take a look at the data set below, it's totally not in the right format that we want

df = pd.read_csv(DATA_DIR / 'covid_19_data.csv', delimiter=',')

print(df.head())
# TODO: sum up confirmed cases and plot a graph over time



# create a dictionary (hash map) to store the total observations on a given date

keys = [date for date in df['ObservationDate'].unique()]

casesMap = dict.fromkeys(keys, 0)



# iterate through each row and add up the number of confirmed cases to a hash map

# TODO: Sum up the total confirmed cases in the hash map

for index, row in df.iterrows():

    # YOUR CODE HERE

    raise NotImplementedError()

# We have each of the totals in a dict object, now we have to put them back into time series data



# create a new dataframe

df = pd.DataFrame()



# add in a row for each of our dates

df['Date'] = keys

print(df.head())



# add in a new row for confirmed cases at each date

df['Confirmed'] = [casesMap[x] for x in keys]

df = df.dropna()



print('\n Confirmed Cases Dataframe')

print(df.head())
# TODO: Plot the number of confirmed cases over time



# YOUR CODE HERE

raise NotImplementedError()
# TODO: Calculate the 5 Day Moving Average and plot it



# YOUR CODE HERE

raise NotImplementedError()
# TODO: Apply linear regression to the graph

model = Prophet()

    

# create a copy of the data frame and rename columns so they work nicely with fbprophet

df = df.rename(columns={'Date':'ds', 'Confirmed': 'y'})



model.fit(df)

future = model.make_future_dataframe(periods=60)

forecast = model.predict(future)



# changing format of the dates

forecast['ds'] = [x.strftime('%m/%d/%Y') for x in forecast['ds']]
fig, ax = plt.subplots(figsize=(20,10))



# plot lower and upper bound of forecast and compare to real data

dates = [d for d in forecast['ds']]

ax.plot(df['ds'], df['y'], color='green', label='Real Confirmed Cases')

ax.plot(dates, forecast['yhat_lower'], color='blue', label='yhat lower')

ax.plot(dates, forecast['yhat_upper'], color='orange', label='yhat upper')



# fancy way to show every nth tick label

n = 7

plt.xticks([x for x in forecast['ds']][::n])

    

fig.autofmt_xdate()

ax.format_xdata = mdates.DateFormatter('%m-%d-%Y')

ax.legend()

plt.show()
# TODO: Use the ARIMA model to forecast the spread of Coronavirus



# YOUR CODE HERE

raise NotImplementedError()
# TODO: Try to forecast 30 days into the future



# YOUR CODE HERE

raise NotImplementedError()