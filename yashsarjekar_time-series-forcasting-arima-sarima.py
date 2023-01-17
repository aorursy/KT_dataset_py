import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/perrin-freres-monthly-champagne-sales/perrin-freres-monthly-champagne.csv')
df.head(5)
df.columns = ['Months','Sales']
df.head(5)
df.isnull()
df.drop(105,axis=0,inplace=True)
df.drop(106,axis=0,inplace=True)
df.isnull().sum()
df.dtypes
df['Months'] = pd.to_datetime(df['Months'])
df.dtypes
df.set_index('Months',inplace=True)
df.head(5)
df.plot(figsize= (12,8))
from statsmodels.tsa.stattools import adfuller
# Accept Null Hpyo means dataset is Not Stationary
# Reject Null Hypo Means dataset is Stationary
def adfuller_test(sales):
    result = adfuller(sales)
    labels = ['ADF Test statistic','p-value','#lags used','Number of Observations']
    for values, label in zip(result,labels):
        print(label +':'+ str(values))
    if(result[1] <= 0.05):
        print("The Dataset is stationary, Reject Null Hypothesis")
    else:
        print("The Dataset is Not Stationary, Accept Null Hypothesis")
adfuller_test(df['Sales'])
df['Seasonal Sales Diff'] = df['Sales'] - df['Sales'].shift(12)
df
adfuller_test(df['Seasonal Sales Diff'].dropna())
df['Seasonal Sales Diff'].plot(figsize=(12,8))
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
fig =plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = plot_acf(df['Seasonal Sales Diff'].iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(df['Seasonal Sales Diff'].iloc[13:],lags= 40,ax=ax2)
# For Non Seasonal Data
#p=1 d=1 q can be 0 or 1
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(df['Sales'],order=(1,1,1))
model_fit = model.fit()
model_fit.summary()
df.tail(20)
df['forecast'] = model_fit.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))
import statsmodels.api as sm
model = sm.tsa.statespace.SARIMAX(df['Sales'],order=(1,1,1),seasonal_order=(1,1,1,12))
results = model.fit()
df['forecast'] = results.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))
from pandas.tseries.offsets import DateOffset
future_dates = [df.index[-1]+ DateOffset(months=x) for x in range(0,24)]
future_dataset_df = pd.DataFrame(index=future_dates[1:],columns=df.columns)
future_dataset_df.tail(20)
future_dataset_df.shape
future_df = pd.concat([df,future_dataset_df])
future_df['forecast'] = results.predict(start=104,end=120,dynamic=True)
future_df[['Sales','forecast']].plot(figsize=(12,8))