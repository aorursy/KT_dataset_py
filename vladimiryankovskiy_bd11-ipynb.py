# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import statsmodels.api as sm

import datetime

from fbprophet import Prophet

from xgboost import XGBRegressor



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
avocado = pd.read_csv("../input/avocado-prices/avocado.csv")

avocado.head()
conventional = avocado[avocado.type=='conventional'].drop(columns=['type', 'year', 'Unnamed: 0'])

conventional.head()
organic = avocado[avocado.type=='organic'].drop(columns=['type', 'year', 'Unnamed: 0'])

organic.head()
date = list(avocado.Date.unique())

date.sort()
total_conventional = pd.DataFrame({'Date': [datetime.date(int(d[0:4]), int(d[5:7]), int(d[8:10])) for d in date],

                                  'AveragePrice': [(conventional[conventional.Date == d].AveragePrice*conventional[conventional.Date == d]['Total Volume']).sum()/conventional[conventional.Date == d]['Total Volume'].sum() for d in date],

                                  'Total Volume': [conventional[conventional.Date == d]['Total Volume'].sum()/1000 for d in date],

                                  '4046': [conventional[conventional.Date == d]['4046'].sum()/1000 for d in date],

                                  '4225': [conventional[conventional.Date == d]['4225'].sum()/1000 for d in date],

                                  '4770': [conventional[conventional.Date == d]['4770'].sum()/1000 for d in date]}).set_index('Date')

total_conventional.head()
total_organic = pd.DataFrame({'Date': [datetime.date(int(d[0:4]), int(d[5:7]), int(d[8:10])) for d in date],

                                  'AveragePrice': [(organic[organic.Date == d].AveragePrice*organic[organic.Date == d]['Total Volume']).sum()/organic[organic.Date == d]['Total Volume'].sum() for d in date],

                                  'Total Volume': [organic[organic.Date == d]['Total Volume'].sum()/1000 for d in date],

                                  '4046': [organic[organic.Date == d]['4046'].sum()/1000 for d in date],

                                  '4225': [organic[organic.Date == d]['4225'].sum()/1000 for d in date],

                                  '4770': [organic[organic.Date == d]['4770'].sum()/1000 for d in date]}).set_index('Date')

total_organic.head()
from statsmodels.tsa.stattools import adfuller



def test_stationarity(timeseries, window = 48, cutoff = 0.05):



    rolmean = timeseries.rolling(window).mean()

    rolstd = timeseries.rolling(window).std()



    fig = plt.figure(figsize=(12, 4))

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show()

    fig = plt.figure(figsize=(12, 4))

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    

    plt.show()

    

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries.values,autolag='AIC' )

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    pvalue = dftest[1]

    if pvalue < cutoff:

        print('p-value = %.4f. The series is likely stationary.' % pvalue)

    else:

        print('p-value = %.4f. The series is likely non-stationary.' % pvalue)

    

    print(dfoutput)
fig = plt.figure(figsize=(12, 4))

plt.plot(total_conventional.index, total_conventional.AveragePrice, 'k-')

plt.show()
sm.tsa.seasonal_decompose(total_conventional['AveragePrice'], model='additive',period= 48).plot()

plt.show()
test_stationarity(total_conventional['AveragePrice'], window=48)
d_c_ap = (total_conventional['AveragePrice']-total_conventional['AveragePrice'].shift(1)).dropna()

sm.tsa.seasonal_decompose(d_c_ap, model='additive',period= 48).plot()

plt.show()

test_stationarity(d_c_ap, window=48)
df_ = pd.DataFrame({'ds': total_conventional.index, 'y': total_conventional['AveragePrice']}).dropna()

prophet_basic = Prophet()

prophet_basic.fit(df_)

future= prophet_basic.make_future_dataframe(periods= 48, freq = 'W', include_history = False)

future['label'] = 'test'

future['y'] = np.nan

df_['label'] = 'train'

df = pd.concat((df_,future), axis = 0)

features = []

for period in range(48,96,4):

    df["lag_period_{}".format(period)] = df.y.shift(period)

    features.append("lag_period_{}".format(period))

df['lagf_mean'] = df[features].mean(axis = 1)

features.extend(['lagf_mean'])

model = XGBRegressor()

train_df = df[df.label == 'train'][features + ['y']].dropna()

test_df = df[df.label == 'test'][features]

model.fit(train_df.drop('y', axis= 1) ,train_df['y'])

fig = plt.figure(figsize=(12, 4))

forecast = model.predict(test_df)

plt.plot(future['ds'], forecast, 'k-')

plt.show()
fig = plt.figure(figsize=(12, 4))

plt.plot(total_conventional.index, total_conventional['Total Volume'], 'k-')

plt.show()
sm.tsa.seasonal_decompose(total_conventional['Total Volume'], model='additive',period= 48).plot()

plt.show()
test_stationarity(total_conventional['Total Volume'], window=48)
df_ = pd.DataFrame({'ds': total_conventional.index, 'y': total_conventional['Total Volume']}).dropna()

prophet_basic = Prophet()

prophet_basic.fit(df_)

future= prophet_basic.make_future_dataframe(periods= 48, freq = 'W', include_history = False)

future['label'] = 'test'

future['y'] = np.nan

df_['label'] = 'train'

df = pd.concat((df_,future), axis = 0)

features = []

for period in range(48,96,4):

    df["lag_period_{}".format(period)] = df.y.shift(period)

    features.append("lag_period_{}".format(period))

df['lagf_mean'] = df[features].mean(axis = 1)

features.extend(['lagf_mean'])

model = XGBRegressor()

train_df = df[df.label == 'train'][features + ['y']].dropna()

test_df = df[df.label == 'test'][features]

model.fit(train_df.drop('y', axis = 1) ,train_df['y'])

forecast = model.predict(test_df)

fig = plt.figure(figsize=(12, 4))

plt.plot(future['ds'], forecast, 'k-')

plt.show()
fig = plt.figure(figsize=(12, 4))

plt.plot(total_conventional.index, total_conventional['4046'], 'k-')

plt.show()
sm.tsa.seasonal_decompose(total_conventional['4046'], model='additive',period= 48).plot()

plt.show()
test_stationarity(total_conventional['4046'], window=48)
df_ = pd.DataFrame({'ds': total_conventional.index, 'y': total_conventional['4046']}).dropna()

prophet_basic = Prophet()

prophet_basic.fit(df_)

future= prophet_basic.make_future_dataframe(periods= 48, freq = 'W', include_history = False)

future['label'] = 'test'

future['y'] = np.nan

df_['label'] = 'train'

df = pd.concat((df_,future), axis = 0)

features = []

for period in range(48,96,4):

    df["lag_period_{}".format(period)] = df.y.shift(period)

    features.append("lag_period_{}".format(period))

df['lagf_mean'] = df[features].mean(axis = 1)

features.extend(['lagf_mean'])

model = XGBRegressor()

train_df = df[df.label == 'train'][features + ['y']].dropna()

test_df = df[df.label == 'test'][features]

model.fit(train_df.drop('y', axis = 1) ,train_df['y'])

forecast = model.predict(test_df)

fig = plt.figure(figsize=(12, 4))

plt.plot(future['ds'], forecast, 'k-')

plt.show()
fig = plt.figure(figsize=(12, 4))

plt.plot(total_conventional.index, total_conventional['4225'], 'k-')

plt.show()
sm.tsa.seasonal_decompose(total_conventional['4225'], model='additive',period= 48).plot()

plt.show()
test_stationarity(total_conventional['4225'], window=48)
d_c_4225 = (total_conventional['4225']-total_conventional['4225'].shift(1)).dropna()

sm.tsa.seasonal_decompose(d_c_4225, model='additive',period= 48).plot()

plt.show()

test_stationarity(d_c_4225, window=48)
df_ = pd.DataFrame({'ds': total_conventional.index, 'y': total_conventional['4225']}).dropna()

prophet_basic = Prophet()

prophet_basic.fit(df_)

future= prophet_basic.make_future_dataframe(periods= 48, freq = 'W', include_history = False)

future['label'] = 'test'

future['y'] = np.nan

df_['label'] = 'train'

df = pd.concat((df_,future), axis = 0)

features = []

for period in range(48,96,4):

    df["lag_period_{}".format(period)] = df.y.shift(period)

    features.append("lag_period_{}".format(period))

df['lagf_mean'] = df[features].mean(axis = 1)

features.extend(['lagf_mean'])

model = XGBRegressor()

train_df = df[df.label == 'train'][features + ['y']].dropna()

test_df = df[df.label == 'test'][features]

model.fit(train_df.drop('y', axis = 1) ,train_df['y'])

forecast = model.predict(test_df)

fig = plt.figure(figsize=(12, 4))

plt.plot(future['ds'], forecast, 'k-')

plt.show()
fig = plt.figure(figsize=(12, 4))

plt.plot(total_conventional.index, total_conventional['4770'], 'k-')

plt.show()
sm.tsa.seasonal_decompose(total_conventional['4770'], model='additive',period= 48).plot()

plt.show()
test_stationarity(total_conventional['4770'], window=48)
d_c_4770 = (total_conventional['4770']-total_conventional['4770'].shift(1)).dropna()

sm.tsa.seasonal_decompose(d_c_4770, model='additive',period= 48).plot()

plt.show()

test_stationarity(d_c_4770, window=48)
df_ = pd.DataFrame({'ds': total_conventional.index, 'y': total_conventional['4770']}).dropna()

prophet_basic = Prophet()

prophet_basic.fit(df_)

future= prophet_basic.make_future_dataframe(periods= 48, freq = 'W', include_history = False)

future['label'] = 'test'

future['y'] = np.nan

df_['label'] = 'train'

df = pd.concat((df_,future), axis = 0)

features = []

for period in range(48,96,4):

    df["lag_period_{}".format(period)] = df.y.shift(period)

    features.append("lag_period_{}".format(period))

df['lagf_mean'] = df[features].mean(axis = 1)

features.extend(['lagf_mean'])

model = XGBRegressor()

train_df = df[df.label == 'train'][features + ['y']].dropna()

test_df = df[df.label == 'test'][features]

model.fit(train_df.drop('y', axis = 1) ,train_df['y'])

forecast = model.predict(test_df)

fig = plt.figure(figsize=(12, 4))

plt.plot(future['ds'], forecast, 'k-')

plt.show()
fig = plt.figure(figsize=(12, 4))

plt.plot(total_organic.index, total_organic.AveragePrice, 'k-')

plt.show()
sm.tsa.seasonal_decompose(total_organic['AveragePrice'], model='additive',period= 48).plot()

plt.show()
test_stationarity(total_organic['AveragePrice'], window=48)
df_ = pd.DataFrame({'ds': total_organic.index, 'y': total_organic['AveragePrice']}).dropna()

prophet_basic = Prophet()

prophet_basic.fit(df_)

future= prophet_basic.make_future_dataframe(periods= 48, freq = 'W', include_history = False)

future['label'] = 'test'

future['y'] = np.nan

df_['label'] = 'train'

df = pd.concat((df_,future), axis = 0)

features = []

for period in range(48,96,4):

    df["lag_period_{}".format(period)] = df.y.shift(period)

    features.append("lag_period_{}".format(period))

df['lagf_mean'] = df[features].mean(axis = 1)

features.extend(['lagf_mean'])

model = XGBRegressor()

train_df = df[df.label == 'train'][features + ['y']].dropna()

test_df = df[df.label == 'test'][features]

model.fit(train_df.drop('y', axis = 1) ,train_df['y'])

forecast = model.predict(test_df)

fig = plt.figure(figsize=(12, 4))

plt.plot(future['ds'], forecast, 'k-')

plt.show()
fig = plt.figure(figsize=(12, 4))

plt.plot(total_organic.index, total_organic['Total Volume'], 'k-')

plt.show()
sm.tsa.seasonal_decompose(total_organic['Total Volume'], model='additive',period= 48).plot()

plt.show()
test_stationarity(total_organic['Total Volume'], window=48)
d_o_tv = (total_organic['Total Volume']-total_organic['Total Volume'].shift(1)).dropna()

sm.tsa.seasonal_decompose(d_o_tv, model='additive',period= 48).plot()

plt.show()

test_stationarity(d_o_tv, window=48)
df_ = pd.DataFrame({'ds': total_organic.index, 'y': total_organic['Total Volume']}).dropna()

prophet_basic = Prophet()

prophet_basic.fit(df_)

future= prophet_basic.make_future_dataframe(periods= 48, freq = 'W', include_history = False)

future['label'] = 'test'

future['y'] = np.nan

df_['label'] = 'train'

df = pd.concat((df_,future), axis = 0)

features = []

for period in range(48,96,4):

    df["lag_period_{}".format(period)] = df.y.shift(period)

    features.append("lag_period_{}".format(period))

df['lagf_mean'] = df[features].mean(axis = 1)

features.extend(['lagf_mean'])

model = XGBRegressor()

train_df = df[df.label == 'train'][features + ['y']].dropna()

test_df = df[df.label == 'test'][features]

model.fit(train_df.drop('y', axis = 1) ,train_df['y'])

forecast = model.predict(test_df)

fig = plt.figure(figsize=(12, 4))

plt.plot(future['ds'], forecast, 'k-')

plt.show()
fig = plt.figure(figsize=(12, 4))

plt.plot(total_organic.index, total_organic['4046'], 'k-')

plt.show()
sm.tsa.seasonal_decompose(total_organic['4046'], model='additive',period= 48).plot()

plt.show()
test_stationarity(total_organic['4046'], window=48)
d_o_4046 = (total_organic['4046']-total_organic['4046'].shift(1)).dropna()

sm.tsa.seasonal_decompose(d_o_4046, model='additive',period= 48).plot()

plt.show()

test_stationarity(d_o_tv, window=48)
df_ = pd.DataFrame({'ds': total_organic.index, 'y': total_organic['4046']}).dropna()

prophet_basic = Prophet()

prophet_basic.fit(df_)

future= prophet_basic.make_future_dataframe(periods= 48, freq = 'W', include_history = False)

future['label'] = 'test'

future['y'] = np.nan

df_['label'] = 'train'

df = pd.concat((df_,future), axis = 0)

features = []

for period in range(48,96,4):

    df["lag_period_{}".format(period)] = df.y.shift(period)

    features.append("lag_period_{}".format(period))

df['lagf_mean'] = df[features].mean(axis = 1)

features.extend(['lagf_mean'])

model = XGBRegressor()

train_df = df[df.label == 'train'][features + ['y']].dropna()

test_df = df[df.label == 'test'][features]

model.fit(train_df.drop('y', axis = 1) ,train_df['y'])

forecast = model.predict(test_df)

fig = plt.figure(figsize=(12, 4))

plt.plot(future['ds'], forecast, 'k-')

plt.show()
fig = plt.figure(figsize=(12, 4))

plt.plot(total_organic.index, total_organic['4225'], 'k-')

plt.show()
sm.tsa.seasonal_decompose(total_organic['4225'], model='additive',period= 48).plot()

plt.show()
test_stationarity(total_organic['4225'], window=48)
d_o_4225 = (total_organic['4225']-total_organic['4225'].shift(1)).dropna()

sm.tsa.seasonal_decompose(d_o_4225, model='additive',period= 48).plot()

plt.show()

test_stationarity(d_o_tv, window=48)
df_ = pd.DataFrame({'ds': total_organic.index, 'y': total_organic['4225']}).dropna()

prophet_basic = Prophet()

prophet_basic.fit(df_)

future= prophet_basic.make_future_dataframe(periods= 48, freq = 'W', include_history = False)

future['label'] = 'test'

future['y'] = np.nan

df_['label'] = 'train'

df = pd.concat((df_,future), axis = 0)

features = []

for period in range(48,96,4):

    df["lag_period_{}".format(period)] = df.y.shift(period)

    features.append("lag_period_{}".format(period))

df['lagf_mean'] = df[features].mean(axis = 1)

features.extend(['lagf_mean'])

model = XGBRegressor()

train_df = df[df.label == 'train'][features + ['y']].dropna()

test_df = df[df.label == 'test'][features]

model.fit(train_df.drop('y', axis = 1) ,train_df['y'])

forecast = model.predict(test_df)

fig = plt.figure(figsize=(12, 4))

plt.plot(future['ds'], forecast, 'k-')

plt.show()
fig = plt.figure(figsize=(12, 4))

plt.plot(total_organic.index, total_organic['4770'], 'k-')

plt.show()
sm.tsa.seasonal_decompose(total_organic['4770'], model='additive',period= 48).plot()

plt.show()
test_stationarity(total_organic['4770'], window=48)
d_o_4770 = (total_organic['4770']-total_organic['4770'].shift(1)).dropna()

sm.tsa.seasonal_decompose(d_o_4046, model='additive',period= 48).plot()

plt.show()

test_stationarity(d_o_tv, window=48)
df_ = pd.DataFrame({'ds': total_organic.index, 'y': total_organic['4770']}).dropna()

prophet_basic = Prophet()

prophet_basic.fit(df_)

future= prophet_basic.make_future_dataframe(periods= 48, freq = 'W', include_history = False)

future['label'] = 'test'

future['y'] = np.nan

df_['label'] = 'train'

df = pd.concat((df_,future), axis = 0)

features = []

for period in range(48,96,4):

    df["lag_period_{}".format(period)] = df.y.shift(period)

    features.append("lag_period_{}".format(period))

df['lagf_mean'] = df[features].mean(axis = 1)

features.extend(['lagf_mean'])

model = XGBRegressor()

train_df = df[df.label == 'train'][features + ['y']].dropna()

test_df = df[df.label == 'test'][features]

model.fit(train_df.drop('y', axis = 1) ,train_df['y'])

forecast = model.predict(test_df)

fig = plt.figure(figsize=(12, 4))

plt.plot(future['ds'], forecast, 'k-')

plt.show()