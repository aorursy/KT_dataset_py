import warnings

warnings.filterwarnings('ignore')

import numpy as np 

import pandas as pd 

from pandas import DataFrame

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import matplotlib.ticker as mticker

from matplotlib.finance import candlestick_ohlc

import seaborn as sns

#importing packages for the prediction of time-series data

from fbprophet import Prophet

from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error



#configuring the Environment

color = sns.color_palette()

%matplotlib inline



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



crypto_data = {}



crypto_data['bitcoin'] = pd.read_csv('../input/bitcoin_price.csv', parse_dates=['Date'])

crypto_data['bitcoin_cash'] = pd.read_csv("../input/bitcoin_cash_price.csv", parse_dates=['Date'])

crypto_data['dash'] = pd.read_csv("../input/dash_price.csv", parse_dates=['Date'])

crypto_data['ethereum'] = pd.read_csv("../input/ethereum_price.csv", parse_dates=['Date'])

crypto_data['iota'] = pd.read_csv("../input/iota_price.csv", parse_dates=['Date'])

crypto_data['litecoin'] = pd.read_csv("../input/litecoin_price.csv", parse_dates=['Date'])

crypto_data['monero'] = pd.read_csv("../input/monero_price.csv", parse_dates=['Date'])

crypto_data['nem'] = pd.read_csv("../input/nem_price.csv", parse_dates=['Date'])

crypto_data['neo'] = pd.read_csv("../input/neo_price.csv", parse_dates=['Date'])

crypto_data['numeraire'] = pd.read_csv("../input/numeraire_price.csv", parse_dates=['Date'])

crypto_data['ripple'] = pd.read_csv("../input/ripple_price.csv", parse_dates=['Date'])

crypto_data['stratis'] = pd.read_csv("../input/stratis_price.csv", parse_dates=['Date'])

crypto_data['waves'] = pd.read_csv("../input/waves_price.csv", parse_dates=['Date'])



for coin in crypto_data:

    df = pd.DataFrame(crypto_data[coin])

    df = df[['Date' , 'Close']]

    

    df['Date_mpl'] = df['Date'].apply(lambda x: mdates.date2num(x)) # making new column 'Date_mpl' by using date2num lamba function

    fig, ax = plt.subplots(figsize=(6,4))

    sns.tsplot(df.Close.values, time=df.Date_mpl.values, alpha=0.8, color=color[3], ax=ax)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

    fig.autofmt_xdate()

    plt.xlabel('Date', fontsize=12)

    plt.ylabel('Price in USD', fontsize=12)

    title_str = "Closing price distribution of " + coin

    plt.title(title_str, fontsize=15)

    plt.show()
for coin in crypto_data:

    df = pd.DataFrame(crypto_data[coin])

    fig = plt.figure(figsize=(6,4))

    ax1 = plt.subplot2grid((1,1), (0,0))

    

    df['Date_mpl'] = df['Date'].apply(lambda x: mdates.date2num(x))

    temp_df = df[df['Date']>'2017-05-01']

    ohlc = []

    for ind, row in temp_df.iterrows():

        ol = [row['Date_mpl'],row['Open'], row['High'], row['Low'], row['Close'], row['Volume']]

        ohlc.append(ol)



    candlestick_ohlc(ax1, ohlc, width=0.4, colorup='#77d879', colordown='#db3f3f')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))



    plt.xlabel("Date", fontsize=12)

    plt.xticks(rotation='vertical')

    plt.ylabel("Price in USD", fontsize=12 )

    title_str = "Candlestick chart for " + coin

    plt.title(title_str, fontsize=15)

    plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)

    plt.show()
del crypto_data['iota'], crypto_data['numeraire'], crypto_data['bitcoin_cash']
df = pd.DataFrame() 

currency_name = []

df['Date'] = crypto_data['bitcoin'].Date 

df = df[df['Date']>'2017-05-01']

for coin in crypto_data:

    currency_name.append(coin)

    temp_df = crypto_data[coin]

    df[coin] = temp_df[temp_df['Date']>'2017-05-01'].Close



temp_df = df[currency_name]

corrmat = temp_df.corr(method='spearman')

fig, ax = plt.subplots(figsize=(5, 5))

sns.heatmap(corrmat, vmax=1., square=True)

plt.title("Spearman correlation map", fontsize=15)

plt.show()

temp_df.corr(method='spearman')
df = pd.DataFrame() 

currency_name = []

df['Date'] = crypto_data['bitcoin'].Date 

df = df[df['Date']>'2017-05-01']

for coin in crypto_data:

    currency_name.append(coin)

    temp_df = crypto_data[coin]

    df[coin] = temp_df[temp_df['Date']>'2017-05-01'].Close



temp_df = df[currency_name]

corrmat = temp_df.corr(method='pearson')

fig, ax = plt.subplots(figsize=(5, 5))

sns.heatmap(corrmat, vmax=1., square=True)

plt.title("Pearson correlation map", fontsize=15)

plt.show()

temp_df.corr(method='pearson')
df = pd.DataFrame() 

currency_name = []

df['Date'] = crypto_data['bitcoin'].Date 

df = df[df['Date']>'2017-05-01']

for coin in crypto_data:

    currency_name.append(coin)

    temp_df = crypto_data[coin]

    df[coin] = temp_df[temp_df['Date']>'2017-05-01'].Close



temp_df = df[currency_name]

corrmat = temp_df.corr(method='kendall')

fig, ax = plt.subplots(figsize=(5, 5))

sns.heatmap(corrmat, vmax=1., square=True)

plt.title("kendall correlation map", fontsize=15)

plt.show()

temp_df.corr(method='kendall')
# for coin in crypto_data:

#     df = pd.DataFrame(crypto_data[coin])

#     temp_df = pd.DataFrame()

#     temp_df['ds'] = df['Date']

#     temp_df['y'] = df['Close']

#     temp_df['ds'] = temp_df['ds'].dt.to_pydatetime()

#     model = Prophet()

#     model.fit(temp_df)

#     future = model.make_future_dataframe(periods = 60)

#     forecast = model.predict(future)

#     title_str = "predicted value of "+ coin

#     model.plot(forecast, uncertainty=False)

#     model.plot_components(forecast, uncertainty=False)
df_bitcoin = pd.DataFrame(crypto_data['bitcoin'])



df_bitcoin = df_bitcoin[['Date','Close']]

df_bitcoin.set_index('Date', inplace = True)

# fit model

model = ARIMA(df_bitcoin, order=(5,1,0))

model_fit = model.fit(disp=0)

print(model_fit.summary())

# plot residual errors

residuals = DataFrame(model_fit.resid)

residuals.plot()

plt.show()

residuals.plot(kind='kde')

plt.show()

print(residuals.describe())


X = df_bitcoin.values

size = int(len(X) * 0.80)

train, test = X[0:size], X[size:len(X)]

history = [x for x in train]

predictions = list()

for t in range(len(test)):

    model = ARIMA(history, order=(5,1,0))

    model_fit = model.fit(disp=0)

    output = model_fit.forecast()

    yhat = output[0]

    predictions.append(yhat)

    obs = test[t]

    history.append(obs)

    #print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)

print('Test MSE: %.3f' % error)

# plot

plt.plot(test)

plt.plot(predictions, color='red')

plt.show()