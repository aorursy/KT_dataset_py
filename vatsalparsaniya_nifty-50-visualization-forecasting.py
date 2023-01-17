import numpy as np 

import pandas as pd

import seaborn as sns

color = sns.color_palette()

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



import plotly.offline as py

from plotly import tools

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px



from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.statespace.sarimax import SARIMAX
nifty_50_df = pd.read_csv("../input/nifty-indices-dataset/NIFTY 50.csv", index_col='Date', parse_dates=['Date'])

nifty_bank_df = pd.read_csv("../input/nifty-indices-dataset/NIFTY BANK.csv", index_col='Date', parse_dates=['Date'])



nifty_50_df.head(5)
nifty_50_df.tail(5)
nifty_50_df = nifty_50_df.fillna(method='ffill')

nifty_bank_df = nifty_bank_df.fillna(method='ffill')
def plot_attribute(df, attritube ,start='2000', end='2020',color ='blue'):

    fig, ax = plt.subplots(1, figsize=(20,5))

    ax.plot(df[start:end].index, df[start:end][attritube],'tab:{}'.format(color))

    ax.set_title("Nifty stock {} from 2000 to 2020".format(attritube))

    

    ax.axhline(y=df[start:end].describe()[attritube]["max"],linewidth=2, color='m')

    ax.axhline(y=df[start:end].describe()[attritube]["min"],linewidth=2, color='c')

    ax.axvline(x=df[attritube].idxmax(),linewidth=2, color='b')

    ax.axvline(x=df[attritube].idxmin() ,linewidth=2, color='y')

    

    ax.text(x=df[attritube].idxmax(),

            y=df[start:end].describe()[attritube]["max"],

            s='MAX',

            horizontalalignment='right',

            verticalalignment='bottom',

            color='blue',

            fontsize=20)

    

    ax.text(x=df[attritube].idxmin(),

            y=df[start:end].describe()[attritube]["min"],

            s='MIN',

            horizontalalignment='left',

            verticalalignment='top',

            color='red',

            fontsize=20)

    

    plt.show()

    print("Max Value :  ",df[start:end].describe()[attritube]["max"])

    print("Min Value :  ",df[start:end].describe()[attritube]["min"])

plot_attribute(nifty_50_df,"Close",color='red')
plot_attribute(nifty_bank_df,"Close",color='red')
normalised_nifty_50_df = nifty_50_df["Close"].div(nifty_50_df["Close"].iloc[0]).mul(100)

normalised_nifty_bank_df = nifty_bank_df["Close"].div(nifty_bank_df['Close'].iloc[0]).mul(100)

normalised_nifty_50_df.plot()

normalised_nifty_bank_df.plot()

plt.legend(['NIFTY 50','NIFTY BANK'])

plt.show()
dicky_fuller_result = adfuller(nifty_50_df['Close'])

dicky_fuller_result
plot_attribute(nifty_50_df.diff(),"Close",color='red')
plot_attribute(nifty_50_df.shift(1)/nifty_bank_df,"Close",color='red')
some_part_of_data = nifty_50_df['2016':'2020']



rolling_nifty_50_df_10 = some_part_of_data['Close'].rolling('10D').mean()

rolling_nifty_50_df_50 = some_part_of_data['Close'].rolling('50D').mean()

rolling_nifty_50_df_100 = some_part_of_data['Close'].rolling('100D').mean()



fig, ax = plt.subplots(1, figsize=(20,5))

ax.plot(some_part_of_data.index,some_part_of_data['Close'])

ax.plot(rolling_nifty_50_df_10.index, rolling_nifty_50_df_10)

ax.plot(rolling_nifty_50_df_50.index, rolling_nifty_50_df_50)

ax.plot(rolling_nifty_50_df_100.index, rolling_nifty_50_df_100)

ax.set_title("Plotting a rolling mean of 10,50,100 day window with original Close attribute of Nifty stocks")

plt.legend(['Data','10D','50D','100D'])

plt.show()
# Obtain data from the data frame

OHLC_data = nifty_50_df['3-2020':'2020']



fig = go.Figure(data=go.Ohlc(x=OHLC_data.index,

                            open=OHLC_data['Open'],

                            high=OHLC_data['High'],

                            low=OHLC_data['Low'],

                            close=OHLC_data['Close']))



fig.update_layout(title_text='Nifty 50 From March 2020 to May 2020',

                  title={

                    'y':0.9,

                    'x':0.5,

                    'xanchor': 'center',

                    'yanchor': 'top'},

                  xaxis_rangeslider_visible=True, 

                  xaxis_title="Time", 

                  yaxis_title="Price")



fig.show()
Candlestick_data = nifty_50_df['3-2020':'2020']



fig = go.Figure(data=go.Candlestick(x=Candlestick_data.index,

                            open=Candlestick_data['Open'],

                            high=Candlestick_data['High'],

                            low=Candlestick_data['Low'],

                            close=Candlestick_data['Close']))



fig.update_layout(title_text='Nifty 50 From March 2020 to May 2020',

                  title={

                    'y':0.9,

                    'x':0.5,

                    'xanchor': 'center',

                    'yanchor': 'top'},

                  xaxis_rangeslider_visible=True, 

                  xaxis_title="Time", 

                  yaxis_title="Price")



fig.show()
decomposition_data = nifty_50_df['2018':'2020']

decomp_results = seasonal_decompose(decomposition_data['Close'], freq=7)

plt.rcParams["figure.figsize"] = (20,15)

figure = decomp_results.plot()



plt.show()
plt.figure(figsize=(10,10))



# ACF of Nifty 50 close price

ax1 = plt.subplot(211)

plot_acf(nifty_50_df["Close"], lags="20",title="nifty 50 autocorrelation",ax=ax1)



# PACF of Nifty 50 close price

ax2 = plt.subplot(212)

plot_pacf(nifty_50_df["Close"], lags="20",title="nifty 50 partial autocorrelation function",ax=ax2)



plt.show()
train_data = nifty_50_df["Close"]["2018":"4-15-2020"]

test_data =  nifty_50_df["Close"]["4-15-2020":]
order_aic_bic =[] 

# Loop over AR order 

for p in range(6): 

    # Loop over MA order 

    for q in range(3): 

        # Fit model 

        for d in range(2):

            model = SARIMAX(train_data, order=(p,d,q)) 

            results = model.fit() 

            # Add order and scores to list 

            order_aic_bic.append((p,d, q, results.aic, results.bic))
order_df = pd.DataFrame(order_aic_bic, columns=['p','d','q', 'aic', 'bic'])

#short value by aic and get value of p d q

order_df.sort_values('aic')[:5]
model = SARIMAX(train_data, order=(5,1,2)) 

results = model.fit()
mae = np.mean(np.abs(results.resid))

print(mae)
plt.rcParams["figure.figsize"] = (15,10)

results.plot_diagnostics() 

plt.show()
prediction = results.get_prediction(start="7-2019")
predictedmean = prediction.predicted_mean

p_bounds = prediction.conf_int()

p_lower_limit = p_bounds.iloc[:,0]

p_upper_limit = p_bounds.iloc[:,1]
forecast = results.get_forecast(steps=len(test_data))
mean_forecast = forecast.predicted_mean

f_bounds = forecast.conf_int()

f_lower_limit = f_bounds.iloc[:,0]

f_upper_limit = f_bounds.iloc[:,1]
plt.figure(figsize=(12,8))



plt.plot(train_data.index, train_data, label='Original Closing Price(train)')

plt.plot(test_data.index, test_data, label='Original Closing Price(test)',color='k')



plt.plot(predictedmean.index, predictedmean, color='r', label='predicted')

plt.plot(test_data.index, mean_forecast, color='m', label='Forecast')



plt.fill_between(predictedmean.index,p_lower_limit,p_upper_limit, color='yellow')

plt.fill_between(test_data.index,f_lower_limit,f_upper_limit, color='y')



plt.xlabel('Date')

plt.ylabel('Nifty 50 Closing Price')

plt.legend()

plt.show()
import keras

import keras.backend as K

K.clear_session()

from sklearn.preprocessing import MinMaxScaler
X = nifty_50_df.drop(["Close","Turnover","P/E","P/B","Div Yield"],axis=1)

y = nifty_50_df["Close"]



# Preprocessing

scaler = MinMaxScaler()

scaler_X = scaler.fit_transform(X)



X_df = pd.DataFrame(data=scaler_X, columns=["Open","High","Low","Volume"],index= X.index)

y_df = pd.DataFrame(data=y, columns=["Close"],index= y.index)



train_X_df = X_df["2000":"1-2-2020"]

test_X_df = X_df["1-2-2020":]



train_y_df = y_df["2000":"1-2-2020"]

test_y_df = y_df["1-2-2020":]



train_X = np.array(train_X_df)

test_X = np.array(test_X_df)



train_y = np.array(train_y_df)

test_y = np.array(test_y_df)



train_X = np.reshape(train_X,(train_X.shape[0],train_X.shape[1],1))

test_X = np.reshape(test_X,(test_X.shape[0],test_X.shape[1],1))



train_y = np.reshape(train_y,(train_y.shape[0],1))

test_y = np.reshape(test_y,(test_y.shape[0],1))
print("Train X shape : ", train_X.shape)

print("Test X shape : ", test_X.shape)



print("Train y shape : ", train_y.shape)

print("Test y shape : ", test_y.shape)
lstm_model = keras.models.Sequential()

lstm_model.add(keras.layers.LSTM(128,

                                 input_shape=(train_X.shape[1],1),

                                 activation='relu',

                                 return_sequences=True

                                ))



lstm_model.add(keras.layers.LSTM(64,return_sequences=False,activation='relu'))



lstm_model.add(keras.layers.Dense(32,activation='relu'))

lstm_model.add(keras.layers.Dense(1))



lstm_model.compile(loss='mean_squared_error', optimizer='adam')
history = lstm_model.fit(train_X, train_y, batch_size=128,epochs=100, verbose=1, shuffle=False)
plt.figure(figsize=(8,6))

plt.plot(history.history['loss'], label='MAE (training data)')

plt.title('MAE')

plt.ylabel('MAE value')

plt.xlabel('No. epoch')

plt.legend(loc="upper left")

plt.show()
prediction = lstm_model.predict(test_X)
np.sqrt(np.mean(((prediction - test_y) ** 2)))
test_y_df["prediction"] = np.reshape(prediction,(prediction.shape[0]))



plt.figure(figsize=(12,8))



plt.plot(nifty_50_df["10-2018":].index,nifty_50_df["10-2018":]["Close"], label="Train Price")

plt.plot(test_y_df.index,test_y_df.prediction, label="predicted Price",color='r')

plt.plot(test_y_df.index,test_y_df.Close, label="test Price",color='m')



plt.xlabel('Date')

plt.ylabel('Nifty 50 Closing Price')

plt.legend()

plt.show()