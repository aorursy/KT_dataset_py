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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize']=10,6

from datetime import datetime



dataset = pd.read_csv("../input/air-passengers/AirPassengers.csv")

# Parse strings to datetime type

dataset['Month'] = pd.to_datetime(dataset['Month'], infer_datetime_format=True)

indexedDataset = dataset.set_index(['Month'])



from datetime import datetime

indexedDataset['1949-03']

indexedDataset['1949-03':'1949-06']

indexedDataset['1949']
df = indexedDataset

df.shape
df.info( )
df.describe()
plt.xlabel("Date")

plt.ylabel("Number of Air Passengers")

plt.grid()

plt.plot(df)
#check Stationary

#determine rolling statistics



rolmean = df.rolling(window=12).mean()



rolstd = df.rolling(window=12).std()

print(rolmean, rolstd)

original_data = plt.plot(df,color='blue',label = 'original')

mean = plt.plot(rolmean,color='red', label='Rolling mean')

std = plt.plot(rolstd,color='black', label = 'Rolling std')

plt.legend(loc='best')

plt.title("Rolling Mean & Standard Deviation")

plt.xlabel("Date")

plt.ylabel("Number Air Passengers")

plt.show(block= False)
from statsmodels.tsa.stattools import adfuller



print('Results of Deckey-Fuller Test: ')

dftest = adfuller(df['#Passengers'],autolag='AIC')



dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key, value in dftest[4].items():

    dfoutput['Critical Value (%s)' %key]= value

    

print(dfoutput)
df_logScale = np.log(df)

plt.plot(df_logScale)

plt.title("Rolling Mean & Standard Deviation",color = 'darkgreen')

plt.xlabel("Date",color = 'darkblue')

plt.ylabel("Number Air Passengers",color ='darkblue')
moving_avg = df_logScale.rolling(window=12).mean()

moving_STD = df_logScale.rolling(window=12).std()

plt.plot(df_logScale)

plt.plot(moving_avg, color = 'red')
df_logScale_moving_avg = df_logScale - moving_avg

df_logScale_moving_avg.head(12)



#Remove NA values

df_logScale_moving_avg.dropna(inplace =True)

df_logScale_moving_avg.head(10)
from statsmodels.tsa.stattools import adfuller

def test_stationary(timeseries):

    

    #Determing rolling statistics

    

    moving_avg = timeseries.rolling(window=12).mean()

    moving_std = timeseries.rolling(window =12).std()

    

    #plot rolling statistics

    orig = plt.plot(timeseries, color = 'blue', label = 'Original')

    mean = plt.plot(moving_avg, color = 'red', label = 'Rolling mean')

    std = plt.plot(moving_std, color = 'darkgray', label = 'Rolling std')

    plt.legend(loc='best')

    plt.title('Rolling mean & Standard deviation')

    plt.show(block=False)

    

    #perform Dickey_Fuller test

    

    print("Results of Dickey_Fuller test: ")

    dftest = adfuller(timeseries['#Passengers'], autolag = 'AIC')

    dfoutput = pd.Series(dftest[0:4],index = ['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)

                         
test_stationary(df_logScale_moving_avg)
exp_dec_weight_avg = df_logScale.ewm(halflife=12, min_periods=0,adjust=True).mean()

plt.plot(df_logScale)

plt.plot(exp_dec_weight_avg, color = 'red')
df_logScale_moving_exp_dec_avg = df_logScale - exp_dec_weight_avg

test_stationary(df_logScale_moving_exp_dec_avg)


df_log_dif_shifting = df_logScale - df_logScale.shift()



plt.plot(df_log_dif_shifting)
df_log_dif_shifting.dropna(inplace= True)

test_stationary(df_log_dif_shifting)
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df_logScale)



trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid



plt.subplot(411)

plt.plot(df_logScale, label='Original')

plt.legend(loc='best')

plt.subplot(412)

plt.plot(trend, label='Trend')

plt.legend(loc='best')

plt.subplot(413)

plt.plot(seasonal,label='Seasonality')

plt.legend(loc='best')

plt.subplot(414)

plt.plot(residual, label='Residuals')

plt.legend(loc='best')

plt.tight_layout()



decomposed_log_data = residual

decomposed_log_data.dropna(inplace =True)

test_stationary(decomposed_log_data)
decomposed_log_data = residual

decomposed_log_data.dropna(inplace=True)

test_stationary(decomposed_log_data)
from statsmodels.tsa.stattools import acf, pacf



lag_acf = acf(df_log_dif_shifting, nlags=20)

lag_pacf = pacf(df_log_dif_shifting, nlags=20, method='ols')





#Plot ACF: 

plt.subplot(121) 

plt.plot(lag_acf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(df_log_dif_shifting)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(df_log_dif_shifting)),linestyle='--',color='gray')

plt.title('Autocorrelation Function')



#Plot PACF:

plt.subplot(122)

plt.plot(lag_pacf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(df_log_dif_shifting)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(df_log_dif_shifting)),linestyle='--',color='gray')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()
from statsmodels.tsa.arima_model import ARIMA



#AR MODEL

model = ARIMA(df_logScale, order=(2, 1, 0))  

results_AR = model.fit(disp=-1)  

plt.plot(df_log_dif_shifting)

plt.plot(results_AR.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-df_log_dif_shifting["#Passengers"])**2))

print('Plotting AR model')
#MA MODEL

model = ARIMA(df_logScale, order=(0, 1, 2))  

results_MA = model.fit(disp=-1)  

plt.plot(df_log_dif_shifting)

plt.plot(results_MA.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-df_log_dif_shifting["#Passengers"])**2))

print('Plotting AR model')
model = ARIMA(df_logScale, order=(2, 1, 2))  

results_ARIMA = model.fit(disp=-1)  

plt.plot(df_log_dif_shifting)

plt.plot(results_ARIMA.fittedvalues, color='red')

plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-df_log_dif_shifting["#Passengers"])**2))
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)

print (predictions_ARIMA_diff.head())
#Convert to cumulative sum

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

print (predictions_ARIMA_diff_cumsum.head())
#predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)

#predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

#predictions_ARIMA_log.head()



predictions_ARIMA_log = pd.Series(df_logScale['#Passengers'].ix[0], index=df_logScale['#Passengers'].index)

predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

predictions_ARIMA_log.head()
predictions_ARIMA = np.exp(predictions_ARIMA_log)

plt.plot(indexedDataset)

plt.plot(predictions_ARIMA)

plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-df_logScale["#Passengers"])**2)/len(df["#Passengers"])))
df_logScale
results_ARIMA.plot_predict(1,264)

x = results_ARIMA.forecast(steps=120)