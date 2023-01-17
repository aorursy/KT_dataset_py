import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 

from statsmodels.tsa.seasonal import seasonal_decompose 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/time-series-datasets/Electric_Production.csv',parse_dates=[0])

df=df.rename(columns={'IPG2211A2N':'usage','DATE':'date'})

df = df.set_index('date')

df.head()
%matplotlib inline

df.plot(figsize=(8,4))
seasonal = seasonal_decompose(df.usage,model='add')

fig = plt.figure()  

fig = seasonal.plot()  

fig.set_size_inches(10, 8)
from statsmodels.tsa.stattools import adfuller   #Dickey-Fuller test

def test_stationarity(timeseries):



    #Determing rolling statistics

    rolmean = timeseries.rolling(window=20).mean()

    rolstd = timeseries.rolling(window=20).std()



    #Plot rolling statistics:

    fig = plt.figure(figsize=(12, 6))

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show()



    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')  #autolag : {‘AIC’, ‘BIC’, ‘t-stat’, None}

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)
test_stationarity(df.usage)
df['first_difference'] = df.usage - df.usage.shift(1)   

test_stationarity(df.first_difference.dropna(inplace=False))
!pip install pmdarima
import pmdarima as pm

from pmdarima.model_selection import train_test_split
df1=df.drop(columns='first_difference')
train, test = train_test_split(df1, train_size=320)
train, test = train_test_split(df1, train_size=320)



# Fit your model

model = pm.auto_arima(train, seasonal=True, m=12)
model.summary()
from statsmodels.tsa.statespace.sarimax import SARIMAX

pred_model = SARIMAX(train.usage, order=(1,0,2), seasonal_order=(0,1,1,12))

results = pred_model.fit()
test_pred=test.copy()

test_pred = results.predict(start = len(train), end = len(df)-1, typ="levels")  
test['usage'].plot(figsize = (12,5), label='real usage')

test_pred.plot(label = 'predicted usage')

plt.legend(loc='upper right')
from statsmodels.tools.eval_measures import rmse

arima_rmse_error = rmse(test['usage'], test_pred)

arima_mse_error = arima_rmse_error**2

print(f'MSE Error: {arima_mse_error}\nRMSE Error: {arima_rmse_error}')
train1=pd.concat([train, train.shift(-1), train.shift(-2),train.shift(-3),train.shift(-4),train.shift(-5),

                 train.shift(-6),train.shift(-7),train.shift(-8),train.shift(-9),train.shift(-10),train.shift(-11),train.shift(-12)

                 ], axis=1).dropna()

train1.columns = ['usage', 'usage1', 'usage2','usage3','usage4', 'usage5','usage6'

                 ,'usage7', 'usage8','usage9','usage10', 'usage11', 'usage12']

train1.head()
test1=pd.concat([test, test.shift(-1), test.shift(-2),test.shift(-3),test.shift(-4),test.shift(-5),

                 test.shift(-6),test.shift(-7),test.shift(-8),test.shift(-9),test.shift(-10),test.shift(-11),test.shift(-12)

                 ], axis=1).dropna()

test1.columns = ['usage', 'usage1', 'usage2','usage3','usage4', 'usage5','usage6'

                 ,'usage7', 'usage8','usage9','usage10', 'usage11','usage12']

test1.head()
train1_y=train1.loc[:, train1.columns == 'usage']

train1_x=train1.loc[:, train1.columns != 'usage']



test1_y=test1.loc[:, test1.columns == 'usage']

test1_x=test1.loc[:, test1.columns != 'usage']
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM



model = Sequential()



model.add(LSTM(20, activation='relu',input_shape=(12, 1)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse',metrics=['mean_squared_error'])



model.summary()
train1_x = np.expand_dims(train1_x, 2)

test1_x = np.expand_dims(test1_x, 2)

print("New train data shape:")

print(train1_x.shape)

print("New test data shape:")

print(test1_x.shape)
run=model.fit(train1_x,train1_y,epochs=40)
plt.plot(run.epoch,run.history.get('loss'))
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers
model.evaluate(test1_x,test1_y)
test1_pred=model.predict(test1_x)

test_pred=pd.DataFrame(test1_pred, columns=['test_pred']) 

#test_true=pd.DataFrame(test1_y, columns=['test_true']) 

test_pred.index=test1_y.index

test_pred=test_pred.merge(test1_y,left_index=True, right_index=True)
plt.figure(figsize=(12,5))

plt.plot( test_pred.index, 'usage', data=test_pred, markerfacecolor='blue', markersize=12, color='skyblue', linewidth=2,label='reality')

plt.plot( test_pred.index, 'test_pred', data=test_pred, color='orange', linewidth=2,label='prediction')

plt.legend(loc='upper right')