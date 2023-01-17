# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.graph_objs as go
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/covid19-in-turkey/covid_19_data_tr.csv")
data.head()
# Province/State column is null
# We work on Turkey data. We do not need Country column
data = data.drop(["Province/State","Country/Region"],axis=1)
data.head()
# confirmed_data = pd.read_csv("../input/covid19-in-turkey/time_series_covid_19_confirmed_tr.csv")
# confirmed_data = confirmed_data.drop(["Province/State", "Country/Region", "Lat", "Long"],axis=1)
# confirmed_data = confirmed_data.transpose()
# confirmed_data.head()
confirmed_data = data['Confirmed']
confirmed_data.head()
# total test numbers
tested_data = pd.read_csv("../input/covid19-in-turkey/time_series_covid_19_tested_tr.csv")
tested_data = tested_data.drop(["Province/State", "Country/Region", "Lat", "Long"],axis=1)
tested_data = tested_data.transpose()
tested_data.head()
recovered_data = data['Recovered']
recovered_data.head()
deaths_data = data['Deaths']
deaths_data.head()
intubated_data = pd.read_csv("../input/covid19-in-turkey/time_series_covid_19_intubated_tr.csv")
intubated_data = intubated_data.drop(["Province/State", "Country/Region", "Lat", "Long"],axis=1)
intubated_data = intubated_data.transpose()
intubated_data.head()
data.info()
dates_data = data['Last_Update']
dates_data.head()
# death plot
plt.plot(dates_data, deaths_data)
plt.rcParams["figure.figsize"] = [10,5]
plt.xlabel("Date")
plt.ylabel("Deaths",rotation=90)
plt.xticks(rotation=90)
plt.show()
# confirmed patient
plt.plot(dates_data, confirmed_data)
plt.rcParams["figure.figsize"] = [15,5]
plt.xlabel("Date")
plt.ylabel("Confirmed")
plt.xticks(rotation=90)
plt.show()
from plotly.offline import init_notebook_mode, iplot, plot
confirmed_scatter = go.Scatter(
    x = dates_data,
    y = confirmed_data,
    mode = "lines+markers",
    name = "Confirmed",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
    text= 'Confirmed'
)

death_scatter = go.Scatter(
    x = dates_data,
    y = deaths_data,
    mode = "lines+markers",
    name = "Deaths",
    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
    text = 'Deaths'
)

test_scatter = go.Scatter(
    x = dates_data,
    y = recovered_data,
    mode = "lines+markers",
    name = "Recovered",
    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
    text = 'Recovered'
)

layout = dict(title = 'Deaths & Confirmed & Recovered',
              xaxis= dict(title= 'Dates',ticklen= 5,zeroline= True)
             )

data_scatter = [confirmed_scatter, death_scatter, test_scatter]
fig = dict(data = data_scatter, layout = layout)
iplot(fig)
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
dates_data = dates_data.values.reshape(-1,1)
recovered_data = recovered_data.values.reshape(-1,1)
deaths_data = deaths_data.values.reshape(-1,1)
confirmed_data = confirmed_data.values.reshape(-1,1)
# scale deaths
scaler_dead = MinMaxScaler(feature_range=(0, 1))
deaths_data_sc = scaler_dead.fit_transform(deaths_data)

# scale recovered
scaler_rec = MinMaxScaler(feature_range=(0, 1))
recovered_data_sc = scaler_rec.fit_transform(recovered_data)

# scale confirmed
scaler_con = MinMaxScaler(feature_range=(0, 1))
confirmed_data_sc = scaler_con.fit_transform(confirmed_data)
train_dead_data_size = int(len(deaths_data_sc) * 0.55)
test_dead_data_size = len(deaths_data_sc) - train_dead_data_size
train_dead_data = deaths_data_sc[0:train_dead_data_size,:]
test_dead_data = deaths_data_sc[train_dead_data_size:len(deaths_data_sc),:]
print("Dead Train data size", len(train_dead_data) , "Dead Test data size", len(test_dead_data))
train_recovered_data_size = int(len(recovered_data_sc) * 0.55)
test_recovered_data_size = len(recovered_data_sc) - train_recovered_data_size
train_recovered_data = recovered_data_sc[0:train_recovered_data_size,:]
test_recovered_data = recovered_data_sc[train_recovered_data_size:len(recovered_data_sc),:]
print("Recovered Train data size", len(train_recovered_data) , "Recovered Test data size", len(test_recovered_data))
train_confirmed_data_size = int(len(confirmed_data_sc) * 0.55)
test_confirmed_data_size = len(confirmed_data_sc) - train_confirmed_data_size
train_confirmed_data = confirmed_data_sc[0:train_confirmed_data_size,:]
test_confirmed_data = confirmed_data_sc[train_confirmed_data_size:len(confirmed_data_sc),:]
print("Confirmed Train data size", len(train_confirmed_data) , "Confirmed Test data size", len(test_confirmed_data))
time_stemp = 4
datax_date = []
datay_deaths = []
datay_confirmed = []
datay_recovered = []

for i in range(len(train_dead_data)-time_stemp-1):
    a = train_dead_data[i:(i+time_stemp), 0]
    datax_date.append(a)
    datay_deaths.append(train_dead_data[i + time_stemp, 0])
trainX_deaths = np.array(datax_date)
trainY_deaths = np.array(datay_deaths)  

datax_date = []
datay_deaths = []

for i in range(len(test_dead_data)-time_stemp-1):
    a = test_dead_data[i:(i+time_stemp), 0]
    datax_date.append(a)
    datay_deaths.append(test_dead_data[i + time_stemp, 0])
testX_deaths = np.array(datax_date)
testY_deaths = np.array(datay_deaths)  
trainX_deaths = np.reshape(trainX_deaths, (trainX_deaths.shape[0], 1, trainX_deaths.shape[1]))
testX_deaths = np.reshape(testX_deaths, (testX_deaths.shape[0], 1, testX_deaths.shape[1]))
lstm_model = Sequential()
lstm_model.add(LSTM(10, input_shape=(1, time_stemp)))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(trainX_deaths, trainY_deaths, epochs=50, batch_size=1)
trainPredict = lstm_model.predict(trainX_deaths)
testPredict = lstm_model.predict(testX_deaths)

trainPredict = scaler_dead.inverse_transform(trainPredict)
trainY = scaler_dead.inverse_transform([trainY_deaths])
testPredict = scaler_dead.inverse_transform(testPredict)
testY = scaler_dead.inverse_transform([testY_deaths])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

trainPredictPlot = np.empty_like(deaths_data_sc)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_stemp:len(trainPredict)+time_stemp, :] = trainPredict

testPredictPlot = np.empty_like(deaths_data_sc)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(time_stemp*2)+1:len(deaths_data_sc)-1, :] = testPredict

plt.plot(scaler_dead.inverse_transform(deaths_data_sc))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.legend()
plt.show()