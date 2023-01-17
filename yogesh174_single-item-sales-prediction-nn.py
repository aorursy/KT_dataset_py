import pandas as pd

import numpy as np

import math

import os

import plotly.graph_objects as go

from keras.models import Sequential

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from keras.layers import Dense

from keras.layers import LSTM
os.chdir("/kaggle/input/wallmart-sales")
df = pd.read_csv("single_item.csv")

df = df.drop(['Unnamed: 0'], axis = 1)

df.head()
dataset = pd.DataFrame(df.iloc[:,0])

scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(dataset)
test_size = 30

train_size = int(len(df) - test_size)

train, test  = dataset[:train_size,0],dataset[train_size:len(dataset),0] 
def create_dataset(dataset, time_step=7):

    dataX, dataY = [], []

    m = len(dataset)

    for i in range(time_step, m):

        dataX.append(dataset[i-time_step:i])

        dataY.append(dataset[i])

    return np.array(dataX), np.array(dataY)



time_step = 7

trainX, trainY = create_dataset(train, time_step)

testX, testY = create_dataset(test, time_step)



trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

testX = np.reshape(testX, (testX.shape[0], testX.shape[1],1))
model = Sequential()

model.add(LSTM(4, input_shape=(time_step, 1)))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)
testPredict = model.predict(testX)



testPredict = scaler.inverse_transform(testPredict)

testY = scaler.inverse_transform([testY])



testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))

print('Test Score: %.2f RMSE' % (testScore))
testPredictPlot = np.empty_like(dataset)

testPredictPlot[:, :] = np.nan

testPredictPlot[len(dataset)-len(testPredict):, :] = testPredict



Data = pd.DataFrame(scaler.inverse_transform(dataset))

Data_test = pd.DataFrame(testPredictPlot)
fig = go.Figure()

fig.add_trace(go.Scatter(x=df.date, y=Data.iloc[:,0], name="Dataset",

                         line_color='deepskyblue'))

fig.add_trace(go.Scatter(x=df.date,y=Data_test.iloc[:,0], name="Dataset test",

                         line_color='red'))

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=df['date'].iloc[-23:,], y=Data.iloc[-23:,0], name="Dataset",

                         line_color='deepskyblue'))

fig.add_trace(go.Scatter(x=df['date'].iloc[-23:,],y=Data_test.iloc[-23:,0], name="Dataset test",

                         line_color='red'))

fig.show()
pred = pd.concat([Data,Data_test], axis =1)

pred = pred[-23:]

pred.columns = ['Actual sales', 'Predicted sales']

pred.head()