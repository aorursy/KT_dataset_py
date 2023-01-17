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
df = pd.read_excel("/kaggle/input/world-bank-ppl-growth/people_growth.xlsx")

df = df.set_index('Country')

import matplotlib.pyplot as plt



plt.figure(figsize=(18,14))

plt.ylim(-8,35)

plt.grid(True)

china_=df.loc["China"]

#plt.plot(df.columns,df.loc["Australia"], '-o') 

#plt.plot(df.columns,df.loc["Canada"], '-o') 

plt.plot(df.columns,df.loc["China"], '-o') 

plt.plot(df.columns,df.loc["Germany"], '-o') 

#plt.plot(df.columns,df.loc["France"], '-o') 

#plt.plot(df.columns,df.loc["United Kingdom"], '-o') 

#plt.plot(df.columns,df.loc["Hong Kong SAR, China"], '-o') 

plt.plot(df.columns,df.loc["India"], '-o') 

plt.plot(df.columns,df.loc["Japan"], '-o') 

#plt.plot(df.columns,df.loc["Mexico"], '-o') 

#plt.plot(df.columns,df.loc["Malaysia"], '-o') 

#plt.plot(df.columns,df.loc["United States"], '-o') 

#plt.plot(df.columns,df.loc["Vietnam"], '-o')

plt.xlabel('year')

plt.ylabel('million')

plt.legend()

plt.show()



import numpy

import math

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt2





# 產生 (X, Y) 資料集, Y 是下一期的乘客數

def create_dataset(dataset, look_back=1):

    dataX, dataY = [], []

    for i in range(len(dataset)-look_back-1):

        a = dataset[i:(i+look_back), 0]

        dataX.append(a)

        dataY.append(dataset[i + look_back, 0])

    return numpy.array(dataX), numpy.array(dataY)



# 載入訓練資料

dataset=df.loc["China"]

dataset = dataset.astype('float32')

# 正規化(normalize) 資料，使資料值介於[0, 1]

scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(dataset)



# 2/3 資料為訓練資料， 1/3 資料為測試資料

train_size = int(len(dataset) * 0.67)

test_size = len(dataset) - train_size

train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]



# 產生 (X, Y) 資料集, Y 是下一期的乘客數(reshape into X=t and Y=t+1)

look_back = 1

trainX, trainY = create_dataset(train, look_back)

testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))



# 建立及訓練 LSTM 模型

model = Sequential()

model.add(LSTM(4, input_shape=(1, look_back)))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)



# 預測

trainPredict = model.predict(trainX)

testPredict = model.predict(testX)



# 回復預測資料值為原始數據的規模

trainPredict = scaler.inverse_transform(trainPredict)

trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)

testY = scaler.inverse_transform([testY])



# calculate 均方根誤差(root mean squared error)

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))

print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))

print('Test Score: %.2f RMSE' % (testScore))



# 畫訓練資料趨勢圖

# shift train predictions for plotting

trainPredictPlot = numpy.empty_like(dataset)

trainPredictPlot[:, :] = numpy.nan

trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict



# 畫測試資料趨勢圖

# shift test predictions for plotting

testPredictPlot = numpy.empty_like(dataset)

testPredictPlot[:, :] = numpy.nan

testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict



# 畫原始資料趨勢圖

# plot baseline and predictions

plt2.plot(scaler.inverse_transform(dataset))

plt2.plot(trainPredictPlot)

plt2.plot(testPredictPlot)

plt2.show()




