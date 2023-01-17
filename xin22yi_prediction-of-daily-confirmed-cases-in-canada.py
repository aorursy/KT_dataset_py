import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM
data=pd.read_csv('../input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv')
#extract the all Canada rows and sum up to get the total confrimed cases

Canada_=data[data['Country/Region'].isin(['Canada'])]

Canada=Canada_.drop(['Province/State','Country/Region','Lat','Long'],axis=1)

Canada_nd=Canada.values

Canada_sum=Canada_nd.sum(axis=0)

dataset=Canada_sum.reshape(87,1)

scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(dataset)

dataset=dataset.reshape(1,87)

Canada_

#use past look_back days to predict

def create_dataset(dataset, look_back):

    dataX, dataY = [], []

    for i in range(dataset.shape[1]-look_back-1):

        a = dataset[0,i:(i+look_back)]

        dataX.append(a)

        dataY.append(dataset[0,i + look_back])

    return np.array(dataX), np.array(dataY)
X,Y=create_dataset(dataset, 3)

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.1,random_state=42)

xtrain=xtrain.reshape(xtrain.shape[0],3,1)

xtest=xtest.reshape(xtest.shape[0],3,1)
#create model

def create_model():

  model = Sequential()

  model.add(LSTM(50,input_shape=(3,1),return_sequences=True,activation='relu'))

  model.add(LSTM(100,activation='relu'))



  model.add(Dense(1))

  return model
model=create_model( )

model.compile(optimizer='adam', loss='mse',metrics=['mse'])

history = model.fit(xtrain,ytrain, epochs=200, verbose=1,validation_data=[xtest,ytest])
model.summary()
#predict future 10 days

predictions=[Canada_sum[-1]]

new=dataset.copy()

for i in range(10):

  prediction=model.predict(dataset[0][-3:].reshape(1,3,1))

  predictions.append(int(prediction*predictions[-1]))

  new=np.append(new,prediction)
#plot



plt.plot([87,88,89,90,91,92,93,94,95,96],predictions[1:])

plt.plot(Canada_sum)

plt.legend(['Origin', 'Prediction'], loc='upper left')

plt.show()