import pandas as pd

import numpy as np
data_train = pd.read_csv('../input/Google_Stock_Price_Train.csv')



data_train.tail()
train = data_train.iloc[:,1:2].values
from sklearn.preprocessing import MinMaxScaler



mms = MinMaxScaler()

train = mms.fit_transform(train)
x_train = []

y_train = []



for i in range(60,1258):

    x_train.append(train[i-60:i,0])

    y_train.append(train[i,0])
x_train = np.array(x_train)

y_train = np.array(y_train)
x_train.shape
x_train
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
from keras.models import Sequential

from keras.layers import Dense,Dropout,LSTM,CuDNNLSTM
model = Sequential()

model.add(CuDNNLSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))

model.add(Dropout(0.2))



model.add(CuDNNLSTM(50,return_sequences=True))

model.add(Dropout(0.2))



model.add(CuDNNLSTM(50,return_sequences=True))

model.add(Dropout(0.2))



model.add(CuDNNLSTM(50))

model.add(Dropout(0.2))



model.add(Dense(1))



model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train,y_train,epochs=100,batch_size=32)
data_test = pd.read_csv('../input/Google_Stock_Price_Test.csv')



data_test.head()
real_values = data_test['Open'].values
real_values
data = pd.concat((data_train['Open'],data_test['Open']),axis=0)
data
inputs = data[len(data) - len(data_test) - 60:]
inputs.iloc[59]
inputs = data[len(data) - len(data_test) - 60:].values
inputs.shape
inputs = inputs.reshape(-1,1)
inputs = mms.fit_transform(inputs)
x_test = []



for i in range(60,80):

    x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test)

x_test = x_test.reshape(20,60,1)

x_test.shape
prediction = model.predict(x_test)
prediction = mms.inverse_transform(prediction)
import matplotlib.pyplot as plt





plt.plot(prediction, color='blue', label='Predicted Values')

plt.plot(real_values, color='red', label='Real Values')

plt.legend()



plt.show()
from sklearn.metrics import mean_squared_error



mean_squared_error(real_values,prediction)