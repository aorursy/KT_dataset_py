import sys 
import warnings
import numpy as np
from numpy import log
import pandas as pd
import matplotlib.pyplot as plt
import itertools

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

#from numpy.random import seed

from scipy import signal
from scipy.stats import randint

import seaborn as sns # used for plot interactive graph. 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline # pipeline making
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score

## for Deep-learing:
import keras
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import emd # must be added to Kaggle default kernel with 'pip install emd' in console

%matplotlib inline
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
# read csv into dataframe
df = pd.read_csv('../input/residential-power-usage-3years-data-timeseries/power_usage_2016_to_2020.csv', parse_dates=['StartDate'])

df.head()
print('Beginning of data:', df['StartDate'].min())
print('End of data:', df['StartDate'].max())
fig = plt.plot(df['Value (kWh)'][0:168].values)
plt.show()
df['Value (kWh)'].fillna(method='ffill')
df.drop('day_of_week',axis=1,inplace=True)
df.drop('notes',axis=1,inplace=True)
#df.rename(columns={"Value (kWh)": "Load (kWh)"},inplace=True)
df.rename(columns={"StartDate": "Datetime"},inplace=True)
df.head()
d = df['Value (kWh)'].values
result = adfuller(d)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
plt.hist(d, bins=range(int(min(d)),int(max(d)),1), log=True)
plt.show()
plt.hist(log(d))
plt.show()
split = round(len(d) / 2 )
d1, d2 = d[0:split], d[split:]
mean1, mean2 = d1.mean(), d2.mean()
var1, var2 = d1.var(), d2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))
plot_acf(d,lags=168)
plt.show()
plot_acf(d,lags=8760)
plt.show()
# data frame EMD
dfemd = df.copy(deep=True)

v = dfemd['Value (kWh)'].values
imf = emd.sift.sift(v)

print('Number of different IMFs: ',imf.shape[1])
fig = emd.plotting.plot_imfs(imf, scale_y=True, cmap=True)
for i in range(imf.shape[1]):
    dfemd['IMF%s'%(i+1)] = imf[:,i]
    

c = dfemd.drop('Datetime',axis=1).corr()
c.style.background_gradient(cmap='coolwarm').set_precision(2)

dfemd.drop('IMF4',axis=1, inplace=True)
dfemd.drop('IMF5',axis=1, inplace=True)
dfemd.drop('IMF6',axis=1, inplace=True)
dfemd.drop('IMF7',axis=1, inplace=True)
dfemd.drop('IMF8',axis=1, inplace=True)
dfemd.drop('IMF10',axis=1, inplace=True)

dfemd.head()
Xdf = df.copy(deep=True)
Xdf.rename(columns={"Value (kWh)": "t"},inplace=True)
Xdf.drop('Datetime',axis=1,inplace=True)
n_in = 168 # number of inputs

for i in range(1,n_in):
    Xdf.insert(0, 't-%sh'%i, Xdf['t'].shift(i), True)    

Ydf = df.copy(deep=True)
Ydf.rename(columns={"Value (kWh)": "t"},inplace=True)
Ydf.drop('Datetime',axis=1,inplace=True)
n_out = 24  # number of outputs

for i in range(1,n_out+1):
    Ydf['t+%sh'%i]=Ydf['t'].shift(-i)
    
Ydf.drop('t',axis=1,inplace=True)
Xdf = Xdf[n_in-1 : -n_out]
Ydf = Ydf[n_in-1 : -n_out]
print('Xdf.shape: ',Xdf.shape)
print('Ydf.shape: ',Ydf.shape)
Xdf
Ydf
split2 = 0.6
L = Xdf.shape[0]
i_split = int(L*split2)


Xdf_train = Xdf[0:i_split]
Xdf_valid = Xdf[i_split:]

Ydf_train = Ydf[0:i_split]
Ydf_valid = Ydf[i_split:]

print('Xdf_train.shape: ',Xdf_train.shape)
print('Xdf_valid.shape: ',Xdf_valid.shape)
print('Ydf_train.shape: ',Ydf_train.shape)
print('Ydf_valid.shape: ',Ydf_valid.shape)
epochs = 40
batch = 256
lr = 0.0003
adam = optimizers.Adam(lr)
model_mlp = Sequential()

model_mlp.add(Dense(100, activation='relu', input_dim=Xdf_train.shape[1]))
model_mlp.add(Dense(Ydf_train.shape[1]))

model_mlp.compile(loss='mse', optimizer=adam)
model_mlp.summary()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

mlp_history =  model_mlp.fit(Xdf_train.values, Ydf_train.values, validation_data=(Xdf_valid, Ydf_valid), epochs=epochs, verbose=2, callbacks=[es])
mlp_train_pred = model_mlp.predict(Xdf_train.values)
mlp_valid_pred = model_mlp.predict(Xdf_valid.values)
print('Train rmse: {:.3f}'.format(np.sqrt(mean_squared_error(Ydf_train.values, mlp_train_pred))))
print('Validation rmse: {:.3f}'.format(np.sqrt(mean_squared_error(Ydf_valid.values, mlp_valid_pred))))
plt.plot(mlp_history.history['loss'])
plt.plot(mlp_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'validation loss'], loc='upper right')
plt.show()
Y_valid_hat = model_mlp.predict(Xdf_valid.values)
k=72
t=np.arange(0,k)
hor = 0

# Horizon is the maximum value
plt.plot(t,Ydf_valid.values[0:k,hor],t,Y_valid_hat[0:k,hor])
plt.ylabel('kWh')
plt.xlabel('hrs')
plt.title('%s Hour Horizon'%(hor+1))
plt.legend(['Y_valid','Y_valid_hat'])
plt.show()
k=72
t=np.arange(0,k)
hor = 11

# Horizon is the maximum value
plt.plot(t,Ydf_valid.values[0:k,hor],t,Y_valid_hat[0:k,hor])
plt.ylabel('kWh')
plt.xlabel('hrs')
plt.legend(['Y_valid','Y_valid_hat'])
plt.title('%s Hour Horizon'%(hor+1))
plt.show()
k=72
t=np.arange(0,k)
hor = 23

# Horizon is the maximum value
plt.plot(t,Ydf_valid.values[0:k,hor],t,Y_valid_hat[0:k,hor])
plt.ylabel('kWh')
plt.xlabel('hrs')
plt.legend(['Y_valid','Y_valid_hat'])
plt.title('%s Hour Horizon'%(hor+1))
plt.show()
x=np.arange(0,n_out)
t=0

plt.plot(x,Ydf_valid.values[t,:],x,Y_valid_hat[t,:])
plt.ylabel('kWh')
plt.xlabel('hrs')
plt.legend(['Y_valid','Y_valid_hat'])
plt.show()
model_mlp1 = Sequential()

model_mlp1.add(Dense(200, activation='relu', input_dim=Xdf_train.shape[1]))
model_mlp1.add(Dense(Ydf_train.shape[1]))

model_mlp1.compile(loss='mse', optimizer=adam)
model_mlp1.summary()

mlp_history1 = model_mlp1.fit(Xdf_train.values, Ydf_train.values, validation_data=(Xdf_valid, Ydf_valid), epochs=epochs, verbose=2, callbacks=[es])

mlp_train_pred1 = model_mlp1.predict(Xdf_train.values)
mlp_valid_pred1 = model_mlp1.predict(Xdf_valid.values)
print('Train rmse: {:.3f}'.format(np.sqrt(mean_squared_error(Ydf_train.values, mlp_train_pred1))))
print('Validation rmse: {:.3f}'.format(np.sqrt(mean_squared_error(Ydf_valid.values, mlp_valid_pred1))))

Y_valid_hat1 = model_mlp1.predict(Xdf_valid.values)

x=np.arange(0,n_out)
t=0

plt.plot(x,Ydf_valid.values[t,:],x,Y_valid_hat1[t,:])
plt.ylabel('kWh')
plt.xlabel('hrs')
plt.legend(['Y_valid','Y_valid_hat'])
plt.show()
model_mlp2 = Sequential()

model_mlp2.add(Dense(100, activation='relu', input_dim=Xdf_train.shape[1]))
model_mlp2.add(Dense(100))
model_mlp2.add(Dense(Ydf_train.shape[1]))

model_mlp2.compile(loss='mse', optimizer=adam)
model_mlp2.summary()

mlp_history2 = model_mlp2.fit(Xdf_train.values, Ydf_train.values, validation_data=(Xdf_valid, Ydf_valid), epochs=epochs, verbose=2, callbacks=[es])

mlp_train_pred2 = model_mlp2.predict(Xdf_train.values)
mlp_valid_pred2 = model_mlp2.predict(Xdf_valid.values)
print('Train rmse: {:.3f}'.format(np.sqrt(mean_squared_error(Ydf_train.values, mlp_train_pred2))))
print('Validation rmse: {:.3f}'.format(np.sqrt(mean_squared_error(Ydf_valid.values, mlp_valid_pred2))))

Y_valid_hat2 = model_mlp2.predict(Xdf_valid.values)

x=np.arange(0,n_out)
t=0

plt.plot(x,Ydf_valid.values[t,:],x,Y_valid_hat2[t,:])
plt.ylabel('kWh')
plt.xlabel('hrs')
plt.legend(['Y_valid','Y_valid_hat'])
plt.show()
model_mlp3 = Sequential()

model_mlp3.add(Dense(200, activation='relu', input_dim=Xdf_train.shape[1]))
model_mlp3.add(Dense(200))
model_mlp3.add(Dense(Ydf_train.shape[1]))

model_mlp3.compile(loss='mse', optimizer=adam)
model_mlp3.summary()

mlp_history3 = model_mlp3.fit(Xdf_train.values, Ydf_train.values, validation_data=(Xdf_valid, Ydf_valid), epochs=epochs, verbose=2, callbacks=[es])

mlp_train_pred3 = model_mlp3.predict(Xdf_train.values)
mlp_valid_pred3 = model_mlp3.predict(Xdf_valid.values)
print('Train rmse: {:.3f}'.format(np.sqrt(mean_squared_error(Ydf_train.values, mlp_train_pred3))))
print('Validation rmse: {:.3f}'.format(np.sqrt(mean_squared_error(Ydf_valid.values, mlp_valid_pred3))))

Y_valid_hat3 = model_mlp3.predict(Xdf_valid.values)

x=np.arange(0,n_out)
t=0

plt.plot(x,Ydf_valid.values[t,:],x,Y_valid_hat3[t,:])
plt.ylabel('kWh')
plt.xlabel('hrs')
plt.legend(['Y_valid','Y_valid_hat'])
plt.show()
model_mlp4 = Sequential()

model_mlp4.add(Dense(1000, activation='relu', input_dim=Xdf_train.shape[1]))
model_mlp4.add(Dense(Ydf_train.shape[1]))

model_mlp4.compile(loss='mse', optimizer=adam)
model_mlp4.summary()

mlp_history4 = model_mlp4.fit(Xdf_train.values, Ydf_train.values, validation_data=(Xdf_valid, Ydf_valid), epochs=epochs, verbose=2, callbacks=[es])

mlp_train_pred4 = model_mlp4.predict(Xdf_train.values)
mlp_valid_pred4 = model_mlp4.predict(Xdf_valid.values)
print('Train rmse: {:.3f}'.format(np.sqrt(mean_squared_error(Ydf_train.values, mlp_train_pred4))))
print('Validation rmse: {:.3f}'.format(np.sqrt(mean_squared_error(Ydf_valid.values, mlp_valid_pred4))))

Y_valid_hat4 = model_mlp4.predict(Xdf_valid.values)

x=np.arange(0,n_out)
t=0

plt.plot(x,Ydf_valid.values[t,:],x,Y_valid_hat4[t,:])
plt.ylabel('kWh')
plt.xlabel('hrs')
plt.legend(['Y_valid','Y_valid_hat'])
plt.show()
X_train_3d = Xdf_train.values.reshape(Xdf_train.shape[0],Xdf_train.shape[1],1)
X_valid_3d = Xdf_valid.values.reshape(Xdf_valid.shape[0],Xdf_valid.shape[1],1)

Y_train_3d = Ydf_train.values.reshape(Ydf_train.shape[0],Ydf_train.shape[1],1)
Y_valid_3d = Ydf_valid.values.reshape(Ydf_valid.shape[0],Ydf_valid.shape[1],1)

print('X_train_3d shape: ',X_train_3d.shape)
print('X_valid_3d shape: ',X_valid_3d.shape)
print('Y_train_3d shape: ',Y_train_3d.shape)
print('Y_valid_3d shape: ',Y_valid_3d.shape)
print('X_train[:3,:5,0]:\n',X_train_3d[:3,:5,0])


model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_3d.shape[1], 1)))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dense(Y_train_3d.shape[1]))
model_cnn.compile(loss='mse', optimizer=adam)
model_cnn.summary()
cnn_history = model_cnn.fit(X_train_3d, Y_train_3d, validation_data=(X_valid_3d, Y_valid_3d), epochs=epochs, verbose=2)
cnn_train_pred = model_cnn.predict(X_train_3d)
cnn_valid_pred = model_cnn.predict(X_valid_3d)
print('Ydf_train shape',Ydf_train.shape)
print('cnn_train_pred shape',cnn_train_pred.shape)
print('Train rmse: {:.3f}'.format(np.sqrt(mean_squared_error(Ydf_train.values, cnn_train_pred))))
print('Validation rmse: {:.3f}'.format(np.sqrt(mean_squared_error(Ydf_valid.values, cnn_valid_pred))))
Y_valid_hat_cnn = model_cnn.predict(X_valid_3d)
print('Y_valid_hat_cnn.shape:',Y_valid_hat_cnn.shape)
k=120
t=np.arange(0,k)

plt.plot(t,Y_valid_3d[0:k,-1],t,Y_valid_hat_cnn[0:k,-1])
plt.ylabel('kWh')
plt.xlabel('hrs')
plt.legend(['Y_valid','Y_valid_hat'])
plt.show()
t=np.arange(0,n_out)

plt.plot(t,Y_valid_3d[0,:],t,Y_valid_hat_cnn[0,:])
plt.ylabel('kWh')
plt.xlabel('hrs')
plt.legend(['Y_valid','Y_valid_hat'])
plt.show()
model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(X_train_3d.shape[1], X_train_3d.shape[2])))
model_lstm.add(Dense(Y_train_3d.shape[1]))
model_lstm.compile(loss='mse', optimizer=adam)
model_lstm.summary()
lstm_history = model_lstm.fit(X_train_3d, Y_train_3d, validation_data=(X_valid_3d, Y_valid_3d), epochs=epochs, verbose=2)
Y_valid_hat_lstm = model_lstm.predict(X_valid_3d)
Y_valid_hat_lstm.shape
lstm_train_pred = model_lstm.predict(X_train_3d)
lstm_valid_pred = model_lstm.predict(X_valid_3d)
print('Train rmse: {:.3f}'.format(np.sqrt(mean_squared_error(Ydf_train.values, lstm_train_pred))))
print('Validation rmse: {:.3f}'.format(np.sqrt(mean_squared_error(Ydf_valid.values, lstm_valid_pred))))
k=72
t=np.arange(0,k)

plt.plot(t,Y_valid_3d[0:k,-1],t,Y_valid_hat_lstm[0:k,-1])
plt.ylabel('kWh')
plt.xlabel('hrs')
plt.legend(['Y_valid','Y_valid_hat'])
plt.show()
t=np.arange(0,n_out)

plt.plot(t,Y_valid_3d[0,:],t,Y_valid_hat_lstm[0,:])
plt.ylabel('kWh')
plt.xlabel('hrs')
plt.legend(['Y_valid','Y_valid_hat'])
plt.show()
Xdfemd = dfemd.copy(deep=True)
Xdfemd.rename(columns={"Value (kWh)": "t"},inplace=True)
Xdfemd.drop('Datetime',axis=1,inplace=True)

n_in = 168 # number of inputs

Xdfemd.head()
for i in range(1,n_in):
    Xdfemd.insert(0, 'IMF1-%sh'%i, Xdfemd['IMF1'].shift(i), True)  
    Xdfemd.insert(0, 'IMF2-%sh'%i, Xdfemd['IMF2'].shift(i), True)    
    Xdfemd.insert(0, 'IMF3-%sh'%i, Xdfemd['IMF3'].shift(i), True)    
    Xdfemd.insert(0, 'IMF9-%sh'%i, Xdfemd['IMF9'].shift(i), True)    
    Xdfemd.insert(0, 't-%sh'%i,    Xdfemd['t'].shift(i),    True)    

    
Xdfemd.head()
Ydfemd = df.copy(deep=True)
Ydfemd.rename(columns={"Value (kWh)": "t"},inplace=True)

Ydfemd.drop('Datetime',axis=1,inplace=True)
n_out = 24  # number of outputs

for i in range(1,n_out+1):
    Ydfemd['t+%sh'%i]=Ydfemd['t'].shift(-i)
    
Ydfemd.drop('t',axis=1,inplace=True)


Xdfemd = Xdfemd[n_in-1 : -n_out]
Ydfemd = Ydfemd[n_in-1 : -n_out]
print('Xdfemd.shape: ',Xdfemd.shape)
print('Ydfemd.shape: ',Ydfemd.shape)
split2 = 0.6
L = Xdfemd.shape[0]
i_split = int(L*split2)


Xdfemd_train = Xdfemd[0:i_split]
Xdfemd_valid = Xdfemd[i_split:]

Ydfemd_train = Ydfemd[0:i_split]
Ydfemd_valid = Ydfemd[i_split:]

print('Xdfemd_train.shape: ',Xdfemd_train.shape)
print('Xdfemd_valid.shape: ',Xdfemd_valid.shape)
print('Ydfemd_train.shape: ',Ydfemd_train.shape)
print('Ydfemd_valid.shape: ',Ydfemd_valid.shape)
Xemd_train_3d = Xdfemd_train.values.reshape(Xdfemd_train.shape[0],Xdfemd_train.shape[1],1)
Xemd_valid_3d = Xdfemd_valid.values.reshape(Xdfemd_valid.shape[0],Xdfemd_valid.shape[1],1)

Yemd_train_3d = Ydfemd_train.values.reshape(Ydfemd_train.shape[0],Ydfemd_train.shape[1],1)
Yemd_valid_3d = Ydfemd_valid.values.reshape(Ydfemd_valid.shape[0],Ydfemd_valid.shape[1],1)

print('X_train_3d shape: ',Xemd_train_3d.shape)
print('X_valid_3d shape: ',Xemd_valid_3d.shape)
print('Y_train_3d shape: ',Yemd_train_3d.shape)
print('Y_valid_3d shape: ',Yemd_valid_3d.shape)
print('X_train[:3,:5,0]:\n',Xemd_train_3d[:3,:5,0])
epochs = 40
batch = 256
lr = 0.0003
adam = optimizers.Adam(lr)
model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(Xemd_train_3d.shape[1], 1)))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dense(Yemd_train_3d.shape[1]))
model_cnn.compile(loss='mse', optimizer=adam)
model_cnn.summary()
cnn_history = model_cnn.fit(Xemd_train_3d, Yemd_train_3d, validation_data=(Xemd_valid_3d, Yemd_valid_3d), epochs=epochs, verbose=2)
cnn_train_pred = model_cnn.predict(Xemd_train_3d)
cnn_valid_pred = model_cnn.predict(Xemd_valid_3d)
print('Ydf_train shape',Ydfemd_train.shape)
print('cnn_train_pred shape',cnn_train_pred.shape)
print('Train rmse: {:.3f}'.format(np.sqrt(mean_squared_error(Ydfemd_train.values, cnn_train_pred))))
print('Validation rmse: {:.3f}'.format(np.sqrt(mean_squared_error(Ydfemd_valid.values, cnn_valid_pred))))

Yemd_valid_hat_cnn = model_cnn.predict(Xemd_valid_3d)
print('Y_valid_hat_cnn.shape:',Yemd_valid_hat_cnn.shape)
k=120
t=np.arange(0,k)

plt.plot(t,Yemd_valid_3d[0:k,-1],t,Yemd_valid_hat_cnn[0:k,-1])
plt.ylabel('kWh')
plt.xlabel('hrs')
plt.legend(['Y_valid','Y_valid_hat'])
plt.show()
t=np.arange(0,n_out)

plt.plot(t,Yemd_valid_3d[0,:],t,Yemd_valid_hat_cnn[0,:])
plt.ylabel('kWh')
plt.xlabel('hrs')
plt.legend(['Y_valid','Y_valid_hat'])
plt.show()
hotel = pd.read_csv('../input/hotel-load-and-solar/hotel_load_and_solar_2016-05-19_2020-09-21.csv', parse_dates=['Datetime'])
hotel.head()