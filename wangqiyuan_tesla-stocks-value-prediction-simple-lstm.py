import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime

import math

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras import backend as K

from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

!pip install mplfinance

import mplfinance as mpf

plt.style.use('seaborn-deep')
def parse(x):

    return datetime.datetime.strptime(x, '%Y-%m-%d')



df = pd.read_csv('../input/tesla-stock-data-from-2010-to-2020/TSLA.csv',  parse_dates = True, index_col=0, date_parser=parse)

df.columns = ['Open', 'High', 'Low', 'Close', 'Adj_close', 'Volume']

df.index.name = 'Date'
print('First observation date: %s \nLast observation date: %s' % (df.index[0].date(), df.index[-1].date()))
df.head(8)
df.info()
df.describe()
print(np.var(df.Close - df.Adj_close))
df.drop(['Adj_close'], inplace = True, axis = 1)

c_names = [name for name in df.columns]
mpf.plot(df.iloc[-50:,:], type='candle', volume=True)
values = df.values



num_f = len(df.columns)



groups = [x for x in range(num_f)]



plt.figure(figsize = (12,16))



i = 1

for group in groups:

    plt.subplot(len(groups), 1, i)

    plt.plot(values[:, group])

    plt.title(df.columns[group], y=0.85, loc='center')

    i += 1

plt.show()
def series_to_supervised(data, c_names, n_in=1, n_out=1, dropnan=True):

    '''

    This function reformats the dataset the way it can be fed to the LSTM.

    '''

    n_vars = 1 if type(data) is list else data.shape[1]

   

    df = pd.DataFrame(data)

    cols, names = list(), list()

    

    # input sequence (t-n, ... t-1)

    for i in range(n_in, 0, -1):



        cols.append(df.shift(i))

        names += ['%s(t-%d)' % (n, i) for n in c_names]

    

    # forecast sequence (t, t+1, ... t+n)

    for i in range(0, n_out):

        cols.append(df.shift(-i))

        if i == 0:

            names += [('%s(t)' % n) for n in c_names]

        else:

            names += [('%s(t+%d)' % (n, i)) for n in c_names]

   

    agg = pd.concat(cols, axis=1)

    agg.columns = names

    

    if dropnan:

        agg.dropna(inplace=True)

    return agg
def root_mean_squared_error(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true)))
# ensure all data is float

values = values.astype('float32')



# normalize features

scaler = MinMaxScaler(feature_range=(0, 1))

scaled = scaler.fit_transform(values)



n_days_back = 21

n_days_future = 1

n_features = num_f



# frame as supervised learning

reframed = series_to_supervised(scaled, c_names, n_days_back, n_days_future)



print(reframed.head(5))
n_obs = n_days_back * n_features



target_idx = [reframed.columns.to_list().index(col) for col in reframed.columns[n_obs:] if 'Open' in col]





# split into train and test sets

values = reframed.values



n_train_days = 2000



train = values[:n_train_days, :]

test = values[n_train_days:, :]
train_X, train_y = train[:, :n_obs], train[:, target_idx]

test_X, test_y = test[:, :n_obs], test[:, target_idx]



# reshape input to fit the LSTM network requirements: [n_samples, window, n_features]

train_X = train_X.reshape((train_X.shape[0], n_days_back, n_features))

test_X = test_X.reshape((test_X.shape[0], n_days_back, n_features))



print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# design network

model = Sequential()

model.add(LSTM(150, input_shape=(train_X.shape[1], train_X.shape[2])))

model.add(Dense(train_y.shape[1]))



checkpoint = ModelCheckpoint('w.hdf5', monitor='val_loss', save_best_only=True)



callback_list = [checkpoint]



model.compile(optimizer = 'adam', loss = root_mean_squared_error)



t = model.fit(train_X, train_y, epochs=50,

              batch_size=32,

              validation_data=(test_X, test_y),

              verbose=1,

              callbacks = callback_list,

              shuffle=False)



model.load_weights('w.hdf5')
# plot training history

plt.figure(figsize = (9,7))

plt.title('Training Loss / Validation Loss')

plt.plot(t.history['loss'], 'tab:red', label='Training loss')

plt.plot(t.history['val_loss'], 'tab:blue', label='Validation loss')

plt.legend()

plt.show()
# make a prediction

yhat = model.predict(test_X)



# invert scaling

yhat_inv = yhat / scaler.scale_[0]

y_inv = test_y / scaler.scale_[0]



# reshape back

yhat_inv_rshp = yhat_inv.reshape((-1,1))

y_inv_rshp = y_inv.reshape((-1,1))



# calculate RMSE

rmse = math.sqrt(mean_squared_error(y_inv_rshp, yhat_inv_rshp))

print('Test set RMSE: %.2f' % rmse)
plt.figure(figsize = (18,7))

plt.title('Real data vs Predicted on the Test set')

plt.plot(y_inv_rshp, 'tab:green', label='Real data')

plt.plot(yhat_inv_rshp, 'tab:blue', label='Predicted')

plt.legend()

plt.show()
print(f'Initial set size: {df.shape[0]}')

print(f'Train set X size: {train_X.shape[0]}, train set y size: {train_y.shape[0]}')

print(f'Test set X size: {test_X.shape[0]}, test set y size: {test_y.shape[0]}')
reframed.tail(1)
last_day = reframed.iloc[-1:, -n_obs:]

last_day = last_day.values

last_day = last_day.reshape((-1, n_days_back, n_features))



t_plus_one = model.predict(last_day)

t_plus_one /= scaler.scale_[0]

print('Last observation\'s date in the dateset is %s\nLast observation\'s value is %.2f' % (df.index[-1].date(), df.iloc[-1, 0]))

print('\nPredicted observation\'s date is %s\nPredicted observation\'s value is %.2f' % (df.index[-1].date() + datetime.timedelta(days=1), t_plus_one))

print('\nReal value for %s is %.2f' % (df.index[-1].date() + datetime.timedelta(days=1), 882.96))
