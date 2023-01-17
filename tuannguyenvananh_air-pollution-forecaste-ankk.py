import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tensorflow.python.keras.layers import Dense, LSTM , Dropout, CuDNNLSTM
from tensorflow.python.keras import Sequential
from math import sqrt
TIME_STEPS = 4
NUM_FEATURES = 4
TOTAL_FEATURES = TIME_STEPS * NUM_FEATURES

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#dataset preprocessing
#datset loading dropping na values

dataset= pd.read_csv('../input/beijing-pm25-data-data-set/PRSA_data_2010.1.1-2014.12.31.csv');
dataset=dataset.dropna();
dataset=dataset.drop('No',axis=1);
dataset=dataset.drop('year',axis=1);
dataset=dataset.drop('month',axis=1);
dataset=dataset.drop('day',axis=1);
dataset=dataset.drop('hour',axis=1);
dataset.head();

values=dataset.values;
foo = values.copy()
dataset.head()
# specify columns to plot
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
# plotting each column
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
values[:, 4]
values
foo = values.copy()
values.T
corr = np.corrcoef(values.T)
print(corr)
corr = np.corrcoef(values.T) # tìm độ tương quan của ma tran chuyen vi
_mask = np.zeros_like(corr) # tao mang 0 co cung shape voi mang ban dau
_mask[np.triu_indices_from(_mask)] = 1

np.triu_indices_from(_mask) # tra ve index cua tam giac tren
np.tril_indices_from(_mask) # tra ve index cua tam giac duoi

# _mask[np.tril_indices_from(_mask)] = True # danh dau
# d = np.tril(_mask) # trả về tuple, 2 mảng array
# print(d[0])

with sns.axes_style("white"):
    f, ax = pyplot.subplots(figsize=(7, 5))
    ax = sns.heatmap(corr, mask=_mask,annot=True, vmax=.2, square=False)
print(foo[:, 0])
values = np.delete(values, [1,3,6,7], 1) # tra ve mang moi, 
    # xoa index duoc chon [1, 3, 6, 7], chua lai nhung cot con lai
print(values)
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1] 
    # neu kieu du lieu la list -> 1 cot else thi la shape[1] cua data -> so cot
#     print(data.shape)
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
#     print(df.shift(3))
#     print(df.shift(2))
#     print(df.shift(1))

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
# frame as supervised learning
reframed = series_to_supervised(values, TIME_STEPS, 1)
#drop columns we don't want to predict
rem_cols = list(range(reframed.shape[1]-NUM_FEATURES+1,reframed.shape[1]))
reframed.drop(reframed.columns[rem_cols], axis=1, inplace=True)
print(reframed.head())
assert reframed.shape[1] == NUM_FEATURES * TIME_STEPS + 1
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(reframed)
# split into train and test sets
values = scaled
n_train_hours = 365 * 24*4
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :TOTAL_FEATURES], train[:, TOTAL_FEATURES]
test_X, test_y = test[:, :TOTAL_FEATURES], test[:, TOTAL_FEATURES]
train_X = train_X.reshape((train_X.shape[0], TIME_STEPS, NUM_FEATURES))
test_X = test_X.reshape((test_X.shape[0], TIME_STEPS, NUM_FEATURES))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
train_X[0]
if tf.test.is_gpu_available():
    lstm_layer = CuDNNLSTM
else:
    import functools
    lstm_layer = functools.partial(
            LSTM, recurrent_activation='sigmoid')
# design network
model = Sequential()
model.add(LSTM(100, return_sequences = True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.3))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(1,activation='linear'))

model.compile(loss='mse', optimizer='adam')
model.summary()
# fit network
earlystop_callback = tf.keras.callbacks.EarlyStopping(
  monitor='val_loss',
  patience=10)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('air_poll_lstm_3_ts.h5', monitor='val_loss', verbose=1, save_best_only=True)
history = model.fit(train_X, train_y, epochs=50, callbacks=[model_checkpoint, earlystop_callback],
                    batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
# make a prediction
yhat = model.predict(test_X)
m_test_X = test_X.reshape((test_X.shape[0], TOTAL_FEATURES))
print(yhat.shape, m_test_X.shape)
# invert scaling for forecast
inv_yhat = concatenate((yhat, m_test_X[:, :]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, m_test_X[:, :]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE and MAE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
mae = (mean_absolute_error(inv_y, inv_yhat))
print('Test MAE: %.3f' % mae)
print('Actual :', inv_y)
print('Predicted:', inv_yhat)
# plot history
pyplot.plot(inv_y, label='Actual')
pyplot.plot(inv_yhat, label='Predicted')
pyplot.legend()
pyplot.show()