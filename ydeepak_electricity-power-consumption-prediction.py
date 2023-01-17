import warnings

warnings.filterwarnings('ignore')



import pandas as pd, numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import glob

import pickle



from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error,r2_score



import keras

from keras.layers import Dense

from keras.models import Sequential

from keras.optimizers import SGD 

from keras.layers import LSTM

from keras.layers import Dropout

from keras import losses

import tensorflow as tf



import os



%matplotlib inline

plt.rcParams['figure.figsize' ]= (10,5)
DIR2008 = "../input/data/data/2008/"

DIR2009 = "../input/data/data/2009/"

DIR2010 = "../input/data/data/2010/"

dirs = [DIR2008,DIR2009,DIR2010]
def combine(path): 

    '''

    this is an auxillary method to supplement the concatenation of 

    multiple csv files into one spread across different years and

    located in different folders.

    '''

    final_df = pd.DataFrame()

    for ix in path:

        interim = glob.glob(ix+"/*")

        interim = sorted(interim, key = lambda k: int(k.split('/')[-1]))    

        try:            

            for p in interim: 

                df = pd.read_csv(p, sep = ',', header = 'infer',

                                parse_dates={'DateTime' : ['Date', 'Time'] },

                                 na_values = ['nan','?'],

                         low_memory=True, index_col = 'DateTime')

                final_df = pd.concat([final_df,df], axis  = 0 )

        except:

            continue

    print(f'Number of rows,cols in final dataframe: {final_df.shape}')

    return final_df



def series_to_supervised(data, lag=1, lead=1, dropnan=True):

    '''

        an auxillary function to prepare the dataset with given lag and lead using pandas shift function.

    '''

    n_vars = data.shape[1]

    dff = pd.DataFrame(data)

    cols, names = [],[]

    

    for i in range(lag, 0, -1):

        cols.append(dff.shift(i))

        names += [('col%d(t-%d)' % (j+1, i)) for j in range(n_vars)]



    for i in range(0, lead):

        cols.append(dff.shift(-i))

        if i == 0:

            names += [('col%d(t)' % (j+1)) for j in range(n_vars)]

        else:

            names += [('col%d(t+%d)' % (j+1, i)) for j in range(n_vars)]



    total = pd.concat(cols, axis=1)

    total.columns = names

    if dropnan:

        total.dropna(inplace=True)

    return total
%%time

paths = []

_= [paths.extend(glob.glob(f"{f}*"))for f in dirs]

paths = sorted(paths, key = lambda k: int(k.split('/')[-1]))



data = combine(paths)
data.shape 
data.isnull().sum(axis = 0 )
for ix in range(7):

#     data.iloc[:,ix] = data.iloc[:,ix].fillna(data.iloc[:,ix].mean(axis = 0))

    data.iloc[:, ix] = data.iloc[:, ix].interpolate(method = 'quadratic' , axis = 0)
data.isnull().sum(axis = 0 )
data.columns
data.describe()
plt.figure(figsize = (15,10))

plt.boxplot(data.resample('D').sum().Global_active_power)

## most of values are outside IQR
from pandas.api.types import is_numeric_dtype

def remove_outlier(df_in):

    for col_name in df_in.columns:

        q1 = df_in[col_name].quantile(0.25)

        q3 = df_in[col_name].quantile(0.75)

        iqr = q3-q1

        fence_low  = q1-1.5*iqr

        fence_high = q3+1.5*iqr

        df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]

    return df_out
t  = remove_outlier(data.resample('D').sum())
t.shape, data.resample('D').sum().shape
plt.figure(figsize = (15,10))

plt.boxplot(t.Global_active_power)
data.Global_active_power.resample('M').sum()
data.Global_active_power.resample('H').sum().plot(title="Global_active_power resampled over Hours for SUM", color = 'green')

plt.ylim(-500,+500)

plt.tight_layout();plt.show()



data.Global_active_power.resample('H').mean().plot(title="Global_active_power resampled over Hours for MEAN", color = 'blue')

plt.ylim(-50,50)

plt.tight_layout();plt.show()
data.Global_active_power.resample('D').sum().plot(title="Global_active_power resampled over Day for SUM", color = 'green')

plt.tight_layout();plt.show()



data.Global_active_power.resample('D').mean().plot(title="Global_active_power resampled over Day for MEAN", color = 'blue')

plt.tight_layout();plt.show()
data.Global_active_power.resample('W').sum().plot(title="Global_active_power resampled over Week for SUM", color = 'green')



plt.tight_layout();plt.show()



data.Global_active_power.resample('W').mean().plot(title="Global_active_power resampled over Week for MEAN", color = 'blue')

plt.tight_layout();plt.show()
data.Global_active_power.resample('M').sum().plot(title="Global_active_power resampled over Months for SUM", color = 'green')



plt.tight_layout();plt.show()



data.Global_active_power.resample('M').mean().plot(title="Global_active_power resampled over Months for MEAN", color = 'blue')

plt.tight_layout();plt.show()
data.Global_active_power.resample('Q').sum().plot(title="Global_active_power resampled over Quarter for SUM", color = 'green')



plt.tight_layout();plt.show()



data.Global_active_power.resample('Q').mean().plot(title="Global_active_power resampled over Quarter for MEAN", color = 'blue')

plt.tight_layout();plt.show()
temp = data.Global_reactive_power.resample('D').agg(['mean', 'std'])

temp.plot(subplots = True, title='Global_reactive_power resampled over day', color='green')

plt.tight_layout();plt.show()
temp1 = data.Global_intensity.resample('D').agg(['mean', 'std'])

temp1.plot(subplots = True, title='Global_intensity resampled over day',color='orange')

plt.tight_layout();plt.show()
data.Voltage.resample('M').mean().plot(kind='barh',color='orange')
data.Voltage.resample('Q').mean().plot(kind='barh',color='orange')

plt.xticks(rotation=75);plt.ylabel('DateTime');plt.xlabel('Voltage')

plt.title('Voltage resampled over Quarter for mean')

plt.show()
data.Sub_metering_1.resample('M').mean().plot(kind='bar', color='blue')

plt.xticks(rotation=60);plt.ylabel('Sub_metering_1')

plt.title('Sub_metering_1 resampled over Quarter for Mean')

plt.tight_layout();plt.show()



data.Sub_metering_2.resample('M').mean().plot(kind='bar', color='green')

plt.xticks(rotation=60);plt.ylabel('Sub_metering_2')

plt.title('Sub_metering_2 resampled over Quarter for Mean')

plt.tight_layout();plt.show()



data.Sub_metering_3.resample('M').mean().plot(kind='bar', color='red')

plt.xticks(rotation=60);plt.ylabel('Sub_metering_3')

plt.title('Sub_metering_3 resampled over Quarter for Mean')

plt.tight_layout();plt.show()
val = data.resample('D').mean().values

ix = 1

plt.figure(figsize = (10,10))

for col in range(7):

    plt.subplot(7, 1, ix)

    plt.plot(val[:, col], color='orange')

    plt.title(data.columns[col], loc='center')

    ix += 1

    plt.tight_layout()

plt.show()
data.Global_active_power.resample('W').mean().plot(color='blue', legend=True)

data.Global_reactive_power.resample('W').mean().plot(color='orange', legend=True)

data.Global_intensity.resample('W').mean().plot(color='red', legend=True)

data.Sub_metering_1.resample('W').mean().plot(color='green', legend=True)

data.Sub_metering_2.resample('W').mean().plot(color='violet', legend=True)

data.Sub_metering_3.resample('W').mean().plot(color='cyan', legend=True)





plt.tight_layout();plt.show()
corr_data=data.resample('W').sum().pct_change()
corr_data.shape
sns.jointplot(x='Global_active_power',y='Global_intensity',data=corr_data.iloc[1:,:])

plt.tight_layout();plt.show()
sns.jointplot(x='Global_active_power',y='Voltage',data=corr_data.iloc[1:,:])

plt.tight_layout();plt.show()
plt.matshow(data.corr(method='pearson'), vmax = 1, vmin = -1, cmap = 'PuBuGn')

plt.colorbar(); plt.title('Original Data without resampling', y  = -0.23)

plt.tight_layout();plt.show()



plt.matshow(data.resample('W').sum().corr(method='pearson'), vmax = 1, vmin = -1, cmap = 'PuBuGn')

plt.colorbar(); plt.title('Data with resampling over Weeks',y = -0.23)

plt.tight_layout();plt.show()
data_resample = data.resample('D').mean() 

data_resample.shape
%%time



values = data_resample.values

scaler = MinMaxScaler(feature_range=(0, 1))

scaled = scaler.fit_transform(values)



train_sup = series_to_supervised(scaled, 1, 1)
train_sup.drop(train_sup.columns[[8,9,10,11,12,13]], axis=1, inplace=True)

print(train_sup.head())
print(data_resample.shape, train_sup.shape)
values = train_sup.values



n_train_time = 365*2



train = values[:n_train_time, :]

test = values[n_train_time:, :]



# split into input and outputs

train_X, train_y = train[:, :-1], train[:, -1]

test_X, test_y = test[:, :-1], test[:, -1]



# reshape input to be 3D [samples, timesteps, features]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 
# with open('train_X.txt','wb') as f:

#     pickle.dump(train_X,f)



# with open('train_y.txt','wb') as f:

#     pickle.dump(train_y,f)



# with open('test_X.txt','wb') as f:

#     pickle.dump(test_X,f)



# with open('test_y.txt','wb') as f:

#     pickle.dump(test_y,f)
# with open('train_X.txt','rb') as f:

#     train_X = pickle.load(f)

# with open('train_y.txt','rb') as f:

#     train_y = pickle.load(f)

# with open('test_X.txt','rb') as f:

#     test_X = pickle.load(f)

# with open('test_y.txt','rb') as f:

#     test_y = pickle.load(f)

# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)     
print(tf.__version__)
model = Sequential()                               

# input (timesteps, #features)

model.add(LSTM(160, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))

model.add(Dropout(0.25))

model.add(LSTM(80))

model.add(Dropout(0.35))

model.add(Dense(1))

model.compile(loss=losses.mean_squared_error, optimizer='adam')
history = model.fit(train_X, train_y, epochs=25, batch_size=60, 

                    validation_data=(test_X, test_y), verbose=2, shuffle=False)
# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()

yhat = model.predict(test_X)

test_X = test_X.reshape((test_X.shape[0], 7))



# invert scaling for forecast

inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)

inv_yhat = scaler.inverse_transform(inv_yhat)

inv_yhat = inv_yhat[:,0]



# invert scaling for actual

test_y = test_y.reshape((len(test_y), 1))

inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)

inv_y = scaler.inverse_transform(inv_y)

inv_y = inv_y[:,0]



# calculate RMSE

rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))

print('Test RMSE: %.3f' % rmse)
newtest = remove_outlier(pd.DataFrame(test))

newtest = newtest.values

newtest.shape
newtest_X, newtest_y = newtest[:, :-1], newtest[:, -1]
newtest_X = newtest_X.reshape((newtest_X.shape[0], 1, newtest_X.shape[1]))
yhat = model.predict(newtest_X)

newtest_X = newtest_X.reshape((newtest_X.shape[0], 7))



# invert scaling for forecast

inv_yhat = np.concatenate((yhat, newtest_X[:, -6:]), axis=1)

inv_yhat = scaler.inverse_transform(inv_yhat)

inv_yhat = inv_yhat[:,0]



# invert scaling for actual

newtest_y = newtest_y.reshape((len(newtest_y), 1))

inv_y = np.concatenate((newtest_y, newtest_X[:, -6:]), axis=1)

inv_y = scaler.inverse_transform(inv_y)

inv_y = inv_y[:,0]



# calculate RMSE

rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))

print('Test RMSE: %.3f' % rmse)
data_resample = remove_outlier(data_resample)
data_resample.shape ## loss of 23 rows
%%time



values = data_resample.values

scaler = MinMaxScaler(feature_range=(0, 1))

scaled = scaler.fit_transform(values)



train_sup = series_to_supervised(scaled, 1, 1)
train_sup.drop(train_sup.columns[[8,9,10,11,12,13]], axis=1, inplace=True)

print(train_sup.head())
values = train_sup.values



n_train_time = 365*2



train = values[:n_train_time, :]

test = values[n_train_time:, :]



# split into input and outputs

train_X, train_y = train[:, :-1], train[:, -1]

test_X, test_y = test[:, :-1], test[:, -1]



# reshape input to be 3D [samples, timesteps, features]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 
model = Sequential()                               

# input (timesteps, #features)

model.add(LSTM(160, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))

model.add(Dropout(0.15))

model.add(LSTM(80))

model.add(Dropout(0.15))

model.add(Dense(1))

model.compile(loss=losses.mean_squared_error, optimizer='adam')
history = model.fit(train_X, train_y, epochs=50, batch_size=80,

                    validation_data=(test_X, test_y),verbose=2, shuffle=False)
# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()
yhat = model.predict(test_X)

test_X = test_X.reshape((test_X.shape[0], 7))



# invert scaling for forecast

inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)

inv_yhat = scaler.inverse_transform(inv_yhat)

inv_yhat = inv_yhat[:,0]



# invert scaling for actual

test_y = test_y.reshape((len(test_y), 1))

inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)

inv_y = scaler.inverse_transform(inv_y)

inv_y = inv_y[:,0]



# calculate RMSE

rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))

print('Test RMSE: %.3f' % rmse)
predictions = []

future = 7 # predict for #days ahead



## we require last 7 days of actual test data to start predicting for next 7 days ahead. 

## hence test_X[-7,:]



for ix in range(future,0,-1):

    row = test_X[-ix,:]

    row = row.reshape(1,1,7)

    out = model.predict(row)

    row = row.reshape((1,7))

    

    inv_yhat = np.concatenate((out, row[:, -6:]), axis=1)

    inv_yhat = scaler.inverse_transform(inv_yhat)

    inv_yhat = inv_yhat[:,0]



    predictions.append(inv_yhat[0])
predictions

