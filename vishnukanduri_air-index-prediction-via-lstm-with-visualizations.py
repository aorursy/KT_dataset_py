import pandas as pd

import numpy as np

import seaborn as sns;

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras import optimizers

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
# loading the data into the dataframe

df = pd.read_csv('/kaggle/input/air-quality-of-cities-in-china/PRSA_Data_20130301-20170228/PRSA_Data_Aotizhongxin_20130301-20170228.csv')

print(df)
# viewing info about the columns

df.info();
#viewing few rows from the top

df.head()
#number of rows and columns in the dataset

print(df.shape)
#statistical information about columns

print(df.describe())
#checking how many null values are in each column

df.isnull().sum()
# dropping all the rows with NaN values

df = df.dropna()
df.isnull().sum()
#defining training and testing data

x_train = df[:24865]

y_train = x_train['PM2.5']

x_test = df[24865:31898]

y_test = x_test['PM2.5']

print(y_test)
df.loc[24865:31898].count() / df.shape[0] * 100
#Normalize training data

train_norm = x_train['PM2.5'] 

train_norm_arr = np.asarray(train_norm)

train_norm = np.reshape(train_norm_arr, (-1, 1))

scaler = MinMaxScaler(feature_range=(0, 1))

train_norm = scaler.fit_transform(train_norm)

count = 0

for i in range(len(train_norm)):

    if train_norm[i] == 0:

        count = count +1

print('Number of null values in train_norm = ', count)
#removing null values 

train_norm = train_norm[train_norm!=0]
test_norm = x_test['PM2.5']

test_norm_arr = np.asarray(test_norm)

test_norm = np.reshape(test_norm_arr, (-1, 1))

scaler = MinMaxScaler(feature_range=(0, 1))

test_norm = scaler.fit_transform(test_norm)
count = 0

for i in range(len(test_norm)):

    if test_norm[i] == 0:

        count = count + 1 

print('Number of null values in test_norm = ', count)
test_norm = test_norm[test_norm != 0]
def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)

        y.append(seq_y)

    return array(X),array(y)
n_steps = 3

X_split_train, y_split_train = split_sequence(train_norm, n_steps)

n_features = 1

X_split_train = X_split_train.reshape((X_split_train.shape[0], X_split_train.shape[1], n_features))

for i in range(1):

    print(X_split_train)
X_split_test, y_split_test = split_sequence(test_norm, n_steps)

for i in range(5):

    print(X_split_test[i], y_split_test[i])

n_features = 1

X_split_test = X_split_test.reshape((X_split_test.shape[0], X_split_test.shape[1], n_features))
# define model

model = Sequential()

model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))

model.add(Dense(1))

#sgd = optimizers.SGD(lr=0.001, decay=1e-5, momentum=1.0, nesterov=False)

sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True) #good

#keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9)

keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
# fit model

hist = model.fit(X_split_train, y_split_train, validation_data=(X_split_test, y_split_test), epochs=10, verbose = 1)
print(hist.history.keys())
yhat = model.predict(X_split_test)

for i in range(5):

    print(yhat[i])
mse = mean_squared_error(y_split_test, yhat)

print('MSE: %.5f' % mse)
plt.plot(yhat)
plt.plot(y_split_test)
_, train_acc = model.evaluate(X_split_train, y_split_train, verbose=0)

_, test_acc = model.evaluate(X_split_test, y_split_test, verbose=0)

print('Train: %.6f, Test: %.5f' % (train_acc, test_acc))
# summarize history for accuracy

plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# summarize history for loss

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
def compute(var):

    train_norm = x_train[var] 

    train_norm_arr = np.asarray(train_norm)

    train_norm = np.reshape(train_norm_arr, (-1, 1))

    scaler = MinMaxScaler(feature_range=(0, 1))

    train_norm = scaler.fit_transform(train_norm)



    test_norm = x_test[var]

    test_norm_arr = np.asarray(test_norm)

    test_norm = np.reshape(test_norm_arr, (-1, 1))

    scaler = MinMaxScaler(feature_range=(0, 1))

    test_norm = scaler.fit_transform(test_norm)



    X_split_train, y_split_train = split_sequence(train_norm, n_steps)

    X_split_train = X_split_train.reshape((X_split_train.shape[0], X_split_train.shape[1], n_features))



    X_split_test, y_split_test = split_sequence(test_norm, n_steps)

    X_split_test = X_split_test.reshape((X_split_test.shape[0], X_split_test.shape[1], n_features))



    hist = model.fit(X_split_train, y_split_train, validation_data=(X_split_test, y_split_test), epochs=10, verbose = 1)



    yhat = model.predict(X_split_test)



    mse = mean_squared_error(y_split_test, yhat)

    print(mse)

    

    plt.plot(hist.history['accuracy'])

    plt.plot(hist.history['val_accuracy'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()



    plt.plot(hist.history['loss'])

    plt.plot(hist.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()

    

# compute('PM2.5')

# compute('PM10')

# compute('SO2')

# compute('NO2')

# compute('CO')

# compute('O3')
#jointplot for PM2.5 concentration and PM10 concentration

sns.jointplot(x=df['PM2.5'], y=df['PM10'], data = df)
#finding correlation

corrmat = df.corr()

fig, ax = plt.subplots(figsize=(11,11))



#Heatmap

sns.heatmap(corrmat)
#To generate pairplots for all features.

g = sns.pairplot(df)
#density plots

df.plot(kind='density', subplots=True, layout=(4,4), sharex=False, figsize=(10,10))

plt.show()
#scatter plots

df.plot.scatter(x='PM2.5', y='PM10', c='DarkBlue')
plt.scatter(y_split_test, yhat)
df.plot.scatter(x='PM10', y='SO2', c='DarkBlue')
df.plot.scatter(x='SO2', y='NO2', c='DarkBlue')
df.plot.scatter(x='NO2', y='CO', c='DarkBlue')
df.plot.scatter(x='CO', y='O3', c='DarkBlue')
correlations = df.corr()

fig, ax = plt.subplots(figsize=(15,15))

sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f', square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})

plt.show();