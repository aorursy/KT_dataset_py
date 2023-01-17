import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import date
%matplotlib inline
df = pd.read_csv('../input/fremont-bridge-hourly-bicycle-counts-by-month-october-2012-to-present.csv')
df.head()
df['Datetime'] = pd.to_datetime(df['Date'])
df = df.drop(['Date','Fremont Bridge Total' ], axis=1)
df['Hour'] = df['Datetime'].dt.hour

df['Day'] = df['Datetime'].dt.day

df['Month'] = df['Datetime'].dt.month

df['Year'] = df['Datetime'].dt.year
df['Dayname'] = df['Datetime'].dt.day_name()
df[df.isnull().any(axis=1)]
df['Fremont Bridge West Sidewalk'].fillna(df.groupby(["Dayname", "Hour"])["Fremont Bridge West Sidewalk"].transform(np.mean), inplace=True)

df['Fremont Bridge East Sidewalk'].fillna(df.groupby(["Dayname", "Hour"])["Fremont Bridge East Sidewalk"].transform(np.mean), inplace=True)
df.rename({'Fremont Bridge East Sidewalk': 'East', 'Fremont Bridge West Sidewalk': 'West'}, axis=1, inplace=True)
df.groupby('Hour').mean()['East'].plot()
df.groupby('Day').mean()['East'].plot()
df.groupby('Month').mean()['East'].plot()
df.groupby('Year').mean()['East'].plot()
values = df.values

groups = [0, 1]

i = 1

plt.figure(figsize=(10,4))

for group in groups:

    plt.subplot(len(groups), 1, i)

    plt.plot(values[:, group])

    plt.title(df.columns[group], y=0.5, loc='right')

    i += 1

plt.show()

dayname_and_hour_data = df[['West','Dayname']][df['Hour']==17]
order_day_in_week = [

                     'Monday',

                     'Tuesday',

                     'Wednesday',

                     'Thursday',

                     'Friday',

                     'Saturday',

                     'Sunday',

                     ]
plt.figure(figsize=(10,4))

sns.stripplot(x='Dayname', y='West', data=dayname_and_hour_data, order=order_day_in_week )
plt.figure(figsize=(10,4))

sns.boxplot(x='Dayname', y='West', data=dayname_and_hour_data, order=order_day_in_week )
df['East-168'] = df['East'].shift(168)

df = df.dropna()
df.corr()
df = df.drop(['Datetime','Dayname'], axis=1 )
X = df.drop('East', axis=1).values

y = df['East'].values
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.python.keras.layers.recurrent import LSTM
print(X_train.shape, y_train.shape)
model = Sequential()

model.add(LSTM(6, input_shape=(X_train.shape[1], X_train.shape[2])))

model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
history = model.fit(X_train, y_train, epochs=60, batch_size=72, validation_data=(X_test, y_test), verbose=2, shuffle=False)
losses = pd.DataFrame(history.history)
losses.plot()
from sklearn.metrics import mean_absolute_error, mean_squared_error
predictions = model.predict(X_test)
mean_absolute_error(y_test, predictions)
mean_squared_error(y_test, predictions)**(1/2)
df['East'].describe()
single_hour = df.drop('East', axis=1).iloc[20:21]

single_hour
single_hour = scaler.transform(single_hour)

single_hour = single_hour.reshape((single_hour.shape[0], 1, single_hour.shape[1]))

model.predict(single_hour)
df.iloc[20:21]