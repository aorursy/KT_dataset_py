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
import pandas as pd

data = pd.read_csv('/kaggle/input/austin-weather/austin_weather.csv')

data_t = data[['TempAvgF']]

data_t.index = pd.to_datetime(data[['Date']].stack(), format='%Y%m%d', errors='ignore')#datetime merge doar pe series si .stack face series

trend = np.linspace(0, len(data_t)-1, 50, dtype='int64')

data_t.head()
import matplotlib.pyplot as plt



plt.figure(figsize=(20, 8))

data_t.plot()

plt.title('Time series')

plt.xlabel('data')

plt.ylabel('temperature')

plt.show()
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Dense, Dropout

from sklearn.preprocessing import MinMaxScaler



copy_data = data_t

scaler = MinMaxScaler(feature_range=(-1, 1))



#vom folosi 20 de zile ca sa prezicem a 21 zi pentru setul nostru de date

#de asemenea separam data ca sa facem niste predictii ca lumea

data_antrenare = data_t.iloc[:1000]

data_test = data_t.iloc[1000:]





def make_data(data_frame, history):

    sequences = []

    sequ_pred = []

    values = data_frame['TempAvgF'].values

    for i in range(len(values)-history-1):

        sequences.append(values[i:i+history])

        sequ_pred.append(values[i+history+1])

    return np.array(sequences), np.array(sequ_pred)



hist_size = 20

train_x, train_y = make_data(data_antrenare, hist_size)

test_x, test_y = make_data(data_test, hist_size)





for i in range(len(train_x)):

    print(train_x[i], train_y[i])

 #scaling training data and reshaping for input

test_x = scaler.fit_transform(test_x)

train_x = scaler.fit_transform(train_x)



train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
#building model

model = Sequential()

model.add(LSTM(64, activation='relu', input_shape=(20, 1)))

model.add(Dense(32))

model.add(Dropout(0.25))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
history = model.fit(train_x, train_y, epochs=200, verbose=0, validation_data=(test_x, test_y), batch_size=20)
print('Loss ', history.history['loss'][199])
predictions = model.predict(test_x)
plt.figure(figsize=(20, 9))

plt.plot(predictions, color='red', linewidth=3)

plt.plot(test_y, color='blue')

plt.legend(('Predicted', 'Actual'))

plt.show()