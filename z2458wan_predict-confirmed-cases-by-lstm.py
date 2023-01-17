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
confirmed_time_series = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv', index_col='Country/Region')
confirmed_time_series = confirmed_time_series.drop(['Province/State','Lat','Long'], 1)

series_by_country = confirmed_time_series.sum(level = 'Country/Region')
# split a univariate sequence into samples

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

    return np.array(X), np.array(y)
from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

def predict_by_lstm(country, n_steps):

    confirmed = series_by_country.loc[country]

    confirmed_cases = [[int(confirmed[i])] for i in range(len(confirmed))]

    

    scaler = MinMaxScaler()

    confirmed_cases = scaler.fit_transform(confirmed_cases)

    

    X, y = split_sequence(confirmed_cases, n_steps)

    

    # reshape from [samples, timesteps] into [samples, timesteps, features]

    n_features = 1

    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # define model

    model = Sequential()

    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))

    model.add(LSTM(50, activation='relu'))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    

    # fit model

    history = model.fit(X, y, epochs=200, validation_split=0.1, verbose=1)

    

    print(history.history.keys())

    

    # Plot training & validation loss values

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()

    new_confirmed_series = np.array([])

    new_confirmed_series = confirmed_cases.copy()

    for i in range(14):

        X_input = new_confirmed_series[-n_steps:]

        X_input = np.array(X_input)

        X_input = X_input.reshape((1, n_steps, 1))

        new_pred = model.predict(X_input, verbose=0)

        new_confirmed_series = np.concatenate((new_confirmed_series , new_pred), axis=0)

    

#     res = pd.DataFrame(data = new_confirmed_series, columns = ['Cases'])

    return scaler.inverse_transform(new_confirmed_series)
ca = predict_by_lstm('Canada', 7)
x1 = [i for i in range(len(ca))]

plt.plot(x1[:-14], ca[:-14])

plt.plot(x1[-14:], ca[-14:])

plt.title('Canada cases (predict in orange)')