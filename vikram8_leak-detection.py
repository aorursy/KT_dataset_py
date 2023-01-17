import numpy as np 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from keras import Sequential

from keras.layers import Dense
data_p4 = pd.read_csv("/kaggle/input/leak-data/EPA_OUT.csv")

data_p4.drop(columns=['Unnamed: 12'], inplace=True)
Y_train = data_p4[[' Q_1', ' Q_2', ' Q_3', ' Q_4', ' Q_5', ' H_1', ' H_2']]

Y_train = np.asarray(Y_train)

Y_train = Y_train.reshape(-1, 1, 7)

X_train = data_p4.drop(columns=[' Q_1', ' Q_2', ' Q_3', ' Q_4', ' Q_5', ' H_1', ' H_2'], inplace=False)

X_train = np.asarray(X_train)

X_train = X_train.reshape(-1, 1, 5)
model = Sequential()

model.add(Dense(128, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))

model.add(Dense(64, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))

model.add(Dense(32, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))

model.add(Dense(16, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))

model.add(Dense(8, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))

model.add(Dense(7, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=32, epochs=2000)
preds = model.predict(X_train)
preds
Y_train