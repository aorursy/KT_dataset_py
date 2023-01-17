

from keras.models import Sequential

from keras.layers import *

from keras.utils import np_utils

import pandas as pd

import numpy as np

from keras.callbacks import EarlyStopping , ModelCheckpoint

from keras.models import load_model

from keras.optimizers import Adam

import matplotlib.pyplot as plt
X_train = pd.read_csv("../input/stock-prediction/Train/x_train.csv").values

yt = pd.read_csv("../input/stock-prediction/Train/y_train.csv").values


from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

X_train = sc.fit_transform(X_train)
xt = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
x_test = pd.read_csv("../input/stock-prediction/Test/x_test.csv").values

x_test = sc.transform(x_test)


from keras.models import Sequential

from keras.layers import *

model=Sequential()

model.add(LSTM(32,input_shape=(None,1),return_sequences=True))

model.add(Dropout(0.5))

model.add(LSTM(32,return_sequences=True))

model.add(Dropout(0.5))

model.add(LSTM(32))

model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer='adam',loss='mean_squared_error',metrics=['acc'])

model.summary()
xt = np.array(xt)

xt = np.reshape(xt, (xt.shape[0], 1, xt.shape[1]))
from keras.callbacks import ModelCheckpoint

modelcheckpoint = ModelCheckpoint("best_model.h5",save_best_only=True)

hist = model.fit(xt,yt, epochs = 50, batch_size = 32, validation_split=0.2,callbacks=[modelcheckpoint])
x_test=x_test.reshape((x_test.shape[0],x_test.shape[1],1))

pred=model.predict(x_test)
print(pred)
df = pd.DataFrame(pred, columns = ["High"])



df.to_csv("output.cav", index = False)