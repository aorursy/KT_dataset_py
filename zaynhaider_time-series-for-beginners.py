# importing necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

import keras
import tensorflow as tf

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# reading csv file
ds = pd.read_csv('/kaggle/input/daily-climate-time-series-data/DailyDelhiClimateTrain.csv')
ds
len(ds)   # length of train dataset
# reading test dataset
tds = pd.read_csv('/kaggle/input/daily-climate-time-series-data/DailyDelhiClimateTest.csv')
tds.head()
len(tds)     # length of test dataset
ds = ds.append(tds, ignore_index=True)
ds.shape
ds = ds['meantemp']
ds = np.array(ds)
ds = ds.reshape(-1,1)
ds
plt.plot(ds)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

ds = scaler.fit_transform(ds)
ds
scaler.scale_
train = ds[0:1462]
test = ds[1462:]
train.shape , test.shape
def get_data(dataset, look_back):
  datax = []
  datay = []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back), 0]
    datax.append(a)
    datay.append(dataset[i+look_back, 0])
  return np.array(datax), np.array(datay)
look_back = 1
x_train, y_train = get_data(train, look_back)
x_test, y_test = get_data(test, look_back)
x_train , y_train
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

x_train.shape , x_test.shape
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(16, activation='tanh', input_shape = (1,1)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss = 'mean_squared_error') 
model.fit(x_train, y_train, epochs = 20, batch_size=1)
y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)
y_pred
y_test = np.array(y_test)
y_test = y_test.reshape(-1,1)
y_test = scaler.inverse_transform(y_test)
plt.plot(y_test, color='blue', label = 'Actual Values')
plt.plot(y_pred, color='brown', label = 'Predicted Values')
plt.ylabel('Passengers')
plt.legend()