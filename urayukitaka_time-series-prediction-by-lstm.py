# Basic libraries
import numpy as np
import pandas as pd

# Directry check
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Libraries
import datetime

# Visualization
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

# Timeseries analysis
from statsmodels.tsa import stattools as st
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

# LSTM
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
df = pd.read_csv("../input/international-airline-passengers/international-airline-passengers.csv", header=0)
df.head()
# Data size
print("Data size:{}".format(df.shape))
# Data info
print("Data info")
df.info()
# Null data check
print("Null data check")
df.isnull()
x = df.iloc[:-1,:]["Month"]
y = df.iloc[:-1,1]

print("Time series")
plt.figure(figsize=(25, 6))
plt.plot(x,y)
plt.xlabel("monthly")
plt.ylabel("passengers")
plt.xticks(rotation =90)
# freq⇒12, monthly
res = sm.tsa.seasonal_decompose(y, freq=12)

fig, ax = plt.subplots(4,1, figsize=(15,9))
plt.subplots_adjust(hspace=0.3)

#
ax[0].plot(res.observed, lw=.6, c='darkblue')
ax[0].set_title('observed')

#
ax[1].plot(res.trend, lw=.6, c='indianred')
ax[1].set_title('trend')

#
ax[2].plot(res.seasonal, lw=.6, c='indianred')
ax[2].set_title('seasonal')

# 
ax[3].plot(res.resid, lw=.6, c='indianred')
ax[3].set_title('residual')
train_data = df.iloc[:-13,1]
# Define parameter for optimizing
# Seasonal parameter is defined=12, so this time, it is not calculated.
max_p = 2
max_d = 1
max_q = 11
max_sp = 1
max_sd = 0
max_sq = 1

pattern = max_p*(max_d+1)*(max_q+1)*(max_sp+1)*(max_sd+1)*(max_sq+1)

modelSelection = pd.DataFrame(index=range(pattern), columns=["model", "aic"])

# Auto SARIMA selection
num = 0
for p in range(1, max_p+1):
    for d in range(0, max_d+1):
        for q in range(0, max_q+1):
            for sp in range(0, max_sp+1):
                for sd in range(0,max_sd+1):
                    for sq in range(0, max_sq+1):
                        sarima = sm.tsa.SARIMAX(
                            train_data, 
                            order=(p,d,q),
                            seasonal_order=(sp,sd,sq,12),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        ).fit()
                        modelSelection.ix[num]["model"] = "order=(" + str(p) + ","+ str(d) + ","+ str(q) + "), season=("+ str(sp) + ","+ str(sd) + "," + str(sq) + ")"
                        modelSelection.ix[num]["aic"] = sarima.aic
                        num = num + 1
# check the aic
modelSelection.sort_values(by='aic').head()
# Model fitting
sarima = sm.tsa.SARIMAX(train_data, order=(1,1,11), season_order=(1,0,1, 12))
model = sarima.fit(ic="aic")
# Checking residuals
plt.figure(figsize=(10,6))
plt.plot(model.resid)
plt.title("Residual_result_SARIMA_model")
plt.xlabel("delta_month")
plt.xticks(rotation=90)
plt.ylabel("Residual")
# Prediction and model summary
sarima_predict = model.predict(start=0, end=143)
print(model.summary())
# Plotting prediction result

plt.figure(figsize=(20,6))
plt.plot(df.iloc[:-1,0], df.iloc[:-1,1], color="gray")
plt.plot(df.iloc[:-13,0], sarima_predict[:-12], color="blue", linestyle='--')
plt.plot(df.iloc[-14:-1,0], sarima_predict[-13:], color="red", linestyle='-')
plt.title("Residual result SARIMA model")
plt.xlabel("Month")
plt.xticks(rotation=90)
plt.ylabel("Residual")
plt.legend(["Base data", "Prediction"])
# 0~143 is study data
data = np.array(df.iloc[:-1,1]).astype('float64')
data_max = max(data)
data = data / data_max
# preprocessing for LSTM, create the traning & target data
x, y = [], []

length = 12 # Seasonaly
pred_length = 12 # Last 12 month will be predicted
for i in range(len(data)-length-pred_length): # stop index=120-1=119
    x.append(data[i:i+length]) # width(12)*120 data.
    y.append(data[i+length]) # Next to x data.

train_data = np.array(x).reshape(len(x), length, 1) # data shape (120,12)⇒(120,12,1)
target = np.array(y).reshape(len(y),1) # data shape (120,)⇒(120,1)
# Model construction

# parameters
length = 12
in_out_neurons = 1
n_hidden = 300

# Model
model = Sequential()
model.add(LSTM(n_hidden,
               batch_input_shape=(None, length, in_out_neurons), # (, 12, 1)
               return_sequences=False)
         )
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))
optimizer = Adam(lr=0.0001)
# Compile
model.compile(loss="mean_squared_error", optimizer=optimizer)

# Learning
history = model.fit(train_data, # ←train_data input
                    target, batch_size=1,
                    epochs=100,
                    validation_split=0.1)
predicted = model.predict(train_data) # ←predicted value of train data.
# Futuer prediction
future_test = train_data[len(train_data)-1] # ←Last data in train_data
future_result = []
length = 12
for i in range(12): # Update data in self loop
    test_data = np.reshape(future_test, (1, length, 1)) # reshape(12,1)⇒(1,12,1), adjust to train_data
    batch_predict = model.predict(test_data) # prediction of test_data
    future_test = np.delete(future_test,0) # delete first value
    future_test = np.append(future_test, batch_predict) # add predicted value for creation 12 data.
    future_result = np.append(future_result, batch_predict) # add predicted value to result list
# base plot value
x = df.iloc[:-1,:]["Month"]
y_base = df.iloc[:-1,1:]

# predicted value
x_pred = df.iloc[:-1,:]["Month"][12:-12]
y_predict = predicted*data_max

# Future value
x_future = df.iloc[:-2,:]["Month"][-12:]
y_future = future_result*data_max

# Visualization
plt.figure(figsize=(20, 6))
plt.plot(x, y_base, color='gray')
plt.plot(x_pred, y_predict, color="blue", linestyle='--')
plt.plot(x_future, y_future, color="red")
plt.xlabel("monthly")
plt.ylabel("passengers")
plt.xticks(rotation =90)