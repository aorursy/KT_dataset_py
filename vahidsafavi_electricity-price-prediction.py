
import numpy as np 
import pandas as pd 
import os
import sys
import seaborn as sns

import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/re-fixed-multivariate-timeseries/re_fixed_multivariate_timeseires.csv')
df.drop(df.loc[df['datetime'].duplicated()].index, inplace = True)
df['datetime'] = pd.to_datetime(df['datetime'])

df.set_index(df['datetime'], inplace = True)
df.drop(columns = [ 'datetime'], inplace = True)
data = df.values
#df.info()
#df.dtypes
#df.shape
df.head()
# To plot pretty figures

%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# plot each column

i = 1
groups = [0, 1, 2, 3, 4, 5]
plt.figure(figsize = (25, 15))

for group in groups:
	plt.subplot(len(groups), 1, i)
	plt.plot(data[:, group])
	plt.title(df.columns[group], y=0.5, loc='right')
	i += 1
plt.show()
n_steps = 168

scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df)

data = scaled_df.values
series_reshaped =  np.array([data[i:i + (n_steps+24)].copy() for i in range(len(data) - (n_steps+24))])

#data.shape
#len(data)
series_reshaped.shape
#scaled_df.head()
X_train = series_reshaped[:43800, :n_steps]
X_valid = series_reshaped[43800:52560, :n_steps]
X_test = series_reshaped[52560:, :n_steps]

Y = np.empty((61134, n_steps, 24))
for step_ahead in range(1, 24 + 1):
    Y[..., step_ahead - 1] = series_reshaped[..., step_ahead:step_ahead + n_steps, 0]
Y_train = Y[:43800]
Y_valid = Y[43800:52560]
Y_test = Y[52560:]

print("Shape of X_train=", X_train.shape, "\n",
      "        X_valid=", X_valid.shape, "\n",
      "        X_test=", X_test.shape, "\n",
      "        Y_train=", Y_train.shape, "\n",
      "        Y_valid=", Y_valid.shape, "\n",
      "        Y_test=", Y_test.shape)
np.random.seed(42)
tf.random.set_seed(42)

# model initialization

model = keras.models.Sequential([
    
#########     Deep Multivariate Recurrant Neural Network     #############
    
#    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 6]),
#    keras.layers.SimpleRNN(20, return_sequences=True),
#    keras.layers.TimeDistributed(keras.layers.Dense(24))
    
#############     Simple Long-Short Term Memory Model     ################
    
#    keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 6]),
#    keras.layers.TimeDistributed(keras.layers.Dense(24))

#########################     Deep LSTM model     #########################
    
    keras.layers.LSTM(80, return_sequences= True, input_shape=(168,6)),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(40, return_sequences= True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(30, return_sequences= True),
    
############################     GRU model     ############################
    
#    keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid", input_shape=[None, 6]),
#    keras.layers.GRU(60, return_sequences=True),
#    keras.layers.GRU(40, return_sequences=True),
    
############################################################################
    
     keras.layers.TimeDistributed(keras.layers.Dense(24))
])    

# compile model
model.compile(loss="mape", 
              optimizer="adam",
              metrics=['accuracy'])

# model summary
model.summary()
########################     Bidirectional model     ######################

#model = tf.keras.Sequential([
  
#    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(80), input_shape=(168,6)),
#    tf.keras.layers.Dropout(0.2),
#    tf.keras.layers.Dense(24, activation='relu'),
#    tf.keras.layers.Dense(1, activation='sigmoid')
#])

#model.compile(loss="mape", 
#              optimizer="adam",
#              metrics=['accuracy'])
#model.summary()
history = model.fit(X_train, Y_train, epochs=10,
                    validation_data=(X_valid, Y_valid))
def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 20, 0, 25])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)

plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()
Y_pred = model.predict(X_test)

last_list=[]

for i in range (0, len(Y_pred)): 
    last_list.append((Y_pred[i][0][23]))
    
actual = pd.DataFrame((X_test[:,0]))
actual.rename(columns = {0:'actual'}, inplace = True)
actual['predictions'] = last_list
actual['difference'] = (actual['predictions'] - actual['actual']).abs()
actual['difference_percentage'] = ((actual['difference'])/(actual['actual']))*100
plt.plot(actual['actual'])
plt.plot(actual['predictions'])

plt.show()
inv_y=actual['actual']
inv_yhat=actual['predictions']

aa=[x for x in range(200)]
plt.plot(aa, inv_y[:200], marker='.', label="actual")
plt.plot(aa, inv_yhat[:200], 'r', label="prediction")
plt.ylabel('Global_active_power', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show()