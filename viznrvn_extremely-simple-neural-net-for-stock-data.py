import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



import tensorflow as tf
sns.set(style='darkgrid')
!ls -l ../input/Data/Stocks | grep "ibm"
time_step = 10



dataframe = pd.read_csv('../input/Data/Stocks/ibm.us.txt', sep=',')

data = dataframe['Close']

steps_per_epoch = len(dataframe)//time_step

batch_data = pd.concat([data[i:len(data)-time_step+i].reset_index().drop('index', axis=1) for i in range(time_step+1)], 

                       axis=1)

batch_data.columns = ['time {}'.format(i) for i in range(len(batch_data.columns)-1)]+['target']
batch_data.head(2)
train_idx = int(len(batch_data)*0.7)

val_idx = int(train_idx+len(batch_data)*0.2)

test_idx = int(val_idx+len(batch_data)*0.1)
X_train = batch_data.drop('target', axis=1).iloc[0:train_idx]

Y_train = batch_data['target'].iloc[0:train_idx]



X_val = batch_data.drop('target', axis=1).iloc[train_idx:val_idx]

Y_val = batch_data['target'].iloc[train_idx:val_idx]



X_test = batch_data.drop('target', axis=1).iloc[val_idx:test_idx]

Y_test = batch_data['target'].iloc[val_idx:test_idx]
print(len(X_train), len(X_val), len(X_test))
input_layer = tf.keras.layers.Input(shape=(10,))

layer1 = tf.keras.layers.Dense(units=64, activation='relu')(input_layer)

layer2 = tf.keras.layers.Dense(units=64, activation='relu')(layer1)

layer3 = tf.keras.layers.Dense(units=64, activation='relu')(layer2)

output = tf.keras.layers.Dense(units=1, activation='linear')(layer3)



model = tf.keras.models.Model(inputs=input_layer, outputs=output)
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(np.array(X_train).reshape(-1, 10), np.array(Y_train).reshape(-1, 1), 

                    epochs=30, validation_data=[X_val, Y_val], verbose=0)
sns.lineplot(x=[i for i in range(len(history.history['loss']))], y=history.history['loss'])
pred = model.predict(np.array(X_test).reshape(-1, 10))
plt.figure(figsize=(16, 6))



sns.lineplot(x=[i for i in range(len(Y_test))], y=Y_test)

sns.lineplot(x=[i for i in range(len(Y_test))], y=pred.reshape(1404,))