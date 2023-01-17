import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats.mstats import hmean 
# N = length of last name = 8
# S = (8 % 5) + 1 = 4

N_SAMPLES = 10000
N_TEST = 1000

# Lengths up to S + 5
MAX_TIMESTEPS = 9
MASK_VALUE = 1

train_X = np.random.uniform(size=(N_SAMPLES, MAX_TIMESTEPS, 1))
train_L = np.random.randint(2, MAX_TIMESTEPS, N_SAMPLES)

test_X = np.random.uniform(size=(N_TEST, MAX_TIMESTEPS, 1))
test_L = np.random.randint(2, MAX_TIMESTEPS, N_TEST)
for i in range(N_SAMPLES):
    train_X[i, train_L[i]:] = MASK_VALUE
for i in range(N_TEST):
    test_X[i, test_L[i]:] = MASK_VALUE
train_y = hmean(train_X, axis=1)
test_y = hmean(test_X, axis=1)
input_ = tf.keras.Input(shape=(None, 1))
masked = tf.keras.layers.Masking(MASK_VALUE)(input_)
lstm1 = tf.keras.layers.LSTM(32, return_sequences=True)(masked)
lstm2 = tf.keras.layers.LSTM(32)(lstm1)
output = tf.keras.layers.Dense(1)(lstm2)

model = tf.keras.Model(inputs=input_, outputs=output)
model.summary()
model.compile('adam', 'mse')
hist = model.fit(train_X, train_y, epochs=8)
prediction = model.predict(test_X)
prediction[:8]
test_y[:8]
mean_squared_error(test_y, prediction)
mean_absolute_error(test_y, prediction)
np.percentile(test_y, np.arange(0, 100, 10)) 