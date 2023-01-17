import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
data = fetch_california_housing()



x, xtest, y, ytest = train_test_split(data.data, data.target)



xtrain, xvalid, ytrain, yvalid = train_test_split(x, y)
scaler = StandardScaler()

xtrain = scaler.fit_transform(xtrain)

xvalid = scaler.fit_transform(xvalid)

xtest = scaler.fit_transform(xtest)
xtrain.shape[1:]
model = tf.keras.models.Sequential([

    tf.keras.layers.Input(shape=[8,]),

    

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(1)

])

model.summary()



model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=0.001))
history = model.fit(xtrain, ytrain, epochs = 10, validation_data=(xvalid, yvalid))
print(f"Loss = {model.evaluate(xtest, ytest)}")
preds = model.predict(xtest)
plt.scatter(xtest[:,0],ytest,color='green', label='Actual result')

plt.scatter(xtest[:,0],preds,color='red', label='Prediction')

plt.legend();