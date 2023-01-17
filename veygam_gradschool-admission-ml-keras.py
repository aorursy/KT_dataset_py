import numpy as np
import pandas as pd

from numpy.random import seed
seed(1)


import os
print(os.listdir("../input"))


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV, Lasso, LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


random_state = 1
test_size = 0.2


#!pip install -q seaborn

from __future__ import absolute_import, division, print_function

import pathlib

import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow import set_random_seed
set_random_seed(2)

print(tf.__version__)
df = pd.read_csv('../input/Admission_Predict.csv')
#df.info()
df.describe()
#df.columns
#df.head()
#df.corr()
df.drop(columns="Serial No.",inplace= True)

y = df['Chance of Admit ']
x = df[['GRE Score', 'TOEFL Score', 'University Rating', 'CGPA', 'SOP', 'LOR ', 'Research']]
x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state=random_state, test_size=test_size)

lin = LinearRegression()
lin.fit(x_train, y_train)
target = lin.predict(x_valid)
print("Mean Squared Error: ",mean_squared_error(y_valid,target))
print('Predicted values', target[:5])
print('Correct values', y_valid[:5].values)
x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state=1)

ridgeCV = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=10)
ridgeCV.fit(x_train, y_train)
target = ridgeCV.predict(x_valid)
print("Mean Squared Error: ",mean_squared_error(y_valid,target))
print('Predicted values', target[:5])
print('Correct values', y_valid[:5].values)
ridgeCV.alpha_
x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state=1)

lassoCV = LassoCV(alphas=[0.1, 1.0, 10.0], cv=10)
lassoCV.fit(x_train, y_train)
target = lassoCV.predict(x_valid)
print("Mean Squared Error: ",mean_squared_error(y_valid,target))
print('Predicted values', target[:5])
print('Correct values', y_valid[:5].values)
lassoCV.alpha_
scores = cross_val_score(lin, x, y, cv=100, scoring='neg_mean_squared_error')
meanScoreLin = scores.mean()*-1
scores = cross_val_score(ridgeCV, x, y, cv=100, scoring='neg_mean_squared_error')
meanScoreRidge = scores.mean()*-1
scores = cross_val_score(lassoCV, x, y, cv=100, scoring='neg_mean_squared_error')
meanScoreLasso = scores.mean()*-1
print('mean MSE Linear Regression', meanScoreLin, 'mean MSE Ridge Regression', meanScoreRidge, 'mean MSE Lasso Regression', meanScoreLasso)
x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state=random_state, test_size=test_size)

Input = [('scaler', MinMaxScaler()), ('linear regression', LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(x_train, y_train)
target = pipe.predict(x_valid)

print("Mean Squared Error: ",mean_squared_error(y_valid,target))
x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state=1)#test_size=0.2, random_state=1)

Input = [('scaler', StandardScaler()), ('linear regression', LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(x_train, y_train)
target = pipe.predict(x_valid)

print("Mean Squared Error: ",mean_squared_error(y_valid,target))
x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state=random_state, test_size=test_size) #.01
def build_model():
  model = keras.Sequential([
    layers.Dense(30, activation=tf.nn.relu, input_shape=[len(x_train.keys())]),
    layers.Dense(15, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
model = build_model()
model.summary()
example_batch = x_train[:10]
example_result = model.predict(example_batch)
example_result
import matplotlib.pyplot as plt

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [chance]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,5])#[0,5]
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [chance]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,5])#[0,20]
# Display training progress by printing a single dot for each completed epoch
model = build_model()

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  x_train, y_train,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

loss, mae, mse = model.evaluate(x_train, y_train, verbose=0)

print("")
print("Training set Mean Abs Error: {:5.10f}".format(mae))
print("Training set Mean Square Error: {:5.10f}".format(mse))

loss, mae, mse = model.evaluate(x_valid, y_valid, verbose=0)

print("")
print("Testing set Mean Abs Error: {:5.10f}".format(mae))
print("Testing set Mean Square Error: {:5.10f}".format(mse))

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

plot_history(history)
test_predictions = model.predict(x_valid).flatten()

plt.scatter(y_valid, test_predictions)
plt.xlabel('True Values [chance of admission]')
plt.ylabel('Predictions [chance of admission]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

history = model.fit(x_train, y_train, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

loss, mae, mse = model.evaluate(x_train, y_train, verbose=0)

print("")
print("Training set Mean Abs Error: {:5.10f}".format(mae))
print("Training set Mean Square Error: {:5.10f}".format(mse))

loss, mae, mse = model.evaluate(x_valid, y_valid, verbose=0)

print("")
print("Testing set Mean Abs Error: {:5.10f}".format(mae))
print("Testing set Mean Square Error: {:5.10f}".format(mse))

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

plot_history(history)
test_predictions = model.predict(x_valid).flatten()

plt.scatter(y_valid, test_predictions)
plt.xlabel('True Values [chance of admission]')
plt.ylabel('Predictions [chance of admission]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])