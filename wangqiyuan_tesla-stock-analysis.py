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
import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

%matplotlib inline

import plotly.graph_objects as go



import seaborn as sns

TSLA = pd.read_csv("../input/tesla-stock-data-from-2010-to-2020/TSLA.csv")
TSLA['Date'] = pd.to_datetime(TSLA['Date'])

TSLA.index = TSLA['Date']

TSLA.head(3)
Tesla=TSLA.rename(columns={'Adj Close': 'AdjClose'})

Tesla.head(3)
sns.pairplot(Tesla[["Open", "High", "Close", "Volume"]], diag_kind="kde")

fig = px.line(Tesla, x='Date', y='Volume')

fig.show()
fig = go.Figure(data=[go.Candlestick(

    x=Tesla['Date'],

    open=Tesla['Open'], high=Tesla['High'],

    low=Tesla['Low'], close=Tesla['Close'],

    increasing_line_color= 'cyan', decreasing_line_color= 'green'

)])



fig.show()
Tesla_HPY = pd.DataFrame({'Date':Tesla['Date'], 'HPY':Tesla['Close'] / Tesla['Open']-1})

Tesla_HPY.head(3)
# Telsa HPY static statement 

Tesla_HPY.describe()
# Tesla visualization 

fig = go.Figure([go.Scatter(x=Tesla_HPY['Date'], y=Tesla_HPY['HPY'])])

fig.show()
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from sklearn.preprocessing import OneHotEncoder
TeslaDF = Tesla.loc[:,['AdjClose','Volume']]

TeslaDF['HPY'] = ((Tesla['Close'] / Tesla['Open'])-1) 

TeslaDF['HL_PCT'] = (Tesla['High'] - Tesla['Low'])/Tesla['Open'] 
TeslaDF.head(3)
train_dataset = TeslaDF.sample(frac=0.8,random_state=0)

test_dataset = TeslaDF.drop(train_dataset.index)
train_stats = train_dataset.describe()

train_stats.pop("AdjClose")

train_stats = train_stats.transpose()

train_stats
train_labels = train_dataset.pop('AdjClose')

test_labels = test_dataset.pop('AdjClose')
def norm(x):

    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)

normed_test_data = norm(test_dataset)
def build_model():

    model = keras.Sequential([

      layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),

      layers.Dense(64, activation='relu'),

      layers.Dense(1)

    ])



    optimizer = tf.keras.optimizers.RMSprop(0.001)



    model.compile(loss='mse',

                  optimizer=optimizer,

                  metrics=['mae', 'mse'])

    return model
model = build_model()
model.summary()

example_batch = normed_train_data[:10]

example_result = model.predict(example_batch)

example_result
class PrintDot(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs):

        if epoch % 100 == 0: print('')

        print('.', end='')



EPOCHS = 1000



history = model.fit(

  normed_train_data, train_labels,

  epochs=EPOCHS, validation_split = 0.2, verbose=0,

  callbacks=[PrintDot()])
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)



print("test_set average mas: {:5.2f} MPG".format(mae))
test_predictions = model.predict(normed_test_data).flatten()



plt.scatter(test_labels, test_predictions)

plt.xlabel('True Values [MPG]')

plt.ylabel('Predictions [MPG]')

plt.axis('equal')

plt.axis('square')

plt.xlim([0,plt.xlim()[1]])

plt.ylim([0,plt.ylim()[1]])

_ = plt.plot([-100, 100], [-100, 100])
error = test_predictions - test_labels

plt.hist(error, bins = 25)

plt.xlabel("Prediction Error [MPG]")

_ = plt.ylabel("Count")