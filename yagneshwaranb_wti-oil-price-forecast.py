from pandas import DataFrame
import pandas as pd
from pandas import Series
from pandas import concat
from pandas import read_csv
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
series = read_csv('../input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend.csv')
# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')
df = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend.csv', parse_dates=[0], index_col=0, date_parser=parser)

df.tail()


train = df
scaler = MinMaxScaler()
scaler.fit(train)
train = scaler.transform(train)
from keras.preprocessing.sequence import TimeseriesGenerator
n_input = 52
n_features = 1
n_length=len(train)
generator = TimeseriesGenerator(train, train, length=n_input, batch_size=1)
print(len(train))

import keras

model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')
history = model.fit_generator(generator,epochs=1,verbose=1)

import numpy as np
pred_list = []
batch = train[-n_input:].reshape((1, n_input, n_features))
for i in range(n_input):   
    pred_list.append(model.predict(batch)[0]) 
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
from pandas import DateOffset
import datetime as dt
Date1 = pd.date_range('2020-07-07', periods=52, freq='D')

future_dates = pd.DataFrame(Date1,columns=df.columns)
df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),index=Date1, columns=['Prediction'])
df_proj = pd.concat([df,df_predict], axis=1)

df_predict.to_csv('C:\\Users\\Owner\\Desktop\\WTI-oil-predict.csv')
df_predict.head(46)
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from matplotlib import pyplot as plt
plot_data = [
    go.Scatter(
        x=df_proj.index,
        y=df_proj['Price'],
        name='actual'
    ),
    go.Scatter(
        x=df_proj.index,
        y=df_proj['Prediction'],
        name='prediction'
    )
]
plot_layout = go.Layout(
        title='WTI Oil price'
    )
plt.plot(df_proj)
plt.show()