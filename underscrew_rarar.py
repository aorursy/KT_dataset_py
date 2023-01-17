# Field Description



# C/A,UNIT,SCP,DATE1,TIME1,DESC1,ENTRIES1,EXITS1,DATE2,TIME2,DESC2,ENTRIES2,EXITS2,DATE3,TIME3,DESC3,ENTRIES3,EXITS3,DATE4,TIME4,DESC4,ENTRIES4,EXITS4,DATE5,TIME5,DESC5,ENTRIES5,EXITS5,DATE6,TIME6,DESC6,ENTRIES6,EXITS6,DATE7,TIME7,DESC7,ENTRIES7,EXITS7,DATE8,TIME8,DESC8,ENTRIES8,EXITS8





# C/A = Control Area (A002)

# UNIT = Remote Unit for a station (R051)

# SCP = Subunit Channel Position represents an specific address for a device (02-00-00)

# DATEn = Represents the date (MM-DD-YY)

# TIMEn = Represents the time (hh:mm:ss) for a scheduled audit event

# DEScn = Represent the "REGULAR" scheduled audit event (occurs every 4 hours)

# ENTRIESn = The comulative entry register value for a device

# EXISTn = The cumulative exit register value for a device







# Example:

# The data below shows the entry/exit register values for one turnstile at control area (A002) from 03/21/10 at 00:00 hours to 03/28/10 at 20:00 hours





# A002,R051,02-00-00,03-21-10,00:00:00,REGULAR,002670738,000917107,03-21-10,04:00:00,REGULAR,002670738,000917107,03-21-10,08:00:00,REGULAR,002670746,000917117,03-21-10,12:00:00,REGULAR,002670790,000917166,03-21-10,16:00:00,REGULAR,002670932,000917204,03-21-10,20:00:00,REGULAR,002671164,000917230,03-22-10,00:00:00,REGULAR,002671181,000917231,03-22-10,04:00:00,REGULAR,002671181,000917231

# A002,R051,02-00-00,03-22-10,08:00:00,REGULAR,002671220,000917324,03-22-10,12:00:00,REGULAR,002671364,000917640,03-22-10,16:00:00,REGULAR,002671651,000917719,03-22-10,20:00:00,REGULAR,002672430,000917789,03-23-10,00:00:00,REGULAR,002672473,000917795,03-23-10,04:00:00,REGULAR,002672474,000917795,03-23-10,08:00:00,REGULAR,002672516,000917876,03-23-10,12:00:00,REGULAR,002672652,000917934

# A002,R051,02-00-00,03-23-10,16:00:00,REGULAR,002672879,000917996,03-23-10,20:00:00,REGULAR,002673636,000918073,03-24-10,00:00:00,REGULAR,002673683,000918079,03-24-10,04:00:00,REGULAR,002673683,000918079,03-24-10,08:00:00,REGULAR,002673722,000918171,03-24-10,12:00:00,REGULAR,002673876,000918514,03-24-10,16:00:00,REGULAR,002674221,000918594,03-24-10,20:00:00,REGULAR,002675082,000918671

# A002,R051,02-00-00,03-25-10,00:00:00,REGULAR,002675153,000918675,03-25-10,04:00:00,REGULAR,002675153,000918675,03-25-10,08:00:00,REGULAR,002675190,000918752,03-25-10,12:00:00,REGULAR,002675345,000919053,03-25-10,16:00:00,REGULAR,002675676,000919118,03-25-10,20:00:00,REGULAR,002676557,000919179,03-26-10,00:00:00,REGULAR,002676688,000919207,03-26-10,04:00:00,REGULAR,002676694,000919208

# A002,R051,02-00-00,03-26-10,08:00:00,REGULAR,002676735,000919287,03-26-10,12:00:00,REGULAR,002676887,000919607,03-26-10,16:00:00,REGULAR,002677213,000919680,03-26-10,20:00:00,REGULAR,002678039,000919743,03-27-10,00:00:00,REGULAR,002678144,000919756,03-27-10,04:00:00,REGULAR,002678145,000919756,03-27-10,08:00:00,REGULAR,002678155,000919777,03-27-10,12:00:00,REGULAR,002678247,000919859

# A002,R051,02-00-00,03-27-10,16:00:00,REGULAR,002678531,000919908,03-27-10,20:00:00,REGULAR,002678892,000919964,03-28-10,00:00:00,REGULAR,002678929,000919966,03-28-10,04:00:00,REGULAR,002678929,000919966,03-28-10,08:00:00,REGULAR,002678935,000919982,03-28-10,12:00:00,REGULAR,002679003,000920006,03-28-10,16:00:00,REGULAR,002679231,000920059,03-28-10,20:00:00,REGULAR,002679475,000920098
import numpy as np

from numpy import newaxis

import pandas as pd 

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from keras.preprocessing.sequence import TimeseriesGenerator

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.normalization import BatchNormalization

from keras.layers.recurrent import *

from keras.models import Sequential

from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt

from keras.callbacks import EarlyStopping

from keras import optimizers



import os

print(os.listdir("../input"))
df = pd.read_csv("../input/dados_desafio/2017.csv.gz",parse_dates=[0])
df.head()
df['turnstile'] = df['ca'] + '-' + df['unit'] + '-' + df['scp'] # por catraca

df = df.sort_values(['station','turnstile','time']).reset_index(drop=True)

df = df[df['desc']=='REGULAR'] # filtra somente as catracas normais



# aplica um delta no registro da catraca para obter o volume por periodo

def subtract_prev(rows):

    if len(rows) != 2:

        return np.nan

    prev, curr = rows

    return curr - prev if prev <= curr else curr

subtract_prev_rolling = lambda ser: ser.rolling(window=2).apply(subtract_prev)

df_grouped_turnstile = df.groupby(['turnstile','station'])

df['exits_per_interval']   = df_grouped_turnstile['exits'].apply(subtract_prev_rolling)

df['entries_per_interval'] = df_grouped_turnstile['entries'].apply(subtract_prev_rolling)



df.drop(columns=['entries','exits','desc'],inplace=True) # dropa colunas agora inuteis

        



#granularidades

df['minute'] = df['time'].dt.minute

df['hour'] = df['time'].dt.hour

df['month'] = df['time'].dt.month

df['year'] = df['time'].dt.year

df['dayofweek'] = df['time'].dt.dayofweek

df['dayofyear'] = df['time'].dt.dayofyear

df['weekofyear'] = df['time'].dt.weekofyear
# Histograma do volume de entrada/saida

samplesize = 1000000

histparams = { 'bins':20, 'range':(0,1000), 'figsize':(25, 8) }

df_samples = df.sample(n=samplesize).loc[:,['exits_per_interval','entries_per_interval']]

df_samples.hist(**histparams)
# Exitem numeros extremos: Filtro de limite

cap = 1000

for col in ['entries_per_interval', 'exits_per_interval']:

    df.loc[df[col] > cap, col] = np.nan
df['busyness_per_interval'] = df.entries_per_interval + df.exits_per_interval



by_hour = df.groupby('hour')



plt.figure(figsize=(25, 8))

by_hour['entries_per_interval'].sum().plot()

by_hour['exits_per_interval'].sum().plot()

by_hour['busyness_per_interval'].sum().plot()

plt.legend()
plt.figure(figsize=(25, 8))

df.groupby('month')['busyness_per_interval'].sum().plot()
df.head()
df.groupby('station')['busyness_per_interval'].sum().count()
import plotly.plotly as py

import plotly.offline

import plotly.graph_objs as go

plotly.offline.init_notebook_mode(connected=True)





y = df.groupby('station')['busyness_per_interval'].sum().sort_values(ascending=False).head(10)

trace = go.Pie(labels=y.index, values=y.values,title='Top 10 mais movimentadas em 2017')

plotly.offline.iplot([trace], filename='basic_pie_chart')
def transform_single_column(scaler_transform,series,orig_size):

    array = np.zeros(shape=(len(series), orig_size) )

    array[:,0] = series.reshape(-1)

    transformed = scaler_transform(array)

    return transformed[:,0]
# Cria as colunas para a predicao da estacao mais ativa do metro

# Sumarizado por dia do mes, tendo como target a media diaria



col_to_predict = 'busyness_per_interval_mean'



# agg_dict = {

#             'busyness_per_interval':['sum','mean','std'],

#             'dayofweek':'first',

#             'weekofyear':'first',

#             'month':'first'

#            }



agg_dict = {

            'busyness_per_interval':['sum','mean']

           }

X = df[df['station']=='34 ST-PENN STA'].groupby('dayofyear').agg(agg_dict)[:-2]

X.columns = ['_'.join(col) for col in X.columns.values]

X.drop(columns='busyness_per_interval_sum',inplace=True)



scaler = MinMaxScaler(feature_range=(0, 1))

X.iloc[:] = scaler.fit_transform(X.values)



TRAIN_SIZE = 0.8



train_size = int(len(X) * TRAIN_SIZE)

test_size = len(X) - train_size

train, test = X[0:train_size], X[train_size:len(X)]



X_train = train.values

y_train = train[col_to_predict].values

X_test = test.values

y_test = test[col_to_predict].values



window_size = 7

batch_size = 1



train_data_gen = TimeseriesGenerator(X_train, y_train,

    length=window_size, sampling_rate=1,stride=1,

    batch_size=batch_size)

test_data_gen = TimeseriesGenerator(X_test, y_test,

    length=window_size, sampling_rate=1,stride=1,

    batch_size=batch_size)
#Usando keras, rede neural com LSTM em cascata

model = Sequential()



model.add(LSTM(256, input_shape=(window_size, X_train.shape[-1]),return_sequences=True))

model.add(Dropout(0.1))

model.add(LSTM(256, activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(units=64))

model.add(Dropout(0.1))

model.add(Dense(units=1))



model.compile(loss="mse", optimizer=optimizers.Adam(0.001))

model.summary()
# Early Stopping para evitar overfit

callbacks = [EarlyStopping(monitor='val_loss', patience=20, verbose=True, mode='auto')]



history = model.fit_generator(train_data_gen, epochs=100,validation_data=test_data_gen,

                              use_multiprocessing=True,workers=8,callbacks=callbacks)
def predict_sequence_full(last_data):

    # Utiliza os primeiros 7 dias do teste para realizar toda a previsao do teste

    predicted = []

    curr_frame = last_data[0][0]

    if len(curr_frame.shape)==2:

        curr_frame = curr_frame.reshape(model.input.shape.as_list()[1:])

    else:

        curr_frame = curr_frame[0,:,:]

    for i in range(len(last_data)*batch_size):

        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])

        curr_frame = curr_frame[1:]

        curr_frame = np.insert(curr_frame, len(curr_frame), predicted[-1], axis=0)

    return np.array(predicted).reshape(-1,1)



testPredict = predict_sequence_full(test_data_gen).reshape(-1, 1) 

testPredict = transform_single_column(scaler.inverse_transform,testPredict,5)



testPredictStep = model.predict_generator(test_data_gen)

testPredictStep = transform_single_column(scaler.inverse_transform,testPredictStep,5)



trainPredict = model.predict_generator(train_data_gen)

trainPredict = transform_single_column(scaler.inverse_transform,trainPredict,5)
X_reversed = X.copy()

X_reversed.iloc[:] = scaler.inverse_transform(X_reversed)

trainPredictPlot = np.array([np.nan]*len(X))

trainPredictPlot[window_size:len(trainPredict)+window_size] = trainPredict.reshape(-1)



testPredictPlot = np.array([np.nan]*len(X))

testPredictPlot[-len(testPredict):] = testPredict.reshape(-1)



testPredictStepPlot = np.array([np.nan]*len(X))

testPredictStepPlot[len(trainPredict)+window_size*2:] = testPredictStep.reshape(-1)



fig = plt.figure(figsize=(20,8))

plt.plot(X_reversed[col_to_predict].shift(-1),label='Real')

plt.plot(trainPredictPlot,label='Treino')

plt.plot(testPredictPlot,label='Predicao usando somente 7 dias')

plt.plot(testPredictStepPlot,label='Predicao dia-a-dia')

plt.xlabel('Dia do ano')

plt.ylabel('Média de movimentação na catraca')

plt.title('Predicao diária da media de movimento das catracas')

plt.legend()

plt.show()
from glob import glob



# Encontrei algums valores negativos. Funcao foi atualizada

def subtract_prev(rows):

    if len(rows) != 2:

        return np.nan

    prev, curr = rows

    subtracted = curr - prev if prev <= curr else curr

    return subtracted if subtracted>0 else np.nan





to_keep_busyest_station = []

to_keep_all_station = []

to_keep_hours = []



for file in glob("../input/dados_desafio/*"):

    df = pd.read_csv(file,parse_dates=[0])

    df['turnstile'] = df['ca'] + '-' + df['unit'] + '-' + df['scp'] # por catraca

    df = df.sort_values(['station','turnstile','time']).reset_index(drop=True)

    df = df[df['desc']=='REGULAR'] # filtra somente as catracas normais

    subtract_prev_rolling = lambda ser: ser.rolling(window=2).apply(subtract_prev)

    df_grouped_turnstile = df.groupby(['turnstile','station'])

    df['exits_per_interval']   = df_grouped_turnstile['exits'].apply(subtract_prev_rolling)

    df['entries_per_interval'] = df_grouped_turnstile['entries'].apply(subtract_prev_rolling)



    df.drop(columns=['entries','exits','desc'],inplace=True) # dropa colunas agora inuteis

    

    # Exitem numeros extremos: Filtro de limite

    cap = 1000

    for col in ['entries_per_interval', 'exits_per_interval']:

        df.loc[df[col] > cap, col] = np.nan

    df['busyness_per_interval'] = df.entries_per_interval + df.exits_per_interval



    agg_dict = {

            'busyness_per_interval':['sum','mean']

           }

    X = df[df['station']=='34 ST-PENN STA'].groupby([df['time'].dt.year,df['time'].dt.month]).agg(agg_dict)

    X.columns = ['_'.join(col) for col in X.columns.values]

    X.drop(columns='busyness_per_interval_sum',inplace=True)

    to_keep_busyest_station.append(X)

    

    X = df.groupby([df['time'].dt.year,df['time'].dt.month]).agg(agg_dict)

    X.columns = ['_'.join(col) for col in X.columns.values]

    X.drop(columns='busyness_per_interval_sum',inplace=True)

    to_keep_all_station.append(X)

    

    X = df.groupby([df['time'].dt.year,df['time'].dt.hour]).agg(agg_dict)

    X.columns = ['_'.join(col) for col in X.columns.values]

    X.drop(columns='busyness_per_interval_sum',inplace=True)

    to_keep_hours.append(X)
X_concat_hours = pd.concat(to_keep_hours)

X_concat_hours.index.names = ['year','hour']

X_concat_hours.reset_index(inplace=True)

X_concat_hours['yh'] = X_concat_hours['year'].astype(str)+'_'+X_concat_hours['hour'].astype(str)

X_concat_hours.sort_values(['year','hour'],inplace=True)



data = []

for year in X_concat_hours['year'].unique():

    race  = go.Scatter(

        x=X_concat_hours[X_concat_hours['year']==year]['hour'],

        y=X_concat_hours[X_concat_hours['year']==year]['busyness_per_interval_mean'],

        mode = 'lines+markers',

        name=f'{year}')

    data.append(race)



layout = go.Layout(showlegend=True,

                   title="Evolucao anual da media do movimento por hora",

    xaxis=dict(

        title="Horario"

    ),

    yaxis=dict(

        title="Media por ano"

    ) )

fig2 = go.Figure(data=data, layout=layout)

plotly.offline.iplot(fig2)
X_concat_all_station = pd.concat(to_keep_all_station)

X_concat_all_station.index.names = ['year','month']

X_concat_all_station.reset_index(inplace=True)

X_concat_all_station['ym'] = X_concat_all_station['year'].astype(str)+'_'+X_concat_all_station['month'].astype(str)

X_concat_all_station.sort_values(['year','month'],inplace=True)



trace  = go.Scatter(

    x=X_concat_all_station['ym'],

    y=X_concat_all_station['busyness_per_interval_mean'],

    mode = 'lines+markers',

    name='Media por mes de todas as estações')

data = [trace]





X_concat_busyest_station = pd.concat(to_keep_busyest_station)

X_concat_busyest_station.index.names = ['year','month']

X_concat_busyest_station.reset_index(inplace=True)

X_concat_busyest_station['ym'] = X_concat_busyest_station['year'].astype(str)+'_'+X_concat_busyest_station['month'].astype(str)

X_concat_busyest_station.sort_values(['year','month'],inplace=True)



trace  = go.Scatter(

    x=X_concat_busyest_station['ym'],

    y=X_concat_busyest_station['busyness_per_interval_mean'],

    mode = 'lines+markers',

    name='Media por mes da estacao mais cheia - 34 ST-PENN STA')

data = [trace,data[0]]



layout = go.Layout(showlegend=True,title="Movimentacao media das estacoes")

fig2 = go.Figure(data=data, layout=layout)

plotly.offline.iplot(fig2)