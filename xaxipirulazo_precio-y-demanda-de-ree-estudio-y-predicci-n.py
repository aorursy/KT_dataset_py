# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv

from keras.utils import to_categorical

from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn import metrics

from sklearn.metrics import silhouette_samples, silhouette_score

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt

from pandas_profiling import ProfileReport

from sklearn.preprocessing import MinMaxScaler

from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import LabelBinarizer

import datetime

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense,Dropout,LSTM

from keras.callbacks import EarlyStopping,ModelCheckpoint



import xgboost



from sklearn.metrics import mean_absolute_error







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

data = pd.read_csv("/kaggle/input/red-electrica-espaola-consumo-y-precios/2020-03-26-15-33_chronograf_data.csv")

N = int(data.shape[0]/2)

data
demandas = data.iloc[:N]

precios = data.iloc[N:]

timestamps = pd.to_datetime(data._time[:N])

df = pd.DataFrame(columns = ['precio','demanda','year','month','day','weekday','weekdayLabel','hour'], index = timestamps)

df.year = df.index.year

df.month = df.index.month

df.day = df.index.day

df.weekday = df.index.dayofweek

df.hour = df.index.hour

df.weekdayLabel = df.weekday.map(lambda num: ['Lunes','Martes','Miercoles','Jueves','Viernes','Sábado','Domingo'][num])

df.precio = precios._value.values

df.demanda = demandas._value.values
df
weeks = 3

days =7



for week in range(weeks):

    fig, axs = plt.subplots(2, days, figsize=(30,5))

    for day in range(days):

        axs[0,day].plot(df.hour[:24],df.precio[week*168+day*24:week*168+day*24+24])

        axs[1,day].plot(df.hour[:24],df.demanda[week*168+day*24:week*168+day*24+24], 'tab:orange')





        axs[0,day].set( ylabel='Precio')

        axs[1,day].set(xlabel=df.weekdayLabel[day*24] , ylabel='Demanda')



# Hide x labels and tick labels for top plots and y ticks for right plots.
df
df[['demanda','precio']]
precio_demanda_dias = df[['demanda','precio']].values.reshape(int(N/24),48)

# k means determine k

distortions = []

K = range(1,10)

for k in K:

    kmeanModel = KMeans(n_clusters=k).fit(precio_demanda_dias)

    kmeanModel.fit(precio_demanda_dias)

    distortions.append(sum(np.min(cdist(precio_demanda_dias, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / precio_demanda_dias.shape[0])



# Plot the elbow

plt.plot(K, distortions, 'bx-')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k for Prices')

plt.show()

K = 3
precio_demanda_dias = df[['demanda','precio']].values.reshape(int(N/24),48)

precio_demanda_ts = df[['demanda','precio']].values.reshape(int(N/24),24,2)

demandas = precio_demanda_ts[:,:,0]

precios = precio_demanda_ts[:,:,1]




columns = ["precio"+str(hour) if hour < 24 else "demanda"+str(hour-24)  for hour in range(48)] +['2017','2018','2019']+[str(i) for i in range(12)]+ [str(i) for i in range(31)] + ['Lunes','Martes','Miercoles','Jueves','Viernes','Sábado','Domingo']+['maxPrice','maxDemanda','minPrice','minDemanda','meanPrice','meanDemanda','weekDayLabel','fridaySaturdaySunday','label',]





maxPrice = precios.max(axis=1).reshape(-1,1)

maxDemanda = demandas.max(axis=1).reshape(-1,1)

minPrecio = precios.min(axis=1).reshape(-1,1)

minDemanda = demandas.min(axis=1).reshape(-1,1)

meanPrecio = precios.mean(axis=1).reshape(-1,1)

meanDemanda = demandas.mean(axis=1).reshape(-1,1)

year = df.year[::24].values.reshape(-1,1)

year = to_categorical(year)[:,-3:]

month = df.month[::24].values.reshape(-1,1)

month = to_categorical(month)[:,-12:]

day = df.day[::24].values.reshape(-1,1)

day = to_categorical(day)[:,-31:]

weekDay = df.weekday[::24].values.reshape(-1,1)

weekday = to_categorical(weekDay)

weekDayLabel = df.weekdayLabel[::24].values.reshape(-1,1)

fridaySaturdaySunday = df.weekday[::24].map( lambda day: 1 if day in [4,5,6] else 0).values.reshape(-1,1)

clusterer = KMeans(n_clusters=K, random_state=10)

labels = clusterer.fit_predict(precio_demanda_dias)

labels = np.array([str(label) for label in labels]).reshape([-1,1])

precios_demanda_etiquetadas_valores = np.hstack([precio_demanda_dias,year,month,day,weekday,maxPrice,maxDemanda,minPrecio,minDemanda,meanPrecio,meanDemanda,weekDayLabel,fridaySaturdaySunday,labels])

precios_demanda_atributos_etiquetadas = pd.DataFrame(precios_demanda_etiquetadas_valores, columns = columns)
precios_demanda_atributos_etiquetadas
seleccion_de_estudio_automatico = precios_demanda_atributos_etiquetadas.iloc[:,-9:]
seleccion_de_estudio_automatico
#profile = ProfileReport(seleccion_de_estudio_automatico, title='Pandas Profiling Report', html={'style':{'full_width':True}})

#profile.to_widgets()

#profile.to_file(output_file="output.html")
tc = pd.crosstab(index=seleccion_de_estudio_automatico.weekDayLabel,

            columns=seleccion_de_estudio_automatico.label, margins=True)

dia_por_etiqueta = tc.values[-1,:-1]

etiqueta_por_dia = tc.values[:-1,-1:]

tc_percentaje_particion_dia = tc.values[:-1,:-1]*100/dia_por_etiqueta

tc_percentaje_dia_particion = tc.values[:-1,:-1]*100/etiqueta_por_dia

distribucion_porcentaje_particion = dia_por_etiqueta*100/tc.values[-1,-1]

distribucion_porcentaje_dia = etiqueta_por_dia*100/tc.values[-1,-1]

distribucion_si_aleatorio = etiqueta_por_dia * dia_por_etiqueta/ tc.values[-1,-1]
tc
tc_percentaje_particion_dia
tc_percentaje_dia_particion
distribucion_porcentaje_dia
distribucion_si_aleatorio
pd.crosstab(index=seleccion_de_estudio_automatico.fridaySaturdaySunday,

            columns=seleccion_de_estudio_automatico.label, margins=True)
pd.crosstab(index=pd.cut(seleccion_de_estudio_automatico.meanPrice, 5

                        ,labels=['muy bajo','bajo','medio','alto','muy alto']),

            columns=seleccion_de_estudio_automatico.fridaySaturdaySunday, margins=True)

pd.crosstab(index=pd.cut(seleccion_de_estudio_automatico.meanDemanda, 5

                        ,labels=['muy bajo','bajo','medio','alto','muy alto']),

            columns=seleccion_de_estudio_automatico.label, margins=True)
pd.crosstab(index=pd.cut(seleccion_de_estudio_automatico.meanDemanda, 5

                        ,labels=['muy bajo','bajo','medio','alto','muy alto']),

            columns=seleccion_de_estudio_automatico.fridaySaturdaySunday, margins=True)
pd.crosstab(index=pd.cut(seleccion_de_estudio_automatico.maxPrice, 5,labels=['muy bajo','bajo','medio','alto','muy alto']

                        ),

            columns=seleccion_de_estudio_automatico.weekDayLabel, margins=True)
pd.crosstab(index=pd.cut(seleccion_de_estudio_automatico.maxPrice, 5,labels=['muy bajo','bajo','medio','alto','muy alto']

                        ),

            columns=seleccion_de_estudio_automatico.fridaySaturdaySunday, margins=True)
pd.crosstab(index=pd.cut(seleccion_de_estudio_automatico.maxDemanda, 5,labels=['muy bajo','bajo','medio','alto','muy alto']

                        ),

            columns=seleccion_de_estudio_automatico.weekDayLabel, margins=True)
pd.crosstab(index=pd.cut(seleccion_de_estudio_automatico.maxDemanda, 5,labels=['muy bajo','bajo','medio','alto','muy alto']

                        ),

            columns=seleccion_de_estudio_automatico.fridaySaturdaySunday, margins=True)
pd.crosstab(index=pd.cut(seleccion_de_estudio_automatico.meanPrice, 5,labels=['muy bajo','bajo','medio','alto','muy alto']

                        ),

            columns=pd.cut(seleccion_de_estudio_automatico.meanDemanda, 5,labels=['muy bajo','bajo','medio','alto','muy alto']), margins=True)
#Parametros

horizon = 32

forecast = 24      

start_test ='2018-09-01'
df


precio = df.precio.values.reshape(-1,1)

demanda = df.demanda.values.reshape(-1,1)

precio_diff = df.precio.pct_change().values.reshape(-1,1)

demanda_diff =  df.demanda.pct_change().values.reshape(-1,1)

hours = to_categorical(df.hour)

columns = ['precio','demanda','precio_diff','demanda_diff']+['hour'+str(i) for i in range(24)]

X = pd.DataFrame(np.hstack([precio,demanda,precio_diff,demanda_diff,hours]),columns = columns)

X.index = data._time[:N].map( lambda t: pd.to_datetime(t))

X = X[1:] #Dropnan





#Reescalamos las variables continuas y volvemos a montar el conjunto de entrenamiento.

original_train = X[X.index<pd.to_datetime(start_test).tz_localize('Europe/Madrid')]

original_test = X[X.index>pd.to_datetime(start_test).tz_localize('Europe/Madrid')]



scaler = MinMaxScaler(feature_range=(-1, 1))

scaler = scaler.fit(original_train.iloc[:,:4])





X = pd.DataFrame(np.hstack([scaler.transform(X.iloc[:,:4]),hours[1:]]),columns = columns)

X.index = data._time[1:N].map( lambda t: pd.to_datetime(t))
price_index = 0



def dataset_slidding_window_supervised(X_dataset,horizon,forecast,output):

    num_features = X_dataset.shape[1]

    num_samples = X_dataset.shape[0] - horizon - forecast

    X = np.zeros((num_samples,horizon,X_dataset.shape[1]))    

    Y = np.zeros((num_samples,forecast))

    for i in range(num_samples):

        subset = np.array(X_dataset.iloc[i:i+horizon,:num_features])

        X[i,:,:] = subset

        subset = np.array(X_dataset.iloc[i+horizon:i+horizon+forecast,output])

        Y[i,:] = subset

    return X,Y



X,Y = dataset_slidding_window_supervised(X,horizon,forecast,price_index)



print(original_test.index[0],original_test.index[30*24+1])


X_train = X[:original_train.shape[0]]

Y_train = Y[:original_train.shape[0]]

X_test = X[original_test.shape[0]:][:30*24+1]  #(Cogemos un mes completo hacia adelante) que corresponde con septiembre.

Y_test = Y[original_test.shape[0]:][:30*24+1]

batch_size = 516



# design network

model = Sequential()

model.add(LSTM(100, input_shape=(X.shape[1], X.shape[2]),return_sequences = False))

model.add(Dropout(0.4))

model.add(Dense(25))

model.add(Dense(Y.shape[1]))



model.compile(loss='mae', optimizer='adam')

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# fit network

history = model.fit(X_train, Y_train, epochs=2000, batch_size=batch_size, validation_data=(X_test, Y_test), verbose=2, shuffle=True, callbacks = [early_stopping,checkpoint])

# plot history

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()

























scaler.min_, scaler.scale_ = scaler.min_[0], scaler.scale_[0]
prediction = model.predict(X_test)



Y_rescaled = []

Y_pred_rescaled = []

for i in range(30): 

    Y_i = scaler.inverse_transform(Y_test[i*24].reshape(-1,1))

    y_i_pred = scaler.inverse_transform(prediction[i*24].reshape(-1,1))

    Y_rescaled.append(Y_i)

    Y_pred_rescaled.append(y_i_pred)

    fig, ax = plt.subplots()

    line1, = ax.plot(Y_i, linewidth=2,

                     label='Valor real')





    line2, = ax.plot(y_i_pred,'--',

                     label='Prediccion',color='red')



    ax.legend(loc='lower left')

    plt.title(str(original_test.index[i*24])+' Hasta '+str(original_test.index[i*24+24]))

    plt.show()

np.array(Y_rescaled).shape

np.array(Y_pred_rescaled).shape
errors = []

for i in range(30):

    errors.append(mean_absolute_error(Y_rescaled[i],Y_pred_rescaled[i]))

plt.plot(errors)
np.mean(errors)