# importar las librerias necesarias
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt

# Importar el conjunto de datos y codificar la fecha

df = pd.read_csv('../input/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')
df['date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date
group = df.groupby('date')
Real_Price = group['Weighted_Price'].mean()

# datos divididos
prediction_days = 30
df_train= Real_Price[:len(Real_Price)-prediction_days]
df_test= Real_Price[len(Real_Price)-prediction_days:]

# Preproceso de datos
training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1)) # Da una nueva forma a una matriz sin cambiar sus datos.
from sklearn.preprocessing import MinMaxScaler #El escalador transforma las características escalándolas a un rango dado, por defecto (0,1), aunque puede ser personalizado. Este tipo de escalado suele denominarse frecuentemente "normalización" de los datos.
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
X_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]
X_train = np.reshape(X_train, (len(X_train), 1, 1))

# Importar las bibliotecas y paquetes de Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Inicializando el RNN
regressor = Sequential()


# Agregar la capa de entrada y la capa LSTM
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))


# Agregar la capa de salida
regressor.add(Dense(units = 1))

# Compilando el RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# Ajuste del RNN al conjunto de entrenamiento
regressor.fit(X_train, y_train, batch_size = 5, epochs = 100)


# Hacer las predicciones
test_set = df_test.values
inputs = np.reshape(test_set, (len(test_set), 1))
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predicted_BTC_price = regressor.predict(inputs)
predicted_BTC_price = sc.inverse_transform(predicted_BTC_price)


# Visualizando los resultados
plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')
ax = plt.gca()  
plt.plot(test_set, color = 'red', label = 'Precio real de BTC')
plt.plot(predicted_BTC_price, color = 'blue', label = 'Precio previsto de BTC')
plt.title('Predicción de precios BTC', fontsize=40)
df_test = df_test.reset_index()
x=df_test.index
labels = df_test['date']
plt.xticks(x, labels, rotation = 'vertical')
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
plt.xlabel('Tiempo', fontsize=40)
plt.ylabel('Precio BTC (USD)', fontsize=40)
plt.legend(loc=2, prop={'size': 25})
plt.show()