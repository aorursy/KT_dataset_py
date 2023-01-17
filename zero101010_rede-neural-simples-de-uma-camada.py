

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style()

#Separar treino de teset

from sklearn.model_selection import train_test_split

#Padrozinar os dados

from sklearn.preprocessing import StandardScaler

#importar dataset

from sklearn.datasets import fetch_california_housing

def sigmoid(x):

    return 1.0/ (1+ np.exp(-x))



x1=10

x2= -2



result = 1-(2*x1)+(5*x2)

y_hat = sigmoid(result)

y_hat

# Exemplificando fução sigmoid

x = np.arange(-10,10,0.1)

y = sigmoid(x)

plt.plot(x,y);

dataset = fetch_california_housing()

features = dataset.feature_names

# Dividir entre treino,validação e teste

X_train_orignal,X_test,y_train_original,y_test = train_test_split(dataset.data,dataset.target)

#Dados de treino e validação

X_train,X_valid,y_train,y_valid = train_test_split(X_train_orignal,y_train_original)
#Criar dataframe 

df = pd.DataFrame(X_train)

# inserir as features do dataset no lugar dos índices

df.columns = features

df.head()
# Padronizando os dados do dataset para ficarem na mesma escala

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_valid = scaler.transform(X_valid)

# Tranformar os dados anteriormente em dataframe 

df = pd.DataFrame(X_train)

df.columns = features

df.head()
# Definir modelo

model = keras.models.Sequential(

    [

        keras.layers.Dense(30,activation="relu",input_shape=(X_train.shape[1:])),

        keras.layers.Dense(1)

    ]

)
# Compilar nn

model.compile(loss='mean_squared_error',optimizer='sgd')

# Obter histórico de loss

history = model.fit(X_train,y_train,epochs=50,validation_data=(X_valid,y_valid))

# Verificar a média quadrática para identificar o erro de acordo com o conjunto de teste

error = model.evaluate(X_test,y_test)
# Fazer uma predição após treinar o modelo

model.predict(X_train[:2])
# Caso queira colocar somente uma entrada para verificar é necessário fazer isso

model.predict(X_train[0].reshape((1,-1)))
#plotar histórico de perda, o que significa que vai mostrar os índice de acerto, a acurácia

pd.DataFrame(history.history).plot();