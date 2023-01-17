# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
base = pd.read_csv('../input/games.csv')
base.head()
base = base.drop('Other_Sales', axis = 1)
base = base.drop('Global_Sales', axis = 1)
base = base.drop('Developer', axis = 1)

base = base.dropna(axis = 0)
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]

games_name = base.Name
base = base.drop('Name', axis = 1)
base.head()
previsores = base.iloc[:, [0,1,2,3,7,8,9,10,11]].values
sales_na = base.iloc[:, 4].values
sales_eu = base.iloc[:, 5].values
sales_jp = base.iloc[:, 6].values
previsores
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
previsores[:, 8] = labelencoder.fit_transform(previsores[:, 8])
previsores[:, 6] = previsores[:, 6].astype(float)
previsores[1]
onehotencoder = OneHotEncoder(categorical_features = [0,2,3,8])
previsores = onehotencoder.fit_transform(previsores).toarray()
previsores.shape #258 linhas e 61 colunas
camada_entrada = Input(shape=(61,))
camada_oculta1 = Dense(units = 32, activation = 'sigmoid')(camada_entrada)
camada_oculta2 = Dense(units = 32, activation = 'sigmoid')(camada_oculta1)
camada_saida1 = Dense(units = 1, activation = 'linear')(camada_oculta2)
camada_saida2 = Dense(units = 1, activation = 'linear')(camada_oculta2)
camada_saida3 = Dense(units = 1, activation = 'linear')(camada_oculta2)
regressor = Model(inputs = camada_entrada,
                  outputs = [camada_saida1, camada_saida2, camada_saida3])
regressor.compile(optimizer = 'adam',
                  loss = 'mse')
regressor.fit(previsores, [venda_na, venda_eu, venda_jp],
              epochs = 5000, batch_size = 100)
previsao_na, previsao_eu, previsao_jp = regressor.predict(previsores)
import matplotlib.pyplot as plt
plt.scatter(previsao_na, games_name, color='green')
plt.scatter(sales_na, games_name, color='red', alpha=0.2)
