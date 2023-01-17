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
# Importando as bibliotecas necessárias para o trabalho

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.ar_model import AR
from random import random
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm

# Importando os arquivos nos dataset df e df2

df = pd.read_excel('/kaggle/input/brasil/Brasil.xlsx',skiprows=12,usecols=[0,1,4],encoding='latin1')
df2 = pd.read_excel('/kaggle/input/brasil/MENSAL_BRASIL-DESDE_Jan2013.xlsx',skiprows=15,usecols=[0,1,4],encoding='latin1')
df.info()
# Visualizando o cabeçalho e as primeiras linhas

df.head()
# Renomeando as colunas do dataset df

df.rename(columns={"MÊS": "periodo", "PRODUTO": "produto","PRECO MÉDIO REVENDA": "valor"},inplace=True)
# Verificando a alteração dos nomes e o tamanho do dataset em linhas e colunas

print(df.info())
print(df.shape)
# Filtrando o dataset pela colunas "produto" e pelo tipo 'ETANOL HIDRATADO'

dfEtanol = df[df['produto'] =='ETANOL HIDRATADO']
# Verificando o dataset criado "dfEtanol" pelas últimas linhas

dfEtanol.tail()
# Removendo a coluna "produto"

dfEtanol = dfEtanol.drop(columns=['produto'])
dfEtanol.head()
# Visualizando o segundo dataset criado

df2.head()
# Renomeando as colunas

df2.rename(columns={"MÊS": "periodo", "PRODUTO": "produto","PREÇO MÉDIO REVENDA": "valor"},inplace=True)
df2.head()
# Criando dataframe com tipo de combustível 'ETANOL HIDRATADO'

dfEtanol2 = df2[df2['produto'] =='ETANOL HIDRATADO']
# Removendo a coluna referente à combustíveis

dfEtanol2 = dfEtanol2.drop(columns=['produto'])
dfEtanol2.head()
# Juntando os dataframes

dados = pd.concat([
    dfEtanol,dfEtanol2
],ignore_index=True,sort=True) 
dados.count()

# Atribuindo a coluna periodo no index da base

dados.index = dados['periodo']
dados.drop('periodo',inplace=True, axis=1)
dados.head()

# Mostra de dados coletados

plt.plot(dados)
plt.title('Evolução do preço do Etanol Hidratado')
plt.xlabel('Periodo')
plt.ylabel('Valores')
plt.show()

X = [i for i in range(0, len(dados))]
X = np.reshape(X, (len(X), 1))
y = dados
modelo = LinearRegression()
modelo.fit(X, y)
modelo.coef_
tendencia = modelo.predict(X)

plt.plot(dados.values, color='blue',label='Original')
plt.title('Evolução do preço do Etanol Hidratado')
plt.ylabel('Periodo')
plt.xlabel('Preços')
plt.legend('Y = 0.00028*x-2.27836')
plt.plot(tendencia,color='red', label='Tendência' )
plt.show()

# Estatísticas do modelo ARMA

modelo_arma = sm.tsa.ARMA(dados, (3,0)).fit(disp=False)
print(modelo_arma.summary())
# Previsão do valor do combustível para 12 meses

fig, ax = plt.subplots(figsize=(8,6))
fig = modelo_arma.plot_predict(start='2001-07-01', end='2020-10-01', ax=ax)
plt.title('Previssão do preço para 12 meses, utilizando o modelo ARMA')
plt.ylabel('Preços')
plt.xlabel('Periodo')
legend = ax.legend(loc='upper left')
# Estatísticas do modelo ARIMA

modelo_arima = ARIMA(dados, order=(0, 1, 1)).fit()
print(modelo_arima.summary())
dados.info()
# Previsão do valor do combustível para 12 meses

fig, ax = plt.subplots(figsize=(8,6))
fig = modelo_arima.plot_predict(start='2001-08-01', end='2020-10-01', ax=ax)
plt.title('Previssão do preço para 12 meses, utilizando o modelo ARIMA')
plt.ylabel('Preços')
plt.xlabel('Periodo')
legend = ax.legend(loc='upper left')
# Coeficientes estatísticos do Modelo ARIMA

naive = ARIMA(dados, order=(0, 0, 0))
naive_treinado = naive.fit()
print(naive_treinado.summary())
# Gráfico

plt.rcParams.update({'figure.figsize':(9,3)})
naive_treinado.plot_predict(start=1,end=230)
plt.show()
# Modelo SNaive com sazonalidade

snaive = ARIMA(dados, order=(0, 1, 0))
snaive_treinado = snaive.fit()
print(snaive_treinado.summary())
# Gráfico

snaive_treinado.plot_predict(start=1,end=230)
plt.show()
# Holt-Winters sem tendência e ajuste sazonal

model = ExponentialSmoothing(dados)
hw_model = model.fit()
pred = hw_model.predict(start=dados.index[0], end=dados.index[219])
plt.title('Holt-Winters Filtering')
plt.plot(dados.index, dados, label='Original', color='k')
plt.plot(pred.index, pred, label='Holt-Winters',color='r')
plt.legend(loc='best');
# Previsão para 36 meses, usando o método Holt-Winters

model = ExponentialSmoothing(dados, seasonal_periods=36)
hw_model = model.fit()
pred = hw_model.predict(start='2019-11-01', end='2022-10-01')
plt.plot(dados.index, dados, label='Original')
plt.title('Forecasts from HoltWinters')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc='best');
plt.show()
# Modelo Holt-Winters com tendência e sem ajuste sazonal

model = ExponentialSmoothing(dados, trend='add',seasonal='add')
hw_model = model.fit()
pred = hw_model.predict(start=0, end=219)
plt.title('Holt-Winters Filtering')
plt.plot(dados.index, dados, label='Original')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc='best');
plt.show()
model = ExponentialSmoothing(dados, trend='mul',seasonal='mul')
hw_model = model.fit(optimized=True, use_boxcox=False, remove_bias=False)
pred = hw_model.predict(start=14, end=219)
plt.title('Holt-Winters Filtering, com Tendencia e Ajuste sazonal')
plt.plot(dados.index, dados, label='Original')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc='best');
plt.show()

# Modelo Holt-Winters com tendência e ajuste sazonal, previsão para 36 meses

model = ExponentialSmoothing(dados, seasonal_periods=7 ,trend='add', seasonal='mul',damped=True,)
hw_model = model.fit()
pred = hw_model.predict(start='2019-11-01', end='2022-10-01')
plt.title('Holt-Winters Filtering, com Tendencia e Ajuste sazonal')
plt.plot(dados.index, dados, label='Original')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc='best');
plt.show()



