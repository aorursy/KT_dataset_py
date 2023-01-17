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
covid19 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
covid19
covid19['Doentes'] = covid19['Confirmed'] - covid19['Deaths'] - covid19['Recovered']
covid19
pais = covid19['Country/Region']=='Brazil'
pais
covid_br = covid19[pais]
print(covid_br.shape)

covid_br.info()
covid_br.head()
covid_br.to_csv('covid-brasil.csv', index=False)
covid_br['ObservationDate'] =  pd.to_datetime(covid_br['ObservationDate'])
covid_br.info()
#Excluindo a Coluna Combustíveisas colunas desnecesssárias
dados= covid_br.drop(columns=["SNo", "Province/State", "Country/Region", "Last Update", "Confirmed", "Deaths", "Recovered"])
dados.head(10)
#Criando um índice na coluna Mes do DF
dados.index = dados['ObservationDate']
dados.drop('ObservationDate',inplace=True, axis=1)
dados.head()
from matplotlib import pyplot as plt
#Plotando gráfico para analisar comportamento dos dados
plt.plot(dados)
plt.title("Análise dos Doentes no Brasil")
plt.ylabel("ObservationDate")
plt.xlabel("Doentes")
plt.show()
#Aplicando a Regressão Linear
from sklearn.linear_model import LinearRegression
X = [i for i in range(0, len(dados))]
X = np.reshape(X, (len(X), 1))
y = dados
modelo = LinearRegression()
modelo.fit(X, y)
modelo.coef_
tendencia = modelo.predict(X)

#Plotando a linha de regressão linear
plt.plot(dados.values, color='blue',label='Original')
plt.title('Análise de Tendência do Número de Doentes por COVID-19 no Brasil')
plt.ylabel('ObservationDate')
plt.xlabel('Doentes')
plt.legend('Y = 0.00028*x-2.27836')
plt.plot(tendencia,color='red', label='Tendência' )
plt.show()
#MODELO NAIVE

train=dados[0:42] 
test=dados[43:]

dd = np.asarray(train['Doentes'])
y_hat = test.copy()
y_hat['naive'] = dd[len(dd)-1]
plt.figure(figsize=(12,8))
plt.plot(train.index, train['Doentes'], label='Train')
plt.plot(test.index,test['Doentes'], label='Test')
plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.show()
# Dividindo o DataFrame
from sklearn.model_selection import train_test_split

# Treino e teste
train, test = train_test_split(dados, test_size=0.15, random_state=42)

# Veificando o tanho dos DataFrames
train.shape, test.shape
# Checking distribution along 25%,50%,75% percentiles
train.describe()
d = train.groupby(train['Doentes']).sum()
d
