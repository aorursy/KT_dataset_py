# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from datetime import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



dataTotal =  pd.read_csv("/kaggle/input/gas-prices-in-brazil/2004-2019.tsv", sep="\t", parse_dates={'datetime': ['DATA INICIAL']})


def acertaValor (data,  campo ):

    for line in data.index:

        if data.loc[line, campo] == '-':

            valor = 0

        else:

            valor = float(data.loc[line, campo])



        if valor > 50000:

            data.loc[line, campo] = round(valor / 100000, 3)

        elif valor > 5000:

            data.loc[line, campo] = round(valor / 10000, 3)

        elif valor > 500:

            data.loc[line, campo] = round(valor / 1000, 3)

        elif valor > 100:

            data.loc[line, campo] = round(valor / 100, 3)

        elif valor > 10:

            data.loc[line, campo] = round(valor / 10, 3)

        else:

            data.loc[line, campo] = round(valor, 3)



def eliminaZeros (data,  campo ):

    data[campo].replace(0 , data[campo].mean())
#dataTotal['PRODUTO'].unique()
dataTotal.loc[1000]
gasolina =  dataTotal[dataTotal['PRODUTO'] == 'GASOLINA COMUM'].copy()

etanol   =  dataTotal[dataTotal['PRODUTO'] == 'ETANOL HIDRATADO'].copy()
#gasolina.sort_values(by="PREÇO MÉDIO REVENDA")
acertaValor(gasolina ,'PREÇO MÉDIO REVENDA')

eliminaZeros(gasolina ,'PREÇO MÉDIO REVENDA')

#acertaValor(etanol ,'PREÇO MÉDIO REVENDA')
#print (gasolina)
ln = LinearRegression()

lenght = list(range(len(gasolina['PREÇO MÉDIO REVENDA'])))

type(lenght)

plt.plot( lenght ,dataTotal[dataTotal['PRODUTO'] == 'GASOLINA COMUM' ]['PREÇO MÉDIO REVENDA'] , 'go', marker='.', markersize = (1))

plt.show()
plt.plot(etanol['DATA INICIAL'],etanol['PREÇO MÉDIO REVENDA'] , marker='.', markersize=(.1))
print(dataTotal[dataTotal['PRODUTO'] == 'GASOLINA COMUM' ]['PREÇO MÉDIO REVENDA'].max())

print(dataTotal[dataTotal['PRODUTO'] == 'GASOLINA COMUM' ]['PREÇO MÉDIO REVENDA'].min())

print(dataTotal[dataTotal['PRODUTO'] == 'GASOLINA COMUM' ]['PREÇO MÉDIO REVENDA'].mean())
acertaValor(gasolina ,'MARGEM MÉDIA REVENDA')

gasolina.corr()
print(type(lenght))

testeLen = (np.asarray(lenght)/max(lenght))

ln.fit(testeLen.reshape(-1,1), gasolina['PREÇO MÉDIO REVENDA' ])

gasSorted = gasolina['PREÇO MÉDIO REVENDA'].sort_values()
plt.plot( lenght ,gasolina['PREÇO MÉDIO REVENDA'] , 'go', marker='.', markersize = (1))

plt.plot(lenght, testeLen*ln.coef_ + ln.intercept_  , color='r', linestyle='-')

plt.show()
print(ln.coef_)

print(ln.intercept_)
ln2 = LinearRegression()
testeData = gasolina['datetime']

print(type(testeData[10:10]))



#timeStamps = datetime.timestamp(testeData[:])

#teste2 = (np.asarray(testeData)/max(lenght))

print(type(gasolina['datetime']))



#ln2.fit(gasolina['datetime'].reshape(-1,1), gasolina['PREÇO MÉDIO REVENDA' ])