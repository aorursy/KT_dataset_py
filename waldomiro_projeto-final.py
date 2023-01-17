import numpy as np

import pandas as pd 

from sklearn.linear_model import LinearRegression

from sklearn import linear_model

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import seaborn as sb

import plotly.graph_objs as go

import plotly.offline as py

import plotly

plotly.offline.init_notebook_mode()

import datetime



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        
dataset=pd.read_csv('../input/gas-prices-in-brazil/2004-2019.tsv',sep='\t')

dataset['DATA INICIAL']=pd.to_datetime(dataset['DATA INICIAL'])

dataset['DATA FINAL']=pd.to_datetime(dataset['DATA FINAL'])

EtanolData=dataset.loc[dataset['PRODUTO'] == 'ETANOL HIDRATADO']

GasolinaData=dataset.loc[dataset['PRODUTO'] == 'GASOLINA COMUM']

DieselData=dataset.loc[dataset['PRODUTO'] == 'ÓLEO DIESEL']

Diesel10Data=dataset.loc[dataset['PRODUTO'] == 'ÓLEO DIESEL S10']

GLPData=dataset.loc[dataset['PRODUTO'] == 'GLP']

GNVData=dataset.loc[dataset['PRODUTO'] == 'GNV']



x1=GasolinaData['ANO']

x2=EtanolData['ANO']

y1=GasolinaData['PREÇO MÉDIO REVENDA']

y2=EtanolData['PREÇO MÉDIO REVENDA']

data=[go.Scatter(x=x1,y=y1)]

data2=[go.Scatter(x=x2,y=y2)]

fig=go.Figure(data=data)

fig2=go.Figure(data=data2)

py.iplot(fig)

py.iplot(fig2)
x1=np.array(GasolinaData['ANO']).reshape(-1, 1)

y1=GasolinaData['PREÇO MÉDIO REVENDA']

x2=np.array(EtanolData['ANO']).reshape(-1, 1)

y2=EtanolData['PREÇO MÉDIO REVENDA']

x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2) 

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2) 
linreg = LinearRegression()

linreg.fit(x_train, y_train)

linreg2 = LinearRegression()

linreg2.fit(x2_train, y2_train)
linreg.predict(np.array(2010).reshape(-1,1))

linreg2.predict(np.array(2010).reshape(-1,1))