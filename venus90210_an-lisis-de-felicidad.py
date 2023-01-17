# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
datasetReportes= pd.read_excel('../input/reportefelicidad2019/Chapter2OnlineData.xls', sep=';',decimal=',')
datasetReportes.head()
datasetReportes.cov()
sns.set(style="white")



# Compute the correlation matrix

corr = datasetReportes.corr()



# Se genera una mascara para el triangulo superior

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Configuración de la imagen

f, ax = plt.subplots(figsize=(11, 9))



# Generar un mapa de colores personalizado

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Dibuja el mapa de calor con la máscara y la relación de aspecto

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
disp= datasetReportes.plot(kind='scatter', x='Log GDP per capita', y='Healthy life expectancy at birth')
disp= datasetReportes.plot(kind='scatter', x='Delivery Quality', y='Democratic Quality')
disp= datasetReportes.plot(kind='scatter',x='Perceptions of corruption', y='Freedom to make life choices')


# histograma duración de erupciones con 8 barras

 

datasetReportes.groupby('Year')['Positive affect'].mean().plot(kind='bar')



datasetReportes.groupby('Country name')['Positive affect'].mean().plot(kind='bar',figsize=(27, 9))
datasetFelicidad= pd.read_excel('../input/indicadoresfelicidad/felicidad.xlsx')
datasetFelicidad.corr()
sns.set(style="white")



# Compute the correlation matrix

corr = datasetFelicidad.corr()



# Se genera una mascara para el triangulo superior

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Configuración de la imagen

f, ax = plt.subplots(figsize=(11, 9))



# Generar un mapa de colores personalizado

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Dibuja el mapa de calor con la máscara y la relación de aspecto

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,vmin=-1,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
disp= datasetFelicidad.plot(kind='scatter', x='Explained by: GDP per capita', y='Happiness score')
disp= datasetFelicidad.plot(kind='scatter', x='Explained by: Healthy life expectancy', y='Happiness score')
datasetFelicidad.sort_values(by=['Happiness score'], ascending=[False])

datasetFelicidad.groupby('Country')['Happiness score'].sum().plot(kind='bar',figsize=(27, 9))