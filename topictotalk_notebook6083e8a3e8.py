print('Proyecto Sephora')
#importamos librerias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt

#tamaño gráficas

sns.set(rc={'figure.figsize':(20,10)})
#cargamos nuestro dataset
sephora = pd.read_csv('../input/sephora/sephora.csv')
#miramos como aparece el dataset
sephora.head()
#miramos ahora las tipologías de variables
sephora.info()
#miramos la descripción del dataset
sephora.describe()
#Empezamos con un gráfico de barras de "rating" // sns es la librería que hemos importado al inicio llamada seaborn 
sns.countplot(x='rating', data=sephora)
#ejecutamos un boxplot. solo sirve para las variables numéricas
sns.boxplot(y='price', data=sephora)
#ejecutamos un boxplot. solo sirve para las variables numéricas
sns.boxplot(y='rating', data=sephora)
#ejecutamos un histograma
sns.distplot(sephora['rating'])
sephora['price']
#Empezamos con análisis multivariadas. Ejecutamos un scatter plot
sns.relplot(x='category', y='price', data=sephora)

#hacemos el mismo pero más lienal para identificar tendencias
sns.relplot (x='price', y='love', data=sephora)
#hacemos el mismo pero más lienal para identificar tendencias
sns.relplot (x='love', y='price', data=sephora)
#cómo miramos todas las variables a la vez
sns.pairplot(sephora)
#vemos la correlación entre variable categórica y variable numérica
print(sns.barplot(x='category', y='rating', data=sephora))
#vemos la correlación entre variable categórica y variable numérica
print(sns.barplot(x='category', y='love', data=sephora))
#vemos la correlación entre variable categórica y variable numérica
print(sns.barplot(x='category', y='price', data=sephora))
#y si queremos mirar todas las correlaciones... solo lo hará con variables numéricas.
corr = sephora.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, vmax=.8, linewidths=0.01,
           square=True, annot=True)
#Probamos otros tipos de gráficos y cruce de variables
sns.boxplot(x="category", y="price", data=sephora)
#Probamos otros tipos de gráficos y cruce de variables
sns.boxplot(x="category", y="price", hue='rating', data=sephora)
#Probamos otros tipos de gráficos y cruce de variables
sns.boxplot(x="category", y="rating", data=sephora)

