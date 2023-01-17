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
# Bibliotecas gráficas

import matplotlib.pyplot as plt

import seaborn as sns
# Dataset

df = pd.read_csv('../input/weatherAUS.csv')
# Tamanho do dataframe (dimensão)

df.shape
# Verificar a cabeça do dataframe para verificar inconsistências

df.head()
# Conferir a calda do dataframe para verificar inconsistências

df.tail()
# Conferir dados aleatórios do dataframe para verificar inconsistências

df.sample(5)
# Nome das colunas

df.columns
# Verificando os tipos das colunas do dataset

df.dtypes
# Converter coluna 'Date' no tipo datetime

df['Date'] = pd.to_datetime(df['Date'])

df.dtypes
# Os 5 dias com maiores temperaturas máximas registradas

df.nlargest(5, 'MaxTemp')
# Os 5 dias com menores temperaturas mínimas registradas

df.nsmallest(5, 'MinTemp')
df['Location'].value_counts()
# Retirando NA`s

df.dropna().reset_index()
df.describe()
# Dias com maior tempo de sol 

df.nlargest(5, 'Sunshine')
# Dias com menor tempo de sol 

df.nsmallest(5, 'Sunshine')
df['Year'] = pd.to_datetime(df['Date']).dt.year
df_anos = df.groupby(['Year', 'Location'], as_index=False)['MaxTemp'].mean()
df_anos['Year'].value_counts()
df_anos.nlargest(5, 'MaxTemp')
df_rain = df.copy()

df_rain = df_rain.applymap(lambda x: 1 if x == 'Yes' else x)

df_rain = df_rain.applymap(lambda x: 0 if x == 'No' else x)
df[df['Year'] == 2007].plot(title='2007', kind='scatter', x='Temp9am', y='Humidity9am',

               alpha=0.5, color='blue')



df[df['Year'] == 2013].plot(title='2013', kind='scatter', x='Temp9am', y='Humidity9am',

               alpha=0.5, color='green')



df[df['Year'] == 2017].plot(title='2017', kind='scatter', x='Temp9am', y='Humidity9am',

               alpha=0.5, color='grey')

plt.show()
f, ax = plt.subplots(figsize=(15,6))

sns.stripplot(x='Location', y='MaxTemp', data=df, jitter=True)

plt.xticks(rotation=90) # deixa as legendas do eixo x em ângulo reto
f, ax = plt.subplots(figsize=(15,6))

sns.boxplot(x='Location', y='MaxTemp', data=df)

plt.xticks(rotation=90) 
f, ax = plt.subplots(figsize=(15,6))

sns.boxplot(x='Location', y='MinTemp', data=df)

plt.xticks(rotation=90) 
f, ax = plt.subplots(figsize=(15,6))

sns.heatmap(df.corr(), annot=True, fmt='.2f')
plt.figure(figsize=(15,6))



sns.scatterplot(df['Humidity9am'], df['Temp9am'],

               hue=df['RainToday'],

               style=df['RainToday'])
# Dados de 2017

df_2017 = df[df['Year'] == 2017]
# Cidades com maior quantidade de chuva em 2017

df_chuva_2017 = df_2017.nlargest(20, 'Rainfall')
df_20_locations.head()
df['HumidityMean'] = (df['Humidity9am'] + df['Humidity3pm'])/2

df['WindSpeedMean'] = (df['WindSpeed9am'] + df['WindSpeed3pm'])/2

df['PressureMean'] = (df['Pressure9am'] + df['Pressure3pm'])/2
df_20_locations = df.nlargest(20, 'Rainfall')
# Fator que mais afeta o happiness_score por Location



location = list(df_20_locations['Location'].unique())

temp = []

humidity = []

windspeed = []

#pressure = []



for l in location:

    df_novo = df_20_locations[df_20_locations['Location'] == l]

    humidity.append(df_novo['HumidityMean'].mean())

    windspeed.append(df_novo['WindSpeedMean'].mean())

   # pressure.append(df_novo['PressureMean'].mean())

    

# Plotar os valores

plt.figure(figsize=(10,5))

#sns.palplot(sns.color_palette("hls", 8))

sns.barplot(x=humidity, y=location, color='turquoise', label='humidity')

sns.barplot(x=windspeed, y=location, color='palegreen', label='windspeed')

#sns.barplot(x=pressure, y=location, color='teal', label='pressure')



plt.legend()