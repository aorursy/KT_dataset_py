import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline

import os



print(os.listdir("../input"))

df = pd.read_csv('../input/BrFlights2/BrFlights2.csv', encoding='latin1')
df.columns = ['Voos', 'Companhia_Aerea', 'Codigo_Tipo_Linha','Partida_Prevista','Partida_Real','Chegada_Prevista','Chegada_Real','Situacao_Voo','Codigo_Justificativa','Aeroporto_Origem','Cidade_Origem','Estado_Destino','Pais_Origem','Aeroporto_Destino','Cidade_Destino','Estado_Destino','Pais_Destino','LongDest','LatDest','LongOrig','LatOrig']
df.head()
df.shape
df.isnull().sum()
df.info()
df['Codigo_Tipo_Linha'].value_counts().plot.bar()
plt.figure(figsize=(15,6))

plot = df.Companhia_Aerea.value_counts().head(20).plot(kind="bar")

plot.set_title("Top Cias Aéreas")
plt.figure(figsize=(15,6))

plot = df[df.Codigo_Tipo_Linha == 'Nacional'].Companhia_Aerea.value_counts().head(15).plot(kind="bar", )

plot.set_title("Top CIA Aerea Nacional")
plt.figure(figsize=(15,6))

plot = df[df.Codigo_Tipo_Linha == 'Regional'].Companhia_Aerea.value_counts().head(15).plot(kind="bar")

plot.set_title("Top CIA Aerea Regional")
plt.figure(figsize=(15,6))



plot = df[df.Codigo_Tipo_Linha == 'Internacional'].Companhia_Aerea.value_counts().head(15).plot(kind="bar")

plot.set_title("Top CIA Aerea Internacional")
df['Situacao_Voo'].value_counts().plot.bar()
plt.figure(figsize=(10,6))

plot = df[df.Situacao_Voo == 'Cancelado'].Companhia_Aerea.value_counts().head(10).plot(kind="bar")

plot.set_title("CIA aerea com mais voos cancelados")
plt.figure(figsize=(10,6))

plot = df.Aeroporto_Origem.value_counts().head(10).plot(kind="bar")

plot.set_title("Top aeroportos origem")
plt.figure(figsize=(10,6))

plot = df.Cidade_Origem.value_counts().head(10).plot(kind="bar")

plot.set_title("Top cidades origem")
plt.figure(figsize=(10,6))

plot = df.Aeroporto_Destino.value_counts().head(10).plot(kind="bar")

plot.set_title("Top aeroportos destino")
plt.figure(figsize=(10,6))

plot = df.Cidade_Destino.value_counts().head(10).plot(kind="bar")

plot.set_title("Top cidades destino")