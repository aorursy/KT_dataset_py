# Importando as bibliotecas



import pandas as pd

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import math
# Carregando o arquivo e verificando os dados iniciais



df = pd.read_csv('../input/vgsales.csv')



df.head()
# Renomeando as colunas



df.rename(columns={'Rank':'ID',

                   'Name':'Nome',

                   'Platform':'Plataforma',

                   'Year':'Ano',

                   'Genre':'Gênero',

                   'Publisher':'Editora',

                   'NA_Sales':'Vendas América do Norte',

                   'EU_Sales':'Vendas EUA',

                   'JP_Sales':'Vendas Japão',

                   'Other_Sales':'Vendas Outros Países',

                   'Global_Sales':'Total Vendas'

                   }, inplace=True)
# Avaliando a mudança de nome das colunas

df.head()
# Verificando missing, tamanhos e tipos

df.info()
# Avaliando o quantitativo de cada coluna

df.count()
# Avaliando a quantidade de missing por coluna

df.isnull().sum()
# Alterando os missings

df.update(df['Editora'].fillna('Não cadastrado'))



df.head()
# Resumo dos Dados

df.describe()
# Dropando os valores Nan

df.dropna
# Identificando os tipos de Gêneros

df['Gênero'].value_counts()
# Quantitativo por Gênero

df['Gênero'].value_counts().plot.bar()
# Top 5 Maiores Totais de Vendas

df.nlargest(5, 'Total Vendas')[['Gênero', 'Ano', 'Total Vendas']].style.hide_index()
# Top 5 Maiores Vendas na América do Norte

df.nlargest(5, 'Vendas América do Norte')[['Gênero', 'Ano', 'Vendas América do Norte']].style.hide_index()
# Top 5 Maiores Vendas nos EUA

df.nlargest(5, 'Vendas EUA')[['Gênero', 'Ano', 'Vendas EUA']].style.hide_index()
# Top 5 Maiores Vendas no Japão

df.nlargest(5, 'Vendas Japão')[['Gênero', 'Ano', 'Vendas Japão']].style.hide_index()
# Top 5 Maiores Vendas em Outros Países

df.nlargest(5, 'Vendas Outros Países')[['Gênero', 'Ano', 'Vendas Outros Países']].style.hide_index()
# Média por Ano de Total de Vendas

df.groupby('Ano')['Total Vendas'].mean()
plt.figure(figsize=(30,5))

sns.pointplot(x='Ano', y='Total Vendas', data=df, color='green')

plt.title('Quantidade de Vendas por Ano')

plt.grid(True, color='grey')
f, ax = plt.subplots(figsize=(25,12))

sns.heatmap(df.corr(), annot=True, fmt='.2f', linecolor='black', ax=ax, lw=.7)
plt.figure(figsize=(25,18))

sns.regplot(x="Vendas América do Norte", y="Total Vendas", data=df, x_estimator=np.mean)
# Top 10 Maiores Vendas por Plataforma

df.nlargest(10, 'Total Vendas')[['Plataforma', 'Ano', 'Total Vendas']].style.hide_index()
# Criando Dumies

df_plat = pd.get_dummies(df,columns=['Plataforma'])
# Total de Vendas de Jogos do Wii por Ano

plt.figure(figsize=(25,10))

sns.lineplot (x='Ano', y='Total Vendas', hue= 'Plataforma_Wii', data=df_plat)

plt.xticks(rotation=90)
# Total de Vendas de Jogos do GB por Ano

plt.figure(figsize=(25,10))

sns.lineplot (x='Ano', y='Total Vendas', hue= 'Plataforma_GB', data=df_plat)

plt.xticks(rotation=90)
# Total de Vendas de Jogos do PS2 por Ano

plt.figure(figsize=(25,10))

sns.lineplot (x='Ano', y='Total Vendas', hue= 'Plataforma_PS2', data=df_plat)

plt.xticks(rotation=90)
# Total de Vendas de Jogos do DS por Ano

plt.figure(figsize=(25,10))

sns.lineplot (x='Ano', y='Total Vendas', hue= 'Plataforma_DS', data=df_plat)

plt.xticks(rotation=90)
# Total de Vendas de Jogos do NES por Ano

plt.figure(figsize=(25,10))

sns.lineplot (x='Ano', y='Total Vendas', hue= 'Plataforma_NES', data=df_plat)

plt.xticks(rotation=90)