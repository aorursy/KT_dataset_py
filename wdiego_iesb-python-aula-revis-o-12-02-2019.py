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
# Carregando os datasets

dados_2015 = pd.read_csv("../input/2015.csv")

dados_2016 = pd.read_csv("../input/2016.csv")

dados_2017 = pd.read_csv("../input/2017.csv")
# Olhando o tamanho dos datasets

print('2015: ', dados_2015.shape)

print('2016: ', dados_2016.shape)

print('2017: ', dados_2017.shape)
# Olhando os dados

dados_2015.head()
dados_2016.head()
dados_2017.head()
# Verificando o nome das colunas

dados_2015.columns
dados_2016.columns
dados_2017.columns
# Vamos manter apenas as colunas mais importantes

dados_2015.drop(columns='Standard Error', inplace=True)

dados_2015.columns
dados_2016.drop(columns=['Lower Confidence Interval', 'Upper Confidence Interval'], inplace=True)

dados_2016.columns
dados_2017.drop(columns=['Whisker.high', 'Whisker.low'], inplace=True)

dados_2017.columns
# Novos nomes de colunas

nomes_2015_2016 = ['country', 'region', 'happiness_rank', 'happiness_score', 'economy', 'family', 'health', 'freedom', 'corruption', 'generosity', 'dystopia']

nomes_2017 = ['country', 'happiness_rank' , 'happiness_score', 'economy', 'family', 'health', 'freedom', 'generosity', 'corruption', 'dystopia']



# Alterando os nomes das colunas

dados_2015.columns = nomes_2015_2016

dados_2016.columns = nomes_2015_2016

dados_2017.columns = nomes_2017
# Os 5 países menos felizes países em 2015

dados_2015.nsmallest(5, 'happiness_score')
# Os 5 países menos felizes países em 2016

dados_2016.nlargest(5, 'happiness_rank')
# Os 5 países menos felizes países em 2017

dados_2016.nlargest(5, 'happiness_rank')
# Qual a posição do Brasil a cada ano?

print('========== Brasil =============')

pos_2015 = dados_2015[dados_2015['country'] == 'Brazil']['happiness_rank']

pos_2016 = dados_2016[dados_2016['country'] == 'Brazil']['happiness_rank']

pos_2017 = dados_2017[dados_2017['country'] == 'Brazil']['happiness_rank']



print('2015: ', pos_2015.iloc[0])

print('2016: ', pos_2016.iloc[0])

print('2017: ', pos_2017.iloc[0])
# Importando as bibliotecas gráficas

import matplotlib.pyplot as plt

import seaborn as sns
# Vamos relacionar happiness_score e PIB

p1 = dados_2015.plot(title='2015', kind='scatter', x='economy', y='happiness_score', alpha=0.5, color='red')

p2 = dados_2016.plot(title='2016', kind='scatter', x='economy', y='happiness_score', alpha=0.5, color='green')

p3 = dados_2017.plot(title='2017', kind='scatter', x='economy', y='happiness_score', alpha=0.5, color='blue')



p1.set_xlabel('Economy')

p1.set_ylabel('Happiness Score')



p2.set_xlabel('Economy')

p2.set_ylabel('Happiness Score')



p3.set_xlabel('Economy')

p3.set_ylabel('Happiness Score')



plt.show()
# Vamos relacionar happiness_score e regiao

sns.stripplot(x='region', y='happiness_score', data=dados_2015, jitter=True)

plt.xticks(rotation=90)
# Correlação

dados_2015.corr()
# Vamos plotar as correlações

f, ax = plt.subplots(figsize=(15,6))

sns.heatmap(dados_2015.corr(), annot=True, fmt='.2f', ax=ax, linecolor='black', lw=.7)
# Qual o fator que mais afeta o happiness_score por região / 2015

region = list(dados_2015['region'].unique())

economy = []

family = []

health = []

freedom = []

corruption = []



for r in region:

    df = dados_2015[dados_2015['region'] == r]

    economy.append(df['economy'].mean())

    family.append(df['family'].mean())

    health.append(df['health'].mean())

    freedom.append(df['freedom'].mean())

    corruption.append(df['corruption'].mean())

    

# Plotar os valores

plt.figure(figsize=(15,6))

sns.barplot(x=economy, y=region, color='pink', label='economy')

sns.barplot(x=family, y=region, color='red', label='family')

sns.barplot(x=health, y=region, color='green', label='health')

sns.barplot(x=freedom, y=region, color='blue', label='freedom')

sns.barplot(x=corruption, y=region, color='orange', label='corruption')



plt.legend()
# Criando uma coluna nova happy_quality, que irá conter o grau de felicidade

# de acordo com os quartis q3, q2 e q1

q3 = dados_2017['happiness_score'].quantile(0.75)

q2 = dados_2017['happiness_score'].quantile(0.5)

q1 = dados_2017['happiness_score'].quantile(0.25)



happy_quality = []



for i in dados_2017['happiness_score']:

    if i >= q3:

        happy_quality.append('Muito Alto')

    elif i < q3 and i >= q2:

        happy_quality.append('Alto')

    elif i< q2 and i >= q1:

        happy_quality.append('Normal')

    else:

        happy_quality.append('Baixo')



dados_2017['happy_quality'] = happy_quality
plt.figure(figsize=(7,7))

sns.boxplot(dados_2017['happy_quality'], dados_2017['economy'])
plt.figure(figsize=(7,7))

sns.swarmplot(dados_2017['happy_quality'], dados_2017['economy'])
plt.figure(figsize=(10,10))

sns.scatterplot(dados_2017['freedom'], dados_2017['economy'], hue=dados_2017['happy_quality'], style=dados_2017['happy_quality'])