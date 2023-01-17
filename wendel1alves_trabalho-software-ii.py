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
import seaborn as sns

import matplotlib.pyplot as plt
fifa = pd.read_csv ('/kaggle/input/fifa-18-more-complete-player-dataset/complete.csv')
fifa.info()
fifa.T
print ('Jogadores:', fifa.shape)
fifa.head().T
#fifa_atacante = fifa.loc[fifa['prefers_st'] == 'True'] 

fifa_habilidade = fifa[['name', 'club', 'age', 'overall', 'potential', 'pac', 'sho',

                        'pas', 'dri', 'def', 'rs', 'rw',

                        'rf', 'rm', 'rb', 'rwb', 'st', 'lw', 'cf', 'cam', 'cm', 'lm',

                        'cdm', 'cb', 'lb', 'lwb', 'ls', 'lf', 'gk']]
#Gráfico boxplot da pontuação geral dos jogadores



plt.figure(figsize=(30, 4))

sns.boxplot(x=fifa_habilidade["overall"])
fifa_times = pd.DataFrame (fifa_habilidade.groupby('club').sum())

fifa_times ['qtd_jogadores']= fifa_habilidade['club'].value_counts()
#media geral

fifa_times['media_geral']= fifa_times['overall'] / fifa_times['qtd_jogadores']





#potencial

fifa_times['potencial']= fifa_times['potential'] / fifa_times['qtd_jogadores']



#Ritimo

fifa_times['ritimo']= fifa_times['pac'] / fifa_times['qtd_jogadores']



#Finalização

fifa_times['finalizacao']= fifa_times['sho'] / fifa_times['qtd_jogadores']



#Passe

fifa_times['passe']= fifa_times['pas'] / fifa_times['qtd_jogadores']





#Drible

fifa_times['drible']= fifa_times['dri'] / fifa_times['qtd_jogadores']



#Defesa

fifa_times['defesa']= fifa_times['def'] / fifa_times['qtd_jogadores']



fifa_times.head()
# criando coluna para o nome do clube

fifa_times ['club'] = fifa_times.index.values



# criando um indice



fifa_times['Index']= list(range(0, 647))





#modificando o indice do dataframe

fifa_times.set_index('Index', inplace=True)



fifa_times
 

fifa_habilidade.corr()
#filtro de habilidades por time

times_filter = pd.DataFrame(fifa_times[['club', 'overall', 'potential', 'pac', 'sho',

                        'pas', 'dri', 'def', 'rs', 'rw',

                        'rf', 'rm', 'rb', 'rwb', 'st', 'lw', 'cf', 'cam', 'cm', 'lm',

                        'cdm', 'cb', 'lb', 'lwb', 'ls', 'lf', 'gk', 'media_geral', 'potencial', 'ritimo',

                                        'finalizacao', 'passe', 'drible', 'defesa', 'qtd_jogadores']])



#correlação de habilidades por time

times_filter.corr()
times_filter.describe().T.reset_index()




plt.figure(figsize=(30, 4))

sns.boxplot(x=times_filter["media_geral"])
#10 maiores médias gerais

times_10 = times_filter.sort_values(by=['media_geral'], ascending=False).head(10)

times_10



#Gráfico dos 10 maiores clubes pela média de pontuação geral



#Aumentando a area do gráfico

fig, ax = plt.subplots(figsize=(14,7))



#Criando gráfico de barras

sns.barplot(y='club', x='media_geral', data=times_10)



#Títulos

plt.title ('Top 10 Clubes', fontsize=24)

plt.ylabel('Clube', fontsize=24)

plt.xlabel('Média Geral', fontsize=24)



plt.show()
#A pontuação de todos os jogadores dos 10 clubes mais bem posicionados conforme gráfico anterior



#DataFrame com todas os jogadores de todos os clubes

fifa_jogadores = fifa[['overall', 'name', 'club']]

#filtrando somente os 10 mais bem avaliados

fifa_jogadores = fifa_jogadores[fifa_jogadores['club'].isin(times_10['club'])]



fifa_jogadores

#Gráfico de média dos jogadores dos 10 melhores clubes



#Aumentando a area do gráfico

fig, ax = plt.subplots(figsize=(14,7))



sns.swarmplot(x='club', y='overall', data=fifa_jogadores)



#Títulos

plt.title ('Top 10 Clubes - Jogadores', fontsize=24)

plt.ylabel('Pontuação individual', fontsize=20)

plt.xlabel('Clube', fontsize=20)



plt.xticks(rotation=65)

plt.show()
#clubes entre os 10 mais com melhor potencial



#Aumentando a area do gráfico

fig, ax = plt.subplots(figsize=(14,5))



#Criando gráfico de barras

sns.barplot(y='club', x='potencial', data=times_10)



#Títulos

plt.title ('Top 10 Clubes - Média de Potencial', fontsize=24)

plt.ylabel('Clube', fontsize=16)

plt.xlabel('Nível de Potencial', fontsize=16)



plt.xticks(rotation=65)

plt.show()





###########

#boxplot

plt.figure(figsize=(14, 5))

#sns.boxplot(x=times_10["potential"])

sns.barplot(y='club', x='potential', data=times_10)

#Títulos

plt.title ('Top 10 Clubes - Total de Potencial', fontsize=24)

plt.ylabel('Clube', fontsize=16)

plt.xlabel('Nível de Potencial', fontsize=16)



plt.xticks(rotation=65)

plt.show()

#clubes entre os 10 mais com melhor ritimo



#Aumentando a area do gráfico

fig, ax = plt.subplots(figsize=(14,5))



#Criando gráfico de barras

sns.barplot(y='club', x='ritimo', data=times_10)



#Títulos

plt.title ('Top 10 Clubes - Média de Ritimo', fontsize=24)

plt.ylabel('Clube', fontsize=16)

plt.xlabel('Nível de Ritimo', fontsize=16)



plt.xticks(rotation=65)

plt.show()



###########

#boxplot

plt.figure(figsize=(14, 5))

#sns.boxplot(x=times_10["potential"])

sns.barplot(y='club', x='pac', data=times_10)

#Títulos

plt.title ('Top 10 Clubes - Total de Ritimo', fontsize=24)

plt.ylabel('Clube', fontsize=16)

plt.xlabel('Nível de Ritimo', fontsize=16)



plt.xticks(rotation=65)

plt.show()


#clubes entre os 10 mais com melhor Finalização



#Aumentando a area do gráfico

fig, ax = plt.subplots(figsize=(14,5))



#Criando gráfico de barras

sns.barplot(y='club', x='finalizacao', data=times_10)



#Títulos

plt.title ('Top 10 Clubes - Média de Finalização', fontsize=24)

plt.ylabel('Clube', fontsize=16)

plt.xlabel('Nível de Finalização', fontsize=16)



plt.xticks(rotation=65)

plt.show()



###########

#boxplot

plt.figure(figsize=(14, 5))

#sns.boxplot(x=times_10["potential"])

sns.barplot(y='club', x='sho', data=times_10)

#Títulos

plt.title ('Top 10 Clubes - Total de Finalização', fontsize=24)

plt.ylabel('Clube', fontsize=16)

plt.xlabel('Nível de Finalização', fontsize=16)



plt.xticks(rotation=65)

plt.show()
#clubes entre os 10 mais com melhor Passe



#Aumentando a area do gráfico

fig, ax = plt.subplots(figsize=(14,5))



#Criando gráfico de barras

sns.barplot(y='club', x='passe', data=times_10)



#Títulos

plt.title ('Top 10 Clubes - Média de Passe', fontsize=24)

plt.ylabel('Clube', fontsize=16)

plt.xlabel('Nível de Passe', fontsize=16)



plt.xticks(rotation=65)

plt.show()





###########

#boxplot

plt.figure(figsize=(14, 5))

#sns.boxplot(x=times_10["potential"])

sns.barplot(y='club', x='pas', data=times_10)

#Títulos

plt.title ('Top 10 Clubes - Total de Passe', fontsize=24)

plt.ylabel('Clube', fontsize=16)

plt.xlabel('Nível de Passe', fontsize=16)



plt.xticks(rotation=65)

plt.show()
#clubes entre os 10 mais com melhor Drible



#Aumentando a area do gráfico

fig, ax = plt.subplots(figsize=(14,5))



#Criando gráfico de barras

sns.barplot(y='club', x='drible', data=times_10)



#Títulos

plt.title ('Top 10 Clubes - Média de Drible', fontsize=24)

plt.ylabel('Clube', fontsize=16)

plt.xlabel('Nível de Drible', fontsize=16)



plt.xticks(rotation=65)

plt.show()





###########

#boxplot

plt.figure(figsize=(14, 5))

#sns.boxplot(x=times_10["potential"])

sns.barplot(y='club', x='dri', data=times_10)

#Títulos

plt.title ('Top 10 Clubes - Total de Drible', fontsize=24)

plt.ylabel('Clube', fontsize=16)

plt.xlabel('Nível de Drible', fontsize=16)



plt.xticks(rotation=65)

plt.show()
#clubes entre os 10 mais com melhor Defesa



#Aumentando a area do gráfico

fig, ax = plt.subplots(figsize=(14,5))



#Criando gráfico de barras

sns.barplot(y='club', x='defesa', data=times_10)



#Títulos

plt.title ('Top 10 Clubes - Média de Defesa', fontsize=24)

plt.ylabel('Clube', fontsize=16)

plt.xlabel('Nível de Defesa', fontsize=16)



plt.xticks(rotation=65)

plt.show()



###########

#boxplot

plt.figure(figsize=(14, 5))

#sns.boxplot(x=times_10["potential"])

sns.barplot(y='club', x='def', data=times_10)

#Títulos

plt.title ('Top 10 Clubes - Total de Defesa', fontsize=24)

plt.ylabel('Clube', fontsize=16)

plt.xlabel('Nível de Defesa', fontsize=16)



plt.xticks(rotation=65)

plt.show()
#criando dataset dos 10 melhores times - defesa - Somatória dos jogadores de defesa



fifa_defesa = times_10[['club','rb', 'cb', 'lb', 'qtd_jogadores']]

fifa_defesa['zagueiro'] = fifa_defesa[['rb', 'cb', 'lb']].sum(axis=1)

fifa_defesa['mediazagueiro'] = fifa_defesa['zagueiro'] / fifa_defesa['qtd_jogadores']



#GRÁFICO1

#Aumentando a area do gráfico

fig, ax = plt.subplots(figsize=(14,5))



#Criando gráfico de barras

sns.pointplot(y='mediazagueiro', x='club', data=fifa_defesa)



#Títulos

plt.title ('Top 10 Clubes - Média de Defesa', fontsize=24)

plt.ylabel('Clube', fontsize=16)

plt.xlabel('Nível Zagueiro', fontsize=16)



plt.xticks(rotation=65)

plt.show()





##########



#GRÁFICO2

#Aumentando a area do gráfico

fig, ax = plt.subplots(figsize=(14,5))



#Criando gráfico de barras

sns.pointplot(y='zagueiro', x='club', data=fifa_defesa)



#Títulos

plt.title ('Top 10 Clubes - Total Defesa', fontsize=24)

plt.ylabel('Clube', fontsize=16)

plt.xlabel('Nível Zagueiro', fontsize=16)



plt.xticks(rotation=65)

plt.show()
#criando dataset dos 10 melhores times - Ataque



fifa_ataque = times_10[['club','rs', 'rf', 'st', 'cam', 'ls', 'lf', 'qtd_jogadores']]

fifa_ataque['ataque'] = fifa_ataque[['rs', 'rf', 'st', 'cam', 'ls', 'lf']].sum(axis=1)

fifa_ataque['mediaataque'] = fifa_ataque['ataque'] / fifa_ataque['qtd_jogadores']





#Gráfico1

#Aumentando a area do gráfico

fig, ax = plt.subplots(figsize=(14,5))



#Criando gráfico de barras

sns.pointplot(y='mediaataque', x='club', data=fifa_ataque)



#Títulos

plt.title ('Top 10 Clubes - Média de Ataque', fontsize=24)

plt.ylabel('Clube', fontsize=16)

plt.xlabel('Nível Ataque', fontsize=16)



plt.xticks(rotation=65)

plt.show()



######

#Gráfico2

#Aumentando a area do gráfico

fig, ax = plt.subplots(figsize=(14,5))



#Criando gráfico de barras

sns.pointplot(y='ataque', x='club', data=fifa_ataque)



#Títulos

plt.title ('Top 10 Clubes - Total de Ataque', fontsize=24)

plt.ylabel('Clube', fontsize=16)

plt.xlabel('Nível Ataque', fontsize=16)



plt.xticks(rotation=65)

plt.show()
#criando dataset dos 10 melhores times - Meio



fifa_meio = times_10[['club','rw', 'rm', 'rwb', 'lw', 'cm', 'lm', 'cdm', 'lwb', 'qtd_jogadores']]

fifa_meio['meio'] = fifa_meio[['rw', 'rm', 'rwb', 'lw', 'cm', 'lm', 'cdm', 'lwb']].sum(axis=1)

fifa_meio['mediameio'] = fifa_meio['meio'] / fifa_meio['qtd_jogadores']



#Gráfico1

#Aumentando a area do gráfico

fig, ax = plt.subplots(figsize=(14,7))



#Criando gráfico de barras

sns.pointplot(y='mediameio', x='club', data=fifa_meio)



#Títulos

plt.title ('Top 10 Clubes - Média de Meio de Campo', fontsize=24)

plt.ylabel('Clube', fontsize=16)

plt.xlabel('Nível Meio de Campo', fontsize=16)



plt.xticks(rotation=65)

plt.show()





######

#Gráfico2

#Aumentando a area do gráfico

fig, ax = plt.subplots(figsize=(14,7))



#Criando gráfico de barras

sns.pointplot(y='meio', x='club', data=fifa_meio)



#Títulos

plt.title ('Top 10 Clubes - Total de Meio de Campo', fontsize=24)

plt.ylabel('Clube', fontsize=16)

plt.xlabel('Nível Meio de Campo', fontsize=16)



plt.xticks(rotation=65)

plt.show()