# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib import pylab

import seaborn as sns

import datetime as dt

sns.set_style("whitegrid")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#atribuindo o dataset a uma variavel

data = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')

#.head -> mostra os primeiros registros do dataset

data.head()
#verificação de valor nulo

data.isnull().any()
#Verificando os registros duplicados

sum(data.duplicated())

#Apagando registros duplicados.

data = data.drop_duplicates()
#Quantidade de instalação de apps pagos x gratuítos

data.Type.value_counts()
#Excluindo o registro com valor zero do dataset

data = data.drop(data[data['Type'] == "0"].index)
#Proporção de Apps Pagos e Gratuitos 

data['Type'].value_counts(normalize = True)
#Acertando a colune tipo (Free/Paid)

data['Type_str'] = data['Type']

data['Type'] = data['Type'].replace({'Free' : 1, 'Paid' : 0})

data.Type.value_counts()
#número de aplicativos pagos

len(data[data['Type'] == 0].index)
#número de aplicativos gratuitos

len(data[data['Type'] == 1].index)
#Acertando a colune tipo (Free/Paid)

data['Type_str'] = data['Type']

data['Type'] = data['Type'].replace({1 : 'Free', 0 : 'Paid'})

data.Type.value_counts()
#Contagem por tipo (free or paid) no gráfico

sns.countplot(x = 'Type', data = data)
#Ajustando a coluna de instalações (retirando + e , e transformando em int) 

data['Installs'] = data['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)

data['Installs'] = data['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)

data['Installs'] = data['Installs'].apply(lambda x: int(x))

data.head()
#ajustando as colunas de tamanho, preço e review

data['Size'] = data['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)



data['Size'] = data['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)

data['Size'] = data['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)

data['Size'] = data['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)



data['Size'] = data['Size'].apply(lambda x: float(x))



data['Price'] = data['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))

data['Price'] = data['Price'].apply(lambda x: float(x))



data['Reviews'] = data['Reviews'].apply(lambda x: int(x))



data.head()
#Contagem por classificacao etaria no gráfico

sns.countplot(x = 'Content Rating', data = data)
#ajustando o nome das colunas

data.columns = ['aplicativo',

                 'categoria',

                 'nota',

                 'visualizacoes',

                 'tamanho',

                 'instalacoes',

                 'tipo',

                 'preco',

                 'classificacao_etaria',

                 'genero',

                 'ultima_atualizacao',

                 'versao_atual',

                 'versao_android',

                 'tipo_numero']

data.info()
#contagem de aplicativos por classificacao_etaria

data.classificacao_etaria.value_counts()
instalacoes_tipo = data.groupby('classificacao_etaria').sum()['instalacoes']

instalacoes_tipo.plot.pie(figsize=(5,5),title = 'Instalacoes por classificacao etaria')
data.categoria.value_counts()
#mostrar os aplicativos mais caros que 150 reais

data[['categoria', 'aplicativo','preco']][data.preco > 150]
#DISPERSAO DOS PREÇOS DOS APLICATIVOS POR CATEGORIA

subset_data = data[data.categoria.isin(['GAME', 'FAMILY', 'PHOTOGRAPHY', 'MEDICAL', 'TOOLS', 'FINANCE',

                                 'LIFESTYLE','BUSINESS'])]

sns.set_style('darkgrid')

fig, ax = plt.subplots()

fig.set_size_inches(15, 8)

subset_data = subset_data[subset_data.preco<100]

p = sns.stripplot(x="preco", y="categoria", data=subset_data, jitter=True, linewidth=1)

title = ax.set_title('App pricing trend across categories')

#mostrar o aplicativo mais caro do montante acima:

data[['categoria', 'aplicativo','preco', 'instalacoes']][data.preco < 100][data.preco > 85]
#CONTAGEM DE APLICATIVOS DAS TOP5 CATEGORIAS

subset_data = data[data.categoria.isin(['FAMILY', 'GAME', 'TOOLS', 'MEDICAL','BUSINESS'])]



sns.countplot(x = 'categoria', data = subset_data)
data.head()