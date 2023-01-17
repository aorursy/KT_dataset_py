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

%matplotlib inline
# Importando o arquivo CSV

dados = pd.read_csv('../input/googleplaystore.csv')
# Explorando informações básicas dos banco de dados 

#quantidade de linhas e colunas

dados.shape



#O banco de dados contem 13 colunas e 10841 linhas.
# Analisando os nomes das colunas 

dados.columns
# tipos de variaveis do meu banco de dados



dados.info()
# Visualizando as 3(três) primeiras linhas do banco de dados

dados.head(3)
#Verificando valores ausentes.

dados.isnull().sum()
#verificamos que a linha 10472 apresenta inconsistencia.

dados.loc[dados['App']=="Life Made WI-Fi Touchscreen Photo Frame"]
#corrigindo linha 10472

dados.loc[dados['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Category']='PHOTOGRAPHY'

dados.loc[dados['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Rating']='1.9'

dados.loc[dados['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Reviews']='19'

dados.loc[dados['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Size']='3.0M'

dados.loc[dados['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Installs']='1,000+'

dados.loc[dados['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Type']='Free'

dados.loc[dados['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Price']='0'

dados.loc[dados['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Content Rating']='Everyone'

dados.loc[dados['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Genres']='Photography'

dados.loc[dados['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Last Updated']='February 11, 2018'

dados.loc[dados['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Current Ver']='1.0.19'

dados.loc[dados['App']=="Life Made WI-Fi Touchscreen Photo Frame",'Android Ver']='4.0 and up'


#excluindo algumas colunas que não são de interesse:



#App: Aplicativo

#Last Updated: "Ultima atualização"

#Current Ver: "Versão Atual"



dados.drop(["App",'Last Updated',"Current Ver"], axis = 1, inplace = True)
#banco de dados com colunas reduzidas

dados.columns
#Analise Univariada: 'Category'(Categoria à qual o aplicativo pertence)

dados["Category"].value_counts()

#verificamos que a maioria dos aplicativos são da categoria(FAMILY). 
#Analise Univariada: 'Rating'(Avaliação - Classificação geral do usuário do aplicativo)

#renomeando a coluna'Rating' para Avaliação e alterando-a para ser do tipo numérico

dados['Avaliacao'] = dados['Rating'].astype('float64')

#Analise Univariada: 'Rating'(Avaliação - Classificação geral do usuário do aplicativo)

#medidas resumo estatistico variavel avaliação

dados["Avaliacao"].describe()



#verificamos que média da avaliação dos Aplicativos é de 4,19 pontos numa escala de 1 a 5.
#histograma das avaliações 

plt.figure(figsize=(8,8))

ax=sns.distplot(dados['Avaliacao'],bins=40,color="blue")

ax.set_xlabel("Avaliacao")

ax.set_ylabel(" Frequencia")

ax.set_title("Distribuição - Avaliação")
#Analise Univariada -Type: (Tipo de Licença)

#grafico de barras para tipo de licença, Free(Livre) e Paid(Pago)

sns.countplot(x="Type", data=dados)
#Analise Univariada - 'Genres': (Gêneros dos App)



dados['Genres'].value_counts()



#verificamos que a maioria 842 Aplicativos são do gênero (Tools: Ferramentas)
#Analise Grafica Boxplot para a Categoria x Avaliação 

plt.figure(figsize=(18,7))

sns.boxplot(dados['Category'],dados['Avaliacao'])

#alterando legenda eixo x

plt.xticks(rotation=90)



#vericamos atráves da mediana que categoria (EVENTS) apresenta melhores avaliações gerais dos usuários.

#Nota-se ainda que existe valores discrepantes para a maioria das categorias, especificamente para valores  mais baixos para avalialção
#Analise Grafica Boxplot para a Tipo de Licença x Avaliação 

plt.figure(figsize=(18,7))

sns.boxplot(dados['Type'],dados['Avaliacao'])

#alterando legenda eixo x

plt.xticks(rotation=90)



#de acordo com o boxplot não existe indícios de diferença entre as avaliações de aplicativos com licença Livre e os app com licença Paga.   