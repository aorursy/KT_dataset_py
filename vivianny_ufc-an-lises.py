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
data = pd.read_csv('/kaggle/input/ufcdata/data.csv')
#Analisando a base

data.info()
#Visualizando o dataframe

data.head(10)
#Selecionando apenas as variáveis necessárias para a analise 

df = pd.DataFrame(data,columns=['R_fighter','B_fighter', 'R_age', 'date','weight_class','location','Winner','R_wins','B_wins', 'B_age', 'no_of_rounds'])
#Verificando as informações do novo dataframe df

df.info()
df['date'].min()
df['date'].describe()
#Verificando se existem valores nulos

pd.isnull(df)
df.isnull().sum
#Imputação de idade pela média nos lutadores do time Vermelho

from scipy.stats import mode

mode(df['R_age'])

mode(df['R_age']).mode[0]

df['R_age'].fillna(df['R_age'].mean(), inplace=True)
#Imputação de idade pela média nos lutadores do time Azul

from scipy.stats import mode

mode(df['B_age'])

mode(df['B_age']).mode[0]

df['B_age'].fillna(df['B_age'].mean(), inplace=True)
#Verificando se os dados foram imputados

df.info()
#Realizando analise descritiva após a imputação dos dados

df.describe()
#converter o campo date

df['date']= pd.to_datetime(df['date'])
#criando novas colunas com base na data

df['day'] = df['date'].dt.day

df['month'] = df['date'].dt.month

df['year']= df['date'].dt.month
#Realizando uma análise descritiva no dataframe

df.head()
#Média de vitórias por mês, lutadores do time vermelho

df.groupby('month')['R_wins'].mean()
import matplotlib.pyplot as plt

plt.plot(df.groupby('month')['R_wins'].mean())

plt.xlabel('Ano')

plt.ylabel('Média de Vitórias')

plt.title('Média de vitórias por mês Lutadores Vermelhos')
#Média de vitórias por mês, lutadores do time Azul

df.groupby('month')['B_wins'].mean()
plt.plot(df.groupby('month')['B_wins'].mean())

plt.xlabel('Ano')

plt.ylabel('Média de Vitórias')

plt.title('Média de vitórias por mês Lutadores Azul')
import seaborn as sns
df.corr()
#GRÁFICO DE CORRELAÇÃO JUNTAMENTE COM A DISTRIBUIÇÃO - Vitórias X Idade dos lutadores

Correlacao= df[['R_wins','no_of_rounds','R_age']]

Correlacao

sns.jointplot(data=Correlacao,y='R_wins',x='R_age',kind='reg', color='r')

#GRÁFICO DE CORRELAÇÃO JUNTAMENTE COM A DISTRIBUIÇÃO - Vitórias X Idade dos lutadores

Correlacao= df[['B_wins','no_of_rounds','B_age']]

Correlacao

sns.jointplot(data=Correlacao,y='B_wins',x='B_age',kind='reg', color='b')
sns.boxplot(x='year', y='B_wins', data=df)
sns.boxplot(x='year', y='R_wins', data=df)
#Box plote mês Vitórias

fig, ax1 = plt.subplots(1, 1, figsize=(15, 4))

sns.boxplot('month', 'R_wins', data=df, ax=ax1)
fig, ax1 = plt.subplots(1, 1, figsize=(15, 4))

sns.boxplot('month', 'B_wins', data=df, ax=ax1)
sns.pairplot(df, height=3, vars=["B_wins", "R_wins"])
#Analise do número de vitorias entre os lutadores do time Vermelho e Azul 

fig, ax = plt.subplots(1, figsize=(15, 5))

sns.distplot(df['R_wins'])

sns.distplot(df['B_wins'])

plt.legend(labels=['Azul','Vermelho'], loc="upper right")
#Gráfico de barras Lutas por idade lutadores azul

sns.barplot(x='year',y='B_wins', data=df)

plt.title('vitórias por ano lutadores Azuis')

plt.xlabel('Ano')

plt.ylabel('Número de Vitórias')
#Gráfico de barras Lutas por idade lutadores vermelho

sns.barplot(x='year',y='R_wins', data=df)

plt.title('vitórias por ano lutadores Vermelhos')

plt.xlabel('Ano')

plt.ylabel('Número de Vitórias')