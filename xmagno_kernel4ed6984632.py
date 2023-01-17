# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Set Pandas display options

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



#Graficos

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.style.use('ggplot')



from datetime import datetime





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dados = pd.read_csv('../input/sorteiosmegasena/sorteios.csv')

dados.head(5)
dados.shape
dados.dtypes
dados.isna().sum()
colunas_valores=['Acumulado_Mega_da_Virada','Arrecadacao_Total', 'Estimativa_Prêmio', 'Valor_Acumulado', 'Rateio_Quadra', 'Rateio_Sena','Rateio_Quina']



for col in colunas_valores:

    dados[col] = dados[col].str.replace(".", "",regex=True).str.replace(",", ".")



#Convertendo todos para float

dados[colunas_valores] = dados[colunas_valores].fillna(0).astype(float)

assert dados[colunas_valores].dtypes.all() == np.float64

dados.dtypes

dados.describe()
#dados.iloc[:,2].head(5)

# Vamos criar uma coluna para armazenar a data e converte-la para datetime

dados['data_sorteio_conv'] = dados.iloc[:,2]

dados.data_sorteio_conv = pd.to_datetime(dados.data_sorteio_conv)



# Vamos quebrar a data em Dia, Mês e Ano

dados['day']   = dados.data_sorteio_conv.dt.day

dados['month'] = dados.data_sorteio_conv.dt.month 

dados['year']  = dados.data_sorteio_conv.dt.year

# Vamos criar um dataframe para analisar os sorteios que tiveram ganhadores

dados_ganhadores = dados[dados['Acumulado'] == 'NÃO']

dados_ganhadores.head()
dados[dados.columns[3:9]].plot.density()
import missingno           as msno



# Visão geral do dataframe

msno.matrix(df=dados.iloc[:,0:dados.shape[1]], figsize=(20, 5), color=(0.42, 0.1, 0.05))
dados.isnull().sum()
# Removendo colunas Cidade e UF

dados = dados.drop(['Cidade', 'UF','Unnamed: 22'], axis=1)

msno.matrix(df=dados.iloc[:,0:dados.shape[1]], figsize=(20, 5), color=(0.42, 0.1, 0.05))
# Ganhadores X Volume do Prêmio por Estado 



ax = dados_ganhadores.groupby(['UF'])['Ganhadores_Sena'].agg('sum').sort_values(ascending=False).plot(kind='bar', title='Ganhadores por Estado', figsize=(15,5), fontsize=12, legend=True, position=1, color='gray')

dados_ganhadores.groupby(['UF'])['Rateio_Sena'].agg('sum').sort_values(ascending=False).plot(kind='bar', ax=ax, secondary_y=True, legend=True, position=0, color='blue')

# Ganhadores X Mes



dados_ganhadores.groupby(['month'])['Ganhadores_Sena'].agg('sum').plot(kind='bar', title='Ganhadores X Mês', fontsize=12, figsize=(15,5), legend=True, color='gray')
# Numero de ganhadores por ano

dados_ganhadores['year'].value_counts()
sns.set_palette("colorblind")



fig, ax = plt.subplots(figsize=(12,4))

ax = sns.countplot(x="year", data=dados_ganhadores)

plt.title("Numero de ganhadores por ano")

plt.show()
ganhadores_ano = dados_ganhadores.groupby("year")["month"].value_counts().to_frame("Ganhadores").reset_index()

ganhadores_ano = ganhadores_ano.pivot("year", "month")

ganhadores_ano
fig, ax = plt.subplots(figsize=(12,5))



ganhadores_ano.plot(kind="barh", stacked=True, ax=ax)



plt.title("Ganhadores  Mes/Ano", fontsize=18)

plt.xlabel("N° Ganhadores")

plt.ylabel("Mes/Ano")



ax.legend(sorted(dados_ganhadores['month'].unique().tolist()))

plt.tight_layout()

plt.show()
#mais_sorteadas = dados[dados.columns[3:9]].sum(axis=1)

df = dados

dezenas = pd.DataFrame(df['1ª Dezena'].tolist() + df['2ª Dezena'].tolist() + df['3ª Dezena'].tolist() + df['4ª Dezena'].tolist() + df['5ª Dezena'].tolist() + df['6ª Dezena'].tolist(), columns=['numeros'])

dezenas['numeros'].value_counts().sort_values(ascending=False).head(10).plot(kind='barh', title='As dez dezenas mais sorteadas em todos os jogos', figsize=(10,5), fontsize=12, legend=True, color='red')
tops=dezenas['numeros'].value_counts().sort_values(ascending=False).head(10).index.array

#dados_ganhadores.groupby(['year','1ª Dezena'])['1ª Dezena','2ª Dezena'].count()

#dados.columns[3:9].array