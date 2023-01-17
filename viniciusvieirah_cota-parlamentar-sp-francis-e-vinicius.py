import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns
cota_parlamentar = pd.read_csv('../input/cota_parlamentar_sp.csv', delimiter=';')
cota_parlamentar.dtypes
cota_parlamentar.shape
cota_parlamentar.head()
correlation = cota_parlamentar[['nudeputadoid','nummes','vlrdocumento',]]

corr = correlation.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

           cmap=sns.diverging_palette(220, 10, as_cmap=True))

plt.show()
soma = cota_parlamentar[['txtfornecedor', 'vlrdocumento']].groupby('txtfornecedor').sum().reset_index()

soma = soma.sort_values(by='vlrdocumento')



soma['vlrdocumento'] = soma['vlrdocumento'] / 1000000



sns.set()

plt.figure(figsize=(16, 8))

sns.barplot(x='txtfornecedor', y='vlrdocumento', data=soma, order=soma['txtfornecedor'].tail(7))

plt.xlabel('')

plt.ylabel('Gasto (milhões)')

plt.title('Principais gastos no período 2010-2018', fontsize=18)

plt.show()
soma = cota_parlamentar[['txnomeparlamentar', 'sgpartido', 'vlrdocumento']].groupby(['txnomeparlamentar', 'sgpartido']).sum().reset_index()

soma = soma.sort_values(by='vlrdocumento')



soma['vlrdocumento'] = soma['vlrdocumento'] / 1000000

soma['parlamentar'] = soma['txnomeparlamentar'] + ' (' + soma['sgpartido'] + ')'



soma.tail()
sns.set()

plt.figure(figsize=(16, 8))

sns.barplot(x='parlamentar', y='vlrdocumento', label='sgpartido', data=soma, order=soma['parlamentar'].tail(6))

plt.xlabel('')

plt.ylabel('Gasto (milhões)')

plt.title('Parlamentares que mais gastaram no período 2010-2018', fontsize=18)

plt.show()
dados = cota_parlamentar.groupby(['nummes']).sum().reset_index()



dados['vlrdocumento'] = dados['vlrdocumento'] / 1000000



sns.set()

fig = plt.figure(figsize=(16, 8))

ax = fig.add_subplot(111)

sns.lineplot(x='nummes', y='vlrdocumento', data=dados, ax=ax)

plt.title('Gastos totais ao longo do ano\n', fontsize=18)

plt.xlabel('Mês', fontsize=13)

plt.ylabel('Valor gasto (milhões)', fontsize=13)

plt.show()
sns.set()

fig = plt.figure(figsize=(16, 8))

ax = fig.add_subplot(111)

sns.lineplot(x='nummes', y='vlrdocumento', data=cota_parlamentar, ax=ax)

plt.title('Gastos individuais ao longo do ano\n', fontsize=18)

plt.xlabel('Mês', fontsize=13)

plt.ylabel('Valor gasto', fontsize=13)

plt.show()
soma = cota_parlamentar[['sgpartido', 'vlrdocumento']].groupby('sgpartido').sum().reset_index()

soma = soma.sort_values(by='vlrdocumento')



soma['vlrdocumento'] = soma['vlrdocumento'] / 1000000



sns.set()

plt.figure(figsize=(16, 8))

sns.barplot(x='sgpartido', y='vlrdocumento', data=soma)

plt.xlabel('Partido')

plt.ylabel('Gasto (milhões)')

plt.title('Gasto total por partido / 2010-2018', fontsize=18)

plt.show()
soma = cota_parlamentar[['sgpartido', 'vlrdocumento']].groupby('sgpartido').sum().reset_index()

soma = soma.sort_values(by='vlrdocumento')



aux = cota_parlamentar[cota_parlamentar.sgpartido.isin(soma['sgpartido'].tail(5))]



aux = aux.groupby(['sgpartido', 'numano']).sum().reset_index()

aux['vlrdocumento'] = aux['vlrdocumento'] / 1000000



sns.set()

plt.figure(figsize=(16, 8))

sns.lineplot(x='numano', y='vlrdocumento', hue='sgpartido', data=aux)

plt.title('Os 5 partidos que mais gastaram ao longo últimos anos\n', fontsize=18)

plt.xlabel('Ano', fontsize=13)

plt.ylabel('Valor no ano (milhões)', fontsize=13)

#plt.legend(loc=1, title='Ano', title_fontsize=13)

plt.show()
soma = cota_parlamentar[['sgpartido', 'vlrdocumento']].groupby('sgpartido').sum().reset_index()

soma = soma.sort_values(by='vlrdocumento')



aux = cota_parlamentar[cota_parlamentar.sgpartido.isin(soma['sgpartido'].tail(5))]



sns.set()

fig = plt.figure(figsize=(16, 8))

ax = fig.add_subplot(111)

sns.lineplot(x='numano', y='vlrdocumento', hue='sgpartido', data=aux, ax=ax)

ax.legend().texts[0].set_text('Partido')

plt.title('Gastos médio ao longo dos anos\n', fontsize=18)

plt.xlabel('Ano', fontsize=13)

plt.ylabel('Valor gasto', fontsize=13)

plt.show()
soma = cota_parlamentar[['sgpartido', 'vlrdocumento']].groupby('sgpartido').sum().reset_index()

soma = soma.sort_values(by='vlrdocumento')



sns.set()

plt.figure(figsize=(16, 10))

sns.boxenplot(x='sgpartido', y='vlrdocumento', hue='numano', data=cota_parlamentar, order=soma['sgpartido'].tail(5))

plt.title('Distribuição dos gastos dos 5 maiores partidos\n', fontsize=18)

plt.xlabel('Partido', fontsize=13)

plt.ylabel('Valores gastos', fontsize=13)

plt.legend(loc=1, title='Ano', title_fontsize=13)

plt.show()
violin = cota_parlamentar.copy()

violin['vlrdocumento'] = violin['vlrdocumento']/1000000

plt.figure(figsize=(50, 30))

sns.set_style('whitegrid')

sns.violinplot(x='sgpartido', y='vlrdocumento', cut=0, scale="count", data=violin.sort_values(by=['sgpartido']))

plt.show()