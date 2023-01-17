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
# Obtendo dados a serem analisados

ocs = pd.read_csv('../input/mri-and-alzheimers/oasis_cross-sectional.csv')
# Visualizando os dados 

ocs.head(5)
# Compreendendo os quantitativos gerais e tipos de dados.

ocs.info()
# Compreendendo as pricipais medidas, assim como as suas distribuições totais.

ocs.describe()
# Usando o seaborn

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt
ocs['Age'].hist(bins=40)
sns.distplot(ocs['Age'].dropna())
# Renomeando a coluna 'M/F' para 'Genero'

ocs.rename(columns={'M/F': 'Genero'}, inplace=True)
# A análise foi feita predominantemente em pessoas de gênero do sexo feminino.

ocs['Genero'].value_counts()
# Vamos entender em qual gênero há maior incidência de demência.

sns.distplot(ocs[ocs['Genero'] == 'F']['MMSE'].dropna(), label='Feminino')

sns.distplot(ocs[ocs['Genero'] == 'M']['MMSE'].dropna(), label='Masculino')

plt.legend()
# Média dos valores de MMSE entre homens e mulheres.

ocs.groupby('Genero')['MMSE'].mean()
sns.boxplot(x='Genero', y='MMSE', data=ocs)
# eTIV - Volume Estimado Intracraneano

sns.distplot(ocs[ocs['Genero'] == 'F']['eTIV'].dropna(), label='Feminino')

sns.distplot(ocs[ocs['Genero'] == 'M']['eTIV'].dropna(), label='Masculino')

plt.legend()
# Media do índice de Volume Intracraneano agrupadas por Gênero

ocs.groupby('Genero')['eTIV'].mean()
# nWBV - Volume Total do Cérebro

sns.distplot(ocs[ocs['Genero'] == 'F']['nWBV'].dropna(), label='Feminino')

sns.distplot(ocs[ocs['Genero'] == 'M']['nWBV'].dropna(), label='Masculino')

plt.legend()
# Média dos valores de nWBV entre homens e mulheres.

ocs.groupby('Genero')['nWBV'].mean()
sns.pairplot(x_vars='nWBV' ,y_vars='eTIV' ,data=ocs , height=5, hue='Genero')
sns.barplot(x='nWBV' ,y='Age' ,data=ocs)


sns.pairplot(x_vars='nWBV' ,y_vars='MMSE' ,data=ocs , height=5, hue='Age')
# Verificando o quantitativo de pessoas com algum grau de classificação de demência.

ocs['CDR'].value_counts()
# Criar novo dataframe somente com quem possui indicação de quadro de demência.

deme = ocs[(ocs['CDR'] > 0.0) & (ocs['MMSE'] <= 24)]
# Visualizando as primeiras 5 linhas do novo dataframe.

deme.head(5)
# Visualizando principais medidas deste grupo de pessoas.

deme.describe()
# Valores máximos de classificação de demência por Status Sócio Econômico.

ocs.groupby('SES')['CDR'].max().sort_values(ascending=False)
# Valores máximos de classificação de demência por Grau de Instrução.

ocs.groupby('Educ')['CDR'].max().sort_values(ascending=False)
def plot_corr(corr):

    # Cortaremos a metade de cima pois é o espelho da metade de baixo

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask, 1)] = True

    sns.heatmap(corr, mask=mask, cmap='RdBu', square=True, linewidths=.5)

# Calculando a correlação

corr = ocs.corr() 

plot_corr(corr)
sns.barplot(x='Educ' ,y='SES' ,data=ocs)