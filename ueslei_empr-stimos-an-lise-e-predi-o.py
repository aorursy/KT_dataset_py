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



# Biblioteca para gráficos interativos

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



import cufflinks as cf

cf.go_offline()

# Para Notebooks

init_notebook_mode(connected=True)
# Carregando a base

emprestimo = pd.read_csv('../input/dados_clube_emprestimo.csv')
# Verificando a quantidade de colunas por TIPO

emprestimo.dtypes.value_counts()
# As 5 menores taxas de juros

emprestimo.nsmallest(5,'int.rate')
# As 5 maiores taxas de juros

emprestimo.nlargest(5,'int.rate')
# Analisando as variáveis, quantidade de registros e tipos 

emprestimo.info()
# Visualização das estatisticas das variáveis

emprestimo.describe()
# Analisando os registros pelo cabeçalho

emprestimo.head()
# Analisando a quantidade da politica de crédito

emprestimo['credit.policy'].value_counts()
# Analisando a quantidade de emprestimo por objetivo

emprestimo['purpose'].value_counts()
# Histogram da quantidade de EMPRESTIMO pelo OBJETIVO

emprestimo['purpose'].iplot(kind='hist')
# Visualizando o histograma da distribuição dos empréstimos pela pontuação

plt.figure(figsize=(16,6))

sns.distplot(emprestimo['fico'], kde=True)
# Apresentando o histograma da distribuição do número de dias que teve linha de crédito

plt.figure(figsize=(16,6))

sns.distplot(emprestimo['days.with.cr.line'], kde=True)
# Apresentando a quantidade total de empréstimos pelo Objetivo

plt.figure(figsize=(16,6))

sns.countplot(x='purpose', data=emprestimo)
# Apresentado o número de EMPRESTIMOS por OBJETIVO agrupados pela politica

plt.figure(figsize=(16,6))

sns.countplot(x='purpose', data=emprestimo, hue='credit.policy')

# Quantidade pela politica de crédito

emprestimo['credit.policy'].value_counts()
# Histograma de duas distribuições FICO umas sobre as outras, uma para cada um dos valores possíveis de credit.policy

plt.figure(figsize=(16,6))

emprestimo[emprestimo['credit.policy']==1]['fico'].hist(bins=30, alpha=0.5, color='blue', label='Credit.Police=1')

emprestimo[emprestimo['credit.policy']==0]['fico'].hist(bins=30, alpha=0.5, color='red', label='Credit.Police=0')

plt.legend()

plt.xlabel('FICO')
emprestimo['not.fully.paid'].value_counts()
# Histograma de duas distribuições FICO umas sobre as outras, uma para cada um dos valores possíveis de not.fully.paid

plt.figure(figsize=(14,6))

emprestimo[emprestimo['not.fully.paid']==1]['fico'].hist(bins=30, alpha=0.5, color='blue', label='not.fully.paid=1')

emprestimo[emprestimo['not.fully.paid']==0]['fico'].hist(bins=30, alpha=0.5, color='red', label='not.fully.paid=0')

plt.legend()

plt.xlabel('FICO')
# Analisando a quantidade por objetivo

emprestimo['purpose'].value_counts()
# Vamos transformar a variável purpose em variáveis dummys.

# E Vamos acrescentar essas variáveis no nosso dataframe.



emprestimo = pd.get_dummies(emprestimo, columns = ['purpose'], drop_first = True)
# Verificando as informações do dataframe com as novas variáveis

emprestimo.info()
# Importando as funções do sklearn

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix, mean_squared_error

from sklearn.tree import DecisionTreeClassifier
# Vamos criar conjunto de treino e teste

# Separar os dados: 20% teste - usando semente aleatória para podermos replicar o resultado

train, test = train_test_split(emprestimo, test_size=0.2,random_state=42)
# Separando o dataframe de treino em treino e validação

train, valid = train_test_split(train,test_size=0.2,random_state=42)
# Verificando a quantidade de registros para cada dataframe

train.shape, valid.shape, test.shape
# Selecionar as colunas para treinamento

feats = [c for c in emprestimo.columns if c not in['not.fully.paid']]
feats
# Carregando o modelo

dtree = DecisionTreeClassifier()

# Treinamento do modelo

dtree.fit(train[feats], train['not.fully.paid'])

# Fazendo a previsão

preds = dtree.predict(valid[feats])
# Relatório de classificação  

print(classification_report(valid['not.fully.paid'],preds))
# Apresentando matriz de confusão.

pd.DataFrame(confusion_matrix(valid['not.fully.paid'],preds))
accuracy_score(test['not.fully.paid'], dtree.predict(test[feats]))
# Carregando o modelo

rfc = RandomForestClassifier(n_estimators=400,random_state=42,n_jobs=-1)
# Treinamento do modelo

rfc.fit(train[feats],train['not.fully.paid'])
# FAzendo a previsão

preds = rfc.predict(valid[feats])
# Relatório de classificação  

print(classification_report(valid['not.fully.paid'],preds))
# Apresentando matriz de confusão.

pd.DataFrame(confusion_matrix(valid['not.fully.paid'],preds)) 
accuracy_score(valid['not.fully.paid'], preds)
# Rodar o modelo para prever o 'not.fully.paid' dos dados de teste

accuracy_score(test['not.fully.paid'], rfc.predict(test[feats]))
# Analisando a importância das características

plt.figure(figsize=(14,6))

pd.Series(rfc.feature_importances_,index=feats).sort_values().plot.barh()