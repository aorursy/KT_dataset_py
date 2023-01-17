import os

print(os.listdir("../input"))
import pandas as pd

import numpy as np

import sklearn
dftreino = pd.read_csv("../input/dataset_treino.csv")

dftreino.head()
dfteste = pd.read_csv('../input/dataset_teste.csv')

dfteste.head()
dftreino.describe()
dfteste.describe()
# verificar se tem nulos

dftreino.isna().sum() 
# verificar se tem nulos

dfteste.isna().sum() 
## verificar se os datasets estão balanceados

dftreino.classe.value_counts()
# treino

from sklearn import preprocessing



ids = dftreino.id

classes = dftreino.classe



x = dftreino.values # retorna uma array numpy 

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

dftreino = pd.DataFrame(x_scaled, columns=dftreino.columns)

dftreino.id = ids

dftreino.classe = classes

dftreino.head()
len(dftreino)
# teste

ids_arquivo_de_teste = dfteste.id



x = dfteste.values # retorna uma array numpy 

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

dfteste = pd.DataFrame(x_scaled, columns=dfteste.columns)

dfteste.id = ids



dfteste.head()
len(dfteste)
global X, y, X_treino, y_treino, X_teste, y_teste, X_teste_kaggle
columns = dftreino.columns

atributos = columns[1:len(columns)-1]

atributos
X = dftreino[atributos].values

y = dftreino['classe'].values



X_treino = X

y_treino = y



X_teste_kaggle = dfteste[atributos].values



y[:5]
from sklearn.svm import SVC



params = {'C': 3.1, 'gamma': 0.34, 'kernel': 'rbf'}

C = params['C']

gamma = params['gamma']

kernel = params['kernel']

classifierSVM = SVC(C=C, gamma = gamma, kernel=kernel)

classifierSVM.fit(X_treino, y_treino)



nome_arquivo = 'Submissao-v3.0.0-SVM.csv'

df_saida = pd.DataFrame()

df_saida['id'] = ids_arquivo_de_teste.values

yteste_previsto = classifierSVM.predict(X_teste_kaggle)         

df_saida['classe'] = yteste_previsto.ravel()

print(df_saida.classe.value_counts())

# Salvando o arquivo

df_saida.to_csv(nome_arquivo, index=False)

print('Arquivo %s salvo...', nome_arquivo)

!head 'Submissao-v3.0.0-SVM.csv'
!ls