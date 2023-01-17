import os

print(os.listdir("../input"))
import pandas as pd

import numpy as np

import sklearn
dftreino = pd.read_csv('../input/dataset_treino.csv')

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
#!pip install xgboost
from xgboost import XGBClassifier

params = {'max_depths': 1.0, 'max_features': 1, 

          'min_samples_leafs': 0.1, 'min_samples_splits': 0.1, 'n_estimators': 16}



n_estimators = 16

max_depths = 1.0

min_samples_splits = 0.1

min_samples_leafs = 0.1

max_features = 1



modelXGB = XGBClassifier(n_estimators = n_estimators,

                      max_depths = max_depths,

                      min_samples_splits = min_samples_splits,

                      min_samples_leafs = min_samples_leafs,

                      max_features = max_features

                     )

modelXGB.fit(X_treino, y_treino)



nome_arquivo = 'Submissao-v3.0.0-XGBoost.csv'

df_saida = pd.DataFrame()

df_saida['id'] = ids_arquivo_de_teste.values

yteste_previsto = modelXGB.predict(X_teste_kaggle)         

df_saida['classe'] =   yteste_previsto.ravel()

print(df_saida.classe.value_counts())

# Salvando o arquivo

df_saida.to_csv(nome_arquivo, index=False)

print('Arquivo %s salvo...', nome_arquivo)

!head 'Submissao-v3.0.0-XGBoost.csv'