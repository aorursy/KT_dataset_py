import sklearn as sl

import warnings

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from pandas import read_csv

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

import pickle

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
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
#Leitura do arquivo de treino

df_santander = pd.read_csv("/kaggle/input/santander-customer-satisfaction/train.csv")
df_santander.head(10)
# Visualizando as dimensões

df_santander.shape
#Aplicando PCA pela grande quantidade de variáveis no dataset



# Feature Extraction



# Carregando os dados

array = df_santander.values



# Separando o array em componentes de input e output

X = array[:,0:370]

Y = array[:,370]



# Normalizando os dados

scaler = MinMaxScaler(feature_range = (0, 1))

rescaledX = scaler.fit_transform(X)



# Seleção de atributos

pca = PCA(n_components = 10)

pca.fit(rescaledX)
#Transformando os componentes em DataFrame

columns = ['pca_%i' % i for i in range(10)]

df_pca = pd.DataFrame(pca.transform(rescaledX), columns=columns)
#Concatenando a coluna target no Dataframe de componentes

finalDf = pd.concat([df_pca, df_santander[['TARGET']]], axis = 1)
finalDf.head(10)
#Análise Exploratória de Dados após PCA



# Histograma Univariado

finalDf.hist()

plt.show()
array = finalDf.values



# Separando o array em componentes de input e output

X_TREINO = array[:,0:10]

Y_TREINO = array[:,10]





# Criando o modelo

modelo = LogisticRegression()



# Treinando o modelo

modelo.fit(X_TREINO, Y_TREINO)



# Salvando o modelo

arquivo = 'modelo_classificador_final.sav'

pickle.dump(modelo, open(arquivo, 'wb'))

print("Modelo salvo!")
df_santander_test = pd.read_csv("/kaggle/input/santander-customer-satisfaction/test.csv")

df_santander_test_target = pd.read_csv("/kaggle/input/santander-customer-satisfaction/sample_submission.csv")
df_santander_test_target.columns
df_santander_test.shape
df_santander_test.columns
array_test = df_santander_test.values



# Separando o array em componentes de input e output

X_TEST = array_test[:,0:369]



# Normalizando os dados

scaler_test = MinMaxScaler(feature_range = (0, 1))

rescaledX_test = scaler_test.fit_transform(X_TEST)



# Seleção de atributos

pca_test = PCA(n_components = 10)

pca_test.fit(rescaledX_test)
columns = ['pca_%i' % i for i in range(10)]

df_pca_test = pd.DataFrame(pca_test.transform(rescaledX_test), columns=columns)



finalDf_test = pd.concat([df_pca_test, df_santander_test_target[['TARGET']]], axis = 1)
finalDf_test.head(10)
# Carregando os dados

array_final_test = finalDf_test.values



# Separando o array em componentes de input e output

X_teste_final = array_final_test[:,0:10]

Y_teste_final = array_final_test[:,10]
modelo_classificador_final = pickle.load(open(arquivo, 'rb'))

modelo_prod = modelo_classificador_final.score(X_teste_final, Y_teste_final)

print("Modelo carregado!")



# Print do resultado

print("Acurácia: %.3f" % (modelo_prod.mean() * 100))