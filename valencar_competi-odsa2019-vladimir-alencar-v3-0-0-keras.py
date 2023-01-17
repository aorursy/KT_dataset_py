import warnings

warnings.filterwarnings('ignore')



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



y[:5]
from keras.layers.advanced_activations import LeakyReLU, PReLU

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout



# cria um Modelo de Rede Neural Profunda usando o Keras, com Dropout e Ativações tipo LeakyReLU

def criar_modelo_v1(Optimizer, Units, Activation):

    model = Sequential()

    model.add(Dense(input_dim=8, units=Units, activation=Activation, kernel_initializer='uniform'))

    model.add(Dense(units=10, activation='relu'))   

    model.add(LeakyReLU(alpha=.001)) 

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Optimizer, metrics=['accuracy'])

    return model
import sklearn.metrics as metrics

def verificar_hiperparametros_V3_0_0():

    Epochs = 10000

    Otimizador = 'Nadam'

    Activation = 'relu'

    Units = 48 

    Epochs =  10000 

    Batch_size = 32  



    modelo = criar_modelo_v1(Otimizador, Units, Activation)

    out = modelo.fit(X_treino, y_treino, epochs=Epochs, verbose=0, 

                     batch_size=Batch_size,

                     validation_split=0.2)



    acc_treino = out.history['acc'][Epochs-1]

    y_pred_treino = modelo.predict_classes(X_treino)

    confusion_matrix_treino = metrics.confusion_matrix(y_true=y_treino, y_pred=y_pred_treino)

    print('==========================================')

    print('Ativação:', Activation,' - Epochs:', Epochs, ' - Otimizador',Otimizador)

    print('Acurácia de treino:',acc_treino)

    print("Confusion Matrix treino:")

    print(confusion_matrix_treino)  

    

    # Gera arquivo para o Kaggle

    nome_arquivo = 'Submissao-v3.0.0-Keras.csv'

    df_saida = pd.DataFrame()

    df_saida['id'] = ids_arquivo_de_teste.values

    yteste_previsto = modelo.predict_classes(X_teste_kaggle)         

    df_saida['classe'] =   yteste_previsto.ravel()

    # Salvando o arquivo

    df_saida.to_csv(nome_arquivo, index=False)

    print('Arquivo %s salvo...', nome_arquivo)

    !head Submissao-v3.0.0-Keras.csv

        

verificar_hiperparametros_V3_0_0()