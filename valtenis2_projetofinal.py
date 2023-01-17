import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

from sklearn.metrics import log_loss

from random import randint
#Dados do csv Train



data_train = pd.read_csv('../input/forest_data.csv')



x_train = data_train.drop('Label', 1)

y_train = pd.Series(LabelEncoder().fit_transform(data_train['Label']))



#Dados do csv Teste



data_test = pd.read_csv('../input/forest_data_teste.csv')



x_test = data_test.drop('Label', 1)

y_test = pd.Series(LabelEncoder().fit_transform(data_test['Label']))
xgboost = XGBClassifier()

xgboost.fit(x_train, y_train)
probpredictXGB = xgboost.predict_proba(x_test)

probpredictXGB.shape
y_test.head(5)
aux = pd.get_dummies(y_test)

aux.head(5)
def medirProba(index, respostas, probabilidades):

    

    start = index

    end = index+1

    

    print('-' * 42)

    print('Amostra {}, Resultado: {}'.format(index, np.array(respostas.iloc[start:end, :])))

    print('-' * 42)

    print('Probabilidade de "forest": {}'.format(np.array(probabilidades[start:end, 0])))

    print('Probabilidade de "ground": {}'.format(np.array(probabilidades[start:end, 1])))

    print('Probabilidade de "sky"...: {}'.format(np.array(probabilidades[start:end, 2])))
qtdTestes = 5



print('\nRealizando testes de predição com {} Amostras...\n'.format(qtdTestes))

print('[["forest", "ground", "sky"]]\n')

for i in range(qtdTestes):

    index = randint(0,y_test.shape[0])

    medirProba(index, aux, probpredictXGB)
scoreXGB = xgboost.score(x_test, y_test)

loglossXGB = log_loss(y_test, probpredictXGB)





auc1 = roc_auc_score(aux.iloc[:, 0], probpredictXGB[:, 0])

auc2 = roc_auc_score(aux.iloc[:, 1], probpredictXGB[:, 1])

auc3 = roc_auc_score(aux.iloc[:, 2], probpredictXGB[:, 2])



geral = pd.DataFrame({

    'Label': ['forest', 'ground', 'sky'],

    'Auc': [auc1, auc2, auc3]

})



print('-' * 40)

print('\tInformações Gerais')

print('-' * 40)

print('logloss = %.3f' % loglossXGB)

print('-' * 40)

print('Score do Modelo = %.3f' % scoreXGB)

print('-' * 40)

print('Scores Separados Utilizando AUC ROC\n\n', geral)

print('-' * 40)