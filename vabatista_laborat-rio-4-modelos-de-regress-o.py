## Importando bibliotecas

import numpy as np

import pandas as pd

from sklearn.model_selection import KFold
np.random.seed(5)
# Importando o conjunto de dados

from sklearn import datasets

import matplotlib.pyplot as plt

boston = datasets.load_boston()
print(boston['feature_names']) # Aqui são os nomes dos atributos
print(boston['DESCR']) ##imprimimos a descrição do conjunto de dados
X = boston['data']

y = boston['target']

print(X.shape, y.shape) # X tem 506 linhas por 13 colunas e y 506 linhas por 1 coluna
kf = KFold(n_splits=3, shuffle=True, random_state=111)
def avalia_classificador(clf, kf, X, y, f_metrica):

    metrica_valid = []

    metrica_train = []

    y_preds = np.zeros(X.shape[0])

    # a cada iteração em kf, temos k-1 conjuntos para treino e 1 para validação

    # train e valid recebem os indices de treino e validação em cada rodada.

    for train, valid in kf.split(X,y):

        x_train = X[train] # escolhe apenas os indices de treino

        y_train = y[train]

        x_valid = X[valid] # escolhe apenas os indices de validação

        y_valid = y[valid]

        clf.fit(x_train, y_train) # treina o classificador com dados de treino

        y_pred_train = clf.predict(x_train) # faz predições nos dados de treino

        y_pred_valid = clf.predict(x_valid) # faz predições nos dados de validação

        y_preds[valid] = y_pred_valid # guarda as previsões do fold corrente

        

        # salvando métricas obtidas no dado de treino (k-1 folds) e validação (1 fold)

        metrica_valid.append(f_metrica(y_valid, y_pred_valid)) 

        metrica_train.append(f_metrica(y_train, y_pred_train)) 

    

    # retorna as previsões e a média das métricas de treino e validação

    # obtidas nas iterações do Kfold

    return y_preds, np.array(metrica_valid).mean(), np.array(metrica_train).mean()
from sklearn.metrics import mean_squared_error

def f_rmse(y_real, y_pred): 

    return mean_squared_error(y_real, y_pred)**0.5
# Média de Y

media_valor_m2 = np.mean(y)

print(media_valor_m2)
#Gera um vetor artificial com o valor da média repetido pelo número de linhas do nosso conjunto.

y_media = np.array([media_valor_m2]*y.shape[0])

print(y_media[:5]) #imprime os 5 primeiros.

print(y_media.shape) # imprime o formato do array

print('\n')



#Calcula o desempenho

print(f_rmse(y, y_media))
from sklearn.linear_model import LinearRegression



#veja a documenação do classificador e veja o que é possível alterar.

lr = LinearRegression(fit_intercept=True, normalize=True) 
preds, rmse_val, rmse_train = avalia_classificador(lr, kf, X, y, f_rmse) # treina, valida e calcula desempenho

print('RMSE (validação): ', rmse_val)

print('RMSE (treino): ', rmse_train)
from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Cria uma nova matriz de atributos com as features polinomiais de grau 2

X_new = PolynomialFeatures(2).fit_transform(X) 



# Reescalona os dados entre 0 e 1 (valor padrão do MinMaxScaler)

X_new = MinMaxScaler().fit_transform(X_new)



print('Número de atributos após transformação:', X_new.shape[1])
preds, rmse_val, rmse_train = avalia_classificador(lr, kf, X_new, y, f_rmse) 

print('RMSE (validação): ', rmse_val)

print('RMSE (treino): ', rmse_train)
%matplotlib inline

import matplotlib.pyplot as plt



plt.scatter(preds, y);

plt.plot(preds,preds, c ='r')
from sklearn import svm

svr = svm.SVR(gamma='auto')
preds, rmse_val, rmse_train = avalia_classificador(svr, kf, X, y, f_rmse) 

print('RMSE (validação): ', rmse_val)

print('RMSE (treino): ', rmse_train)
from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor(n_neighbors=3)
preds, rmse_val, rmse_train = avalia_classificador(neigh, kf, X, y, f_rmse) 

print('RMSE (validação): ', rmse_val)

print('RMSE (treino): ', rmse_train)
from sklearn import tree

dt = tree.DecisionTreeRegressor(max_features=5, max_depth=3, random_state=10)
preds, rmse_val, rmse_train = avalia_classificador(dt, kf, X, y, f_rmse) 

print('RMSE (validação): ', rmse_val)

print('RMSE (treino): ', rmse_train)
from sklearn import tree

import graphviz 

dot_data = tree.export_graphviz(dt, out_file=None, 

                                feature_names=boston['feature_names'],  

                                filled=True, rounded=True,  

                                special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 