#Importando Bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
#Obter dados

dados = pd.read_csv('Wine.csv', header=None)
dados.head()
#Mudar o nome das colunas
dados.columns = ['#1 Wine class labels', '#14.23 Malic acid', '#1.71', '#2.43','#15.6','#127','#2.8','#3.06','#.28','#2.29', '#5.64','#1.04','#3.92', '1065']
dados
#Número de classe
dados['#1 Wine class labels'].unique()
#Verificando se existe dados null(NAN)
dados.isnull().sum()
#Informação sobre os dados
dados.info()
dados['#1 Wine class labels'].value_counts()
sns.countplot(x="#1 Wine class labels", data= dados)
plt.show()
#organizando em entrada e saidas
X = dados.iloc[:,1:-1].values # Todos os dados da segunda ate a ultima coluna
y = dados.iloc[:,0].values # Todos os dados da primeira coluna do dataset
X
y
#dividir em treinamento 75% dos dados e 25% para teste
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.25, random_state= 0) #random_state é uma semente para gerar valores aleatorios
y_train
#Treinando o Modelo

neuronios = 15 #quanto menor o número do neoronios menor a acc(accuracy)
funcAtivacao = 'identity'
algoTreinamento = 'sgd'
taxaDeAprendizado = 0.001 #representa aquantidade de saltos até chegar no objetivo (não é um valor exato)
mlpWine = MLPClassifier(hidden_layer_sizes=neuronios,  activation=funcAtivacao, solver=algoTreinamento, learning_rate_init=taxaDeAprendizado )
mlpWine
mlpWine.fit(X,y)
mlpWine.classes_
y_pred = mlpWine.predict(X_train)
acc = metrics.accuracy_score(y_train, y_pred)
print(acc)
#Testando os dados

X_test
y_pred = mlpWine.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
X_test[0]
var = np.array([ 13.74,   1.67,   2.25,  16.4 , 118.  ,   2.6 ,   2.9 ,   0.21,
         1.62,   5.85,   0.92,   3.2 ])
mlpWine.predict(var.reshape(1,-1))
#MLP (Multi Layer Perceptron) – Modelo de aprendizado supervisionado onde é fornecido uma amostra de 
#dados informando a classificação dos mesmo, assim treinando o modelo para que quando uma nova amostra 
#for recebedi o mesmo seja capaz de estabelecer a que grupo uma amostra pertence. Usando os dados de 
#treino o modelo alcançou uma taxa de 91% de acuracia e 86% com os dados de teste. 
#Observando que a quandtidade de neurônios influência diretamente na precisão do modelo. 