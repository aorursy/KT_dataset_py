#Importando Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
#Obter dados

dados = pd.read_csv('Wine.csv', header=None)
dados.head()
#Mudar o nome das colunas
dados.columns = ['#1 Wine class labels', '#14.23 Malic acid', '#1.71', '#2.43','#15.6','#127','#2.8','#3.06','#.28','#2.29', '#5.64','#1.04','#3.92', '1065']
dados
#Número de classe
dados['#1 Wine class labels'].unique()
#Verificando valores NULL (NAN)
dados.isnull().sum()
#Informação dos dados
dados.info()
#Plotando dados por quantidade de classes
dados['#1 Wine class labels'].value_counts()
sns.countplot(x="#1 Wine class labels", data=dados)
plt.show()
#Organizando em entrada (features) e saída (label/classe)
X = dados.iloc[:,1:-1].values # Todos os dados da segunda ate a ultima coluna
y = dados.iloc[:,0].values # Todos os dados da primeira coluna do dataset
X
y
#Dividir em treinamento 75% dos dados e 25% para teste
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.25, random_state= 0) #random_state é uma semente para gerar valores aleatorios
y_train
wine = DecisionTreeClassifier(max_depth = 3, criterion="entropy") 
#max_depht - A profundidade máxima da árvore.
#criterion (entropy) - A função para medir a qualidade de uma divisão. "Entropia" para o ganho de informações.
wine.fit(X,y)
#plotando a árvore
tree.plot_tree(wine)
wine.classes_
y_pred = wine.predict(X_train)
acc = metrics.accuracy_score(y_train, y_pred)
print(acc)
#Testando
y_pred = wine.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
print (pd.crosstab(y_test,y_pred,rownames=['Classe'], colnames=['        Predito'], margins=True))
#Comprovando o acerto e precisão da árvore
X_test[0]
var = np.array([[ 13.74,   1.67,   2.25,  16.4 , 118.  ,   2.6 ,   2.9 ,   0.21,
         1.62,   5.85,   0.92,   3.2 ]])
wine.predict(var)
wine.predict(var.reshape(1,-1))
#Árvore de decisão – Algoritmo de aprendizado supervisionado, muito útil em problemas de classificação. 
#A árvore é de fácil entendimento, desde que sua profundidade não seja tão alta. Com uma profundidade 
#definida para 3 foi possível obter 98% de acuracia com os dados de treino e 100% com os de teste.
#Concluindo que o melhor desempenho se obteve através do algoritmo de árvore de decisão. 
