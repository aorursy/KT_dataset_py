import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
#Obtendo os Dados
dados = pd.read_csv('Wine.csv',header=None)
#printando os dados
dados.head()
#mudar o nome das colunas
dados.columns = ['#1 Wine class labels', '#14.23 Malic acid', '#1.71', '#2.43','#15.6','#127','#2.8','#3.06','#.28','#2.29', '#5.64','#1.04','#3.92', '1065']
dados
#número de classes
dados['#1 Wine class labels'].unique()
#estatistica dos dados
dados.describe()
#verificando se existe valor null(NAN)
dados.isnull().sum()
#informações dos dados
dados.info()
#Quantidade por classe
dados['#1 Wine class labels'].value_counts()
#Plotando quantidade por classe
sns.countplot(x='#1 Wine class labels', data=dados)
plt.show()
dados.head()
#organizando em entrada(features) e saída (label/classe)

#X = dados.iloc[:,:-1].values
#y = dados.iloc[:,-1].values

X = dados.iloc[:,1:-1].values # Todos os dados da segunda ate a ultima coluna
y = dados.iloc[:,0].values # Todos os dados da primeira coluna do dataset
X
y
#dividir em treinamento 75% dos dados e 25% para teste
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.25, random_state= 0) #random_state é uma semente para gerar valores aleatorios
y_train
# Definindo o número de vizinhos.
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X,y)
knn
knn.classes_
y_pred = knn.predict(X_train)
acc = metrics.accuracy_score(y_train, y_pred)
print(acc)
y_pred = knn.predict(X_test)
X_test[0]
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
x = np.array([[  13.74,   1.67,   2.25,  16.4 , 118.  ,   2.6 ,   2.9 ,   0.21,
         1.62,   5.85,   0.92,   3.2 ]])
knn.predict(x)
knn.predict(x.reshape(1, -1))
#KNN (K nearest neighboors) – K vizinho mais próximo, modelo de classificação de uma nova amostra baseado 
#na distância das amostras vizinhas de um conjunto de treinamento. A vantagem do KNN é a abordagem simples, 
#todavia, caso a base de dados seja muito grande pode se tornar uma atividade computacionamente custosa. 
#Por fim o Knn alcançou um acuracia de 94% com os dados de treino e 93% com os da teste.