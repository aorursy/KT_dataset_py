#wesley silva soares 7626838
import numpy as np
import sklearn 
import pandas as pd
import matplotlib.pyplot as graphic
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
# Importando as bases de treinamento e de teste 
adult = pd.read_csv("../input/train_data.csv",na_values="?")
testadult = pd.read_csv("../input/test_data.csv",na_values="?")
# excluindo as linhas contendo missing data 
nadult = adult.dropna()
ntestadult = testadult.dropna()
adult.head
adult["native.country"].value_counts()
adult["age"].value_counts().plot(kind="bar")
adult["education"].value_counts().plot(kind="bar")
adult["sex"].value_counts().plot(kind="bar")
Xadult = nadult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Xtestadult = ntestadult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Xadult
#define o target
Yadult = nadult.income
n_neighborsMaiorScore =0
melhorneighbors = 0
#procura pelo melhor knn para os classificadores escolhidos
for n_neighborsTemp in range(3,80,2):
    knn = KNeighborsClassifier(n_neighbors=n_neighborsTemp)
    scores = cross_val_score(knn, Xadult, Yadult, cv=10)
    scores.mean()
    if scores.mean() > n_neighborsMaiorScore:
        n_neighborsMaiorScore = scores.mean()
        melhorneighbors = n_neighborsTemp
#mostra a media dos scores de treino
n_neighborsMaiorScore
#mostra o melhor knn 
melhorneighbors
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(Xtestadult)
YtestPred