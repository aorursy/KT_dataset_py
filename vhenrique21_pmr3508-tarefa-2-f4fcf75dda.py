import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib as plt
%matplotlib inline
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import os
print(os.listdir("../input/spam-data-set"))
Train_Data = pd.read_csv("../input/spam-data-set/train_data.csv",
        engine='python')
Train_Data.head()
HamSet = Train_Data[Train_Data.ham == True]
SpamSet = Train_Data[Train_Data.ham == False]
HamSet.describe(percentiles=[0.125, .25, .50, .75, .90])
SpamSet.describe(percentiles=[0.125, .25, .50, .75, .90])
indices = np.array(HamSet.columns)
indicesValidos = []
for indice in indices:
    if (SpamSet[indice].mean() >= HamSet[indice].mean() and HamSet[indice].std() <= 1.5 and SpamSet[indice].std() <= 1.5) :
        #print(indice)
        indicesValidos.append(indice)
X_Train = Train_Data[indicesValidos]
X_Train2 = Train_Data.drop(columns=['ham'])
Y_Train = Train_Data.ham
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
knn_F = KNeighborsClassifier(n_neighbors=3)
knn_F.fit(X_Train,Y_Train)
resultado = cross_val_score(knn_F, X_Train, Y_Train, cv=50)
print("Acuracia:", resultado.mean())
print("Desvio Padrão:", resultado.std())
knn_SF = KNeighborsClassifier(n_neighbors=3)
knn_SF.fit(X_Train2,Y_Train)
resultado = cross_val_score(knn_SF, X_Train2, Y_Train, cv=50)
print("Acuracia:", resultado.mean())
print("Desvio Padrão:", resultado.std())
distrNormal_F = GaussianNB()
distrNormal_F.fit(X_Train, Y_Train)
resultado = cross_val_score(distrNormal_F, X_Train, Y_Train, cv=50)
#print(resultado)
print("Acuracia:", resultado.mean())
print("Desvio Padrão:", resultado.std())
distrNormal_SF = GaussianNB()
distrNormal_SF.fit(X_Train2, Y_Train)
resultado = cross_val_score(distrNormal_SF, X_Train2, Y_Train, cv=50)
#print(resultado)
print("Acuracia:", resultado.mean())
print("Desvio Padrão:", resultado.std())
distrBernoulli_F = BernoulliNB()
distrBernoulli_F.fit(X_Train, Y_Train)
resultado = cross_val_score(distrBernoulli_F, X_Train, Y_Train, cv=50)
#print(resultado)
print("Acuracia:", resultado.mean())
print("Desvio Padrão:", resultado.std())
distrBernoulli_SF = BernoulliNB()
distrBernoulli_SF.fit(X_Train2, Y_Train)
resultado = cross_val_score(distrBernoulli_SF, X_Train2, Y_Train, cv=50)
#print(resultado)
print("Acuracia:", resultado.mean())
print("Desvio Padrão:", resultado.std())
distrMulti_F = MultinomialNB()
distrMulti_F.fit(X_Train, Y_Train)
resultado = cross_val_score(distrMulti_F, X_Train, Y_Train, cv=50)
#print(resultado)
print("Acuracia:", resultado.mean())
print("Desvio Padrão:", resultado.std())
distrMulti_SF = MultinomialNB()
distrMulti_SF.fit(X_Train2, Y_Train)
resultado = cross_val_score(distrMulti_SF, X_Train2, Y_Train, cv=50)
#print(resultado)
print("Acuracia:", resultado.mean())
print("Desvio Padrão:", resultado.std())
Test_Data = pd.read_csv("../input/spam-data-set/test_features.csv",
        engine='python')
X_Test = Test_Data[indicesValidos]
resultadoTest = distrBernoulli_F.predict(X_Test)
Id = Test_Data['Id']
csvTest = pd.DataFrame({'Id':Id,'ham':resultadoTest[:]})
csvTest.to_csv("Bernoulli_com_filtro.csv", index = False)
resultadoTest = distrBernoulli_SF.predict(Test_Data)
Id = Test_Data['Id']
csvTest = pd.DataFrame({'Id':Id,'ham':resultadoTest[:]})
csvTest.to_csv("Bernoulli_sem_filtro.csv", index = False)
resultadoTest = knn_F.predict(X_Test)
Id = Test_Data['Id']
csvTest = pd.DataFrame({'Id':Id,'ham':resultadoTest[:]})
csvTest.to_csv("KNN_com_filtro.csv", index = False)