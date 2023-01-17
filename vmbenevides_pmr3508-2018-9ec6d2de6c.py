import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
adult = pd.read_csv("../input/adult-data/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.head ()
adult.shape
#Retirando linhas com valores NaN
adult = adult.dropna ()
#Paises
adult["native.country"].value_counts()
#Idade
adult["age"].value_counts().plot ('bar')
#Tempo de estudo
adult["education.num"].value_counts().plot(kind="bar")
#Gênero
adult["sex"].value_counts().plot(kind="bar")
#Estado civil
adult["marital.status"].value_counts().plot(kind="bar")
#Tempo de trabalho por semana, gráfico em formato diferente pois é mais fácil de compreender do que em barras.
adult["hours.per.week"].value_counts().plot('pie')
#Setor de trabalho
adult["workclass"].value_counts().plot('bar')
#Etnia
adult["race"].value_counts().plot('bar')
#Rotulos
adult["income"].value_counts().plot('bar')
testadult = pd.read_csv("../input/adult-data/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
testadult.head ()
testadult.shape
#testadult = testadult.dropna ()
#testadult.shape 
adult.loc[adult['sex']!="Male", 'sex'] = '0'
adult.loc[adult['sex']=="Male", 'sex'] = '1'
testadult.loc[testadult['sex']!="Male", 'sex'] = '0'
testadult.loc[testadult['sex']=="Male", 'sex'] = '1'
adult['sex'].value_counts()
testadult['sex'].value_counts()
adult.loc[adult['race']!="White", 'race'] = '0'
adult.loc[adult['race']=="White", 'race'] = '1'
testadult.loc[testadult['race']!="White", 'race'] = '0'
testadult.loc[testadult['race']=="White", 'race'] = '1'
adult['race'].value_counts()
testadult['race'].value_counts()
adult.loc[adult['native.country']!="United-States", 'native.country'] = '0'
adult.loc[adult['native.country']=="United-States", 'native.country'] = '1'
testadult.loc[testadult['native.country']!="United-States", 'native.country'] = '0'
testadult.loc[testadult['native.country']=="United-States", 'native.country'] = '1'
adult['native.country'].value_counts()
testadult['native.country'].value_counts()
Xadult = adult[['education.num','age','race','sex','capital.gain','capital.loss','hours.per.week']]
Yadult = adult.income
Xtestadult = testadult[['education.num','age','race','sex','capital.gain','capital.loss','hours.per.week']]
knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(Xadult,Yadult)
cval = 10
scores = cross_val_score(knn, Xadult, Yadult, cv=cval)
scores
total = 0
for i in scores:
    total += i
acuracia_esperada = total/cval
acuracia_esperada
YtestPred = knn.predict (Xtestadult)
YtestPred
maior_50 = 0
menor_50 = 0
for i in YtestPred:
    if i == '<=50K':
        menor_50 += 1
    else:
        maior_50 += 1
dicio = {'<=50K':menor_50, '>50K':maior_50}
plt.bar(range(len(dicio)), list(dicio.values()), align='center',color = ('blue','orange'))
plt.xticks(range(len(dicio)), list(dicio.keys()))
result = np.vstack((testadult["Id"], YtestPred)).T
x = ["Id","income"]
Resultado = pd.DataFrame(columns = x, data = result)
Resultado.to_csv("Resultado1.csv", index = False)
Xadult = adult[['education.num','age','sex','capital.gain','capital.loss','hours.per.week','native.country']]
Yadult = adult.income
Xtestadult = testadult[['education.num','age','sex','capital.gain','capital.loss','hours.per.week','native.country']]
knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(Xadult,Yadult)
cval = 10
scores = cross_val_score(knn, Xadult, Yadult, cv=cval)
scores
total = 0
for i in scores:
    total += i
acuracia_esperada = total/cval
acuracia_esperada
YtestPred = knn.predict (Xtestadult)
YtestPred
maior_50 = 0
menor_50 = 0
for i in YtestPred:
    if i == '<=50K':
        menor_50 += 1
    else:
        maior_50 += 1
dicio = {'<=50K':menor_50, '>50K':maior_50}
plt.bar(range(len(dicio)), list(dicio.values()), align='center',color = ('blue','orange'))
plt.xticks(range(len(dicio)), list(dicio.keys()))
result = np.vstack((testadult["Id"], YtestPred)).T
x = ["Id","income"]
Resultado = pd.DataFrame(columns = x, data = result)
Resultado.to_csv("Resultado2.csv", index = False)
Xadult = adult[['education.num','race','sex','capital.gain','capital.loss','hours.per.week','native.country']]
Yadult = adult.income
Xtestadult = testadult[['education.num','race','sex','capital.gain','capital.loss','hours.per.week','native.country']]
knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(Xadult,Yadult)
cval = 10
scores = cross_val_score(knn, Xadult, Yadult, cv=cval)
scores
total = 0
for i in scores:
    total += i
acuracia_esperada = total/cval
acuracia_esperada
YtestPred = knn.predict (Xtestadult)
YtestPred
maior_50 = 0
menor_50 = 0
for i in YtestPred:
    if i == '<=50K':
        menor_50 += 1
    else:
        maior_50 += 1
dicio = {'<=50K':menor_50, '>50K':maior_50}
plt.bar(range(len(dicio)), list(dicio.values()), align='center',color = ('blue','orange'))
plt.xticks(range(len(dicio)), list(dicio.keys()))
result = np.vstack((testadult["Id"], YtestPred)).T
x = ["Id","income"]
Resultado = pd.DataFrame(columns = x, data = result)
Resultado.to_csv("Resultado3.csv", index = False)
