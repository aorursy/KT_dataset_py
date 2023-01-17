import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
pd.options.display.max_columns = 150
test = pd.read_csv ("../input/costarican/test.csv")
test.head ()        
test.shape
train = pd.read_csv ("../input/costarican/train.csv")
train.head ()
train.shape
train.isnull().values.sum(axis=0)
missing_train = pd.DataFrame(train.isnull().sum()).rename(columns = {0:'total'})
missing_train ['percent'] = missing_train ['total']/len(train)
missing_train.sort_values('percent', ascending = False).head(10)
missing_train_novo = missing_train.drop(missing_train[missing_train['percent'] < 0.5].index)
index_ruim_train = missing_train_novo.index.values
train_novo = train.drop(index_ruim_train,axis = 1)
train_novo.shape
ntrain_novo = train_novo.dropna()
tipos = ntrain_novo.columns.to_series().groupby(ntrain_novo.dtypes).groups
ntrain_novo = ntrain_novo.drop(['idhogar', 'dependency', 'edjefe', 'edjefa'], axis = 1)
missing_test = pd.DataFrame(test.isnull().sum()).rename(columns = {0:'total'})
missing_test ['percent'] = missing_test ['total']/len(test)
missing_test.sort_values('percent', ascending = False).head(10)
missing_test_novo = missing_test.drop(missing_test[missing_test['percent'] < 0.5].index)
index_ruim_test = missing_test_novo.index.values
test_novo = test.drop(index_ruim_test,axis = 1)
test_novo.shape
ntest_novo = test_novo.fillna(test_novo.mean())
ntest_novo = ntest_novo.drop(['idhogar', 'dependency', 'edjefe', 'edjefa'], axis = 1)
ntest_novo.shape
Xtrain = ntrain_novo.iloc [:,1:(len(ntrain_novo.columns)-1)]
Ytrain = ntrain_novo.Target
Xtrain.shape
Xtest = ntest_novo.iloc[:,1:len(ntest_novo.columns)]
Xtest.shape
cval= 10
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=cval)
total = 0
for i in scores:
    total += i
acuracia_esperada = total/cval
acuracia_esperada
knn.fit(Xtrain,Ytrain)
YtestPred = knn.predict (Xtest)
um = 0
dois = 0
tres = 0
quatro = 0 
for i in YtestPred:
    if i == 1:
        um += 1
    elif i == 2:
        dois += 1
    elif i == 3:
        tres += 1
    else:
        quatro += 1      
dicio = {'1':um, '2':dois,'3':tres,'4':quatro}
print (um,dois,tres,quatro,sep=',')

plt.bar(range(len(dicio)), list(dicio.values()), align='center')
plt.xticks(range(len(dicio)), list(dicio.keys()))
result = np.vstack((ntest_novo["Id"], YtestPred)).T
x = ["Id","Target"]
Resultado = pd.DataFrame(columns = x, data = result)
Resultado.to_csv("PMR3508-2018-9ec6d2de6c-HouseholdIncome.csv", index = False)