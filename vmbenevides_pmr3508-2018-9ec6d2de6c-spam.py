import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import chi2
train = pd.read_csv ("../input/spam-and-ham/train_data.csv")
train.head()
train.columns
res = train['ham'].value_counts()
spam_perc = res[False]/(res[True]+res[False])
print ('Porcentagem de Spam: ',round(spam_perc*100,3),"%",sep='')
test = pd.read_csv ("../input/spam-and-ham/test_features.csv")
test.head ()
mk = 0
acuracia_anterior = 0
adicio = {}
for i in range (1,31):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train.iloc[:,0:57],train.ham)
    c_val = 15
    scores = cross_val_score (knn,train.iloc[:,0:57],train.ham,cv=c_val)
    total = 0
    for j in scores:
        total += j
    acuracia_esperada = total/c_val
    if acuracia_esperada > acuracia_anterior:
        acuracia_anterior = acuracia_esperada
        mk = i
    adicio [i] = acuracia_esperada
for i in adicio:
    plt.scatter (i,adicio[i])
print ("O melhor k foi:", mk)
a = 0
b = 0
P = 15
for i in range (train.shape[0]):
    if train.iloc[i].ham == True:
        a += train.iloc[i]
    else:
        b += train.iloc[i]
#Mudando a e b para os valores médios
a = a/2251
b = b/1429
lista = []
for i in range (0,57):
    if a[i] >= P*b[i] or b[i] >= P*a[i]:
        lista.append (i)
knn = KNeighborsClassifier (n_neighbors = mk)
knn.fit (train.iloc[:,lista],train.ham)
c_val = 10
scores = cross_val_score (knn,train.iloc[:,lista],train.ham,cv=c_val)
total = 0
for j in scores:
    total += j
acuracia_esperada = total/c_val
acuracia_esperada
Ytest_pred1 = knn.predict (test.iloc[:,lista])
verd1 = 0
fal1 = 0
for i in Ytest_pred1:
    if i == True:
        verd1 += 1
    else:
        fal1 += 1
dicio1 = {"Spam":fal1, "Ham": verd1}
plt.bar(range(len(dicio1)), list(dicio1.values()), align='center',color = ('blue','orange'))
plt.xticks(range(len(dicio1)), list(dicio1.keys()))
print ('Spam:', fal1)
print ('Ham:', verd1)
print ('Porcentagem de Spam: ',round(fal1/(verd1+fal1)*100,4),'%',sep='')
result1 = np.vstack((test["Id"], Ytest_pred1)).T
x1 = ["Id","ham"]
Resultado = pd.DataFrame(columns = x1, data = result1)
Resultado.to_csv("resultados_knn_1.csv", index = False)
melhor = [0,0]
for z in range (1,len(test.columns)-1):
    X_new = SelectKBest(f_regression, k=z).fit(train.iloc[:,0:len(test.columns)-1], train.ham)
    a = X_new.get_support (indices = True)
    b = []
    for i in a:
        b.append (i)
    Xtrain_kbest = train[train.columns[b]]
    Ytrain_kbest = train.ham
    Xtest_kbest = test[test.columns[b]]
    knn = KNeighborsClassifier(n_neighbors=mk)
    knn.fit(Xtrain_kbest,Ytrain_kbest)    
    c_val = 15
    scores = cross_val_score (knn, Xtrain_kbest, Ytrain_kbest, cv=c_val)
    total = 0
    for i in scores:
        total += i
    acuracia_esperada = total/c_val
    if acuracia_esperada > melhor[0] and acuracia_esperada != 1.0: 
        melhor[0] = acuracia_esperada
        melhor[1] = z
print (melhor) 
X_new = SelectKBest(f_regression, k=melhor[1]).fit(train.iloc[:,0:len(test.columns)-1], train.ham)
a = X_new.get_support (indices = True)
b = []
for i in a:
        b.append (i)
Xtrain_kbest = train[train.columns[b]]
Ytrain_kbest = train.ham
Xtest_kbest = test[test.columns[b]]
knn = KNeighborsClassifier(n_neighbors=mk)
knn.fit(Xtrain_kbest,Ytrain_kbest)    
c_val = 15
scores = cross_val_score (knn, Xtrain_kbest, Ytrain_kbest, cv=c_val)
total = 0
for i in scores:
    total += i
acuracia_esperada = total/c_val
acuracia_esperada
Ytest_pred2 = knn.predict (Xtest_kbest)
verd2 = 0
fal2 = 0
for i in Ytest_pred2:
    if i == True:
        verd2 += 1
    else:
        fal2 += 1
dicio2 = {"Spam":fal2, "Ham": verd2}
plt.bar(range(len(dicio2)), list(dicio2.values()), align='center',color = ('blue','orange'))
plt.xticks(range(len(dicio2)), list(dicio2.keys()))
print ('Spam:', fal2)
print ('Ham:', verd2)
print ('Porcentagem de Spam: ',round(fal2/(verd2+fal2)*100,4),'%',sep='')
result2 = np.vstack((test["Id"], Ytest_pred2)).T
x2 = ["Id","ham"]
Resultado2 = pd.DataFrame(columns = x2, data = result2)
Resultado.to_csv("resultados_knn_2.csv", index = False)
Xtrain_nb = train.iloc [:,1:58]
Ytrain_nb = train.ham
Xtest_nb = test.iloc [:,1:58]
Nb = BernoulliNB(binarize = 0.2)
Nb.fit (Xtrain_nb,Ytrain_nb)
c_val = 10
scores = cross_val_score (Nb, Xtrain_nb,Ytrain_nb, cv = c_val,scoring ='f1')
total = 0
for i in scores:
    total += i
acuracia_esperada = total/c_val
acuracia_esperada
Y_pred_nb = Nb.predict(Xtest_nb)
verd_nb= 0
fal_nb= 0
for i in Y_pred_nb:
    if i == True:
        verd_nb += 1
    else:
        fal_nb += 1
dicio_nb = {"Spam":fal_nb, "Ham": verd_nb}
plt.bar(range(len(dicio_nb)), list(dicio_nb.values()), align='center',color = ('blue','orange'))
plt.xticks(range(len(dicio_nb)), list(dicio_nb.keys()))
print ('Spam:', fal_nb)
print ('Ham:',verd_nb)
print ('Porcentagem de Spam: ', round(fal_nb/(verd_nb+fal_nb)*100,4), '%', sep = '')
result_nb= np.vstack((test["Id"], Y_pred_nb)).T
x_nb = ["Id","ham"]
Resultado_nb = pd.DataFrame(columns = x_nb, data = result_nb)
Resultado_nb.to_csv("resultados_bayes_nb.csv", index = False)
melhor = [0,0]
for z in range (1,len(test.columns)-1):
    X_new = SelectKBest(f_regression, k=z).fit(train.iloc[:,0:len(test.columns)-1], train.ham)
    a = X_new.get_support (indices = True)
    b = []
    for i in a:
        b.append (i)
    Xtrain_b = train[train.columns[b]]
    Ytrain_b = train.ham
    Xtest_b = test[test.columns[b]]
    Nb1 = BernoulliNB(binarize = 0.2)
    Nb1.fit (Xtrain_b,Ytrain_b)
    c_val = 15
    scores = cross_val_score (Nb1, Xtrain_b, Ytrain_b, cv=c_val)
    total = 0
    for i in scores:
        total += i
    acuracia_esperada = total/c_val
    if acuracia_esperada > melhor[0]:
        melhor[0] = acuracia_esperada
        melhor[1] = z
print (melhor)
X_new = SelectKBest(f_regression, k=melhor[1]).fit(train.iloc[:,0:len(test.columns)-1], train.ham)
a = X_new.get_support (indices = True)
b = []
for i in a:
    b.append (i)
Xtrain_b = train[train.columns[b]]
Ytrain_b = train.ham
Xtest_b = test[test.columns[b]]
Nb_b = BernoulliNB(binarize = 0.2)
Nb_b.fit (Xtrain_b,Ytrain_b)
c_val = 15
scores = cross_val_score (Nb_b, Xtrain_b, Ytrain_b, cv=c_val)
total = 0
for i in scores:
    total += i
acuracia_esperada = total/c_val
acuracia_esperada
Y_pred_b = Nb_b.predict(Xtest_b)
verd_b= 0
fal_b= 0
for i in Y_pred_b:
    if i == True:
        verd_b += 1
    else:
        fal_b += 1
dicio_b = {"Spam":fal_b, "Ham": verd_b}
plt.bar(range(len(dicio_b)), list(dicio_b.values()), align='center',color = ('blue','orange'))
plt.xticks(range(len(dicio_b)), list(dicio_b.keys()))
print ('Spam:', fal_b)
print ('Ham:',verd_b)
print ('Porcentagem de Spam: ', fal_b/(verd_b+fal_b)*100, '%', sep = '')
result3= np.vstack((test["Id"], Y_pred_b)).T
x3 = ["Id","ham"]
Resultado3 = pd.DataFrame(columns = x3, data = result3)
Resultado3.to_csv("resultados_bayes_kbest.csv", index = False)

from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
svm = LinearSVC()
melhor3 = [0,0,0]
for i in range (1,57):
    rfe = RFE(svm,i)
    fit = rfe.fit(train.iloc[:,0:len(test.columns)-1], train.ham)
    d = fit.support_
    a = [ l for l,e in enumerate (d) if e == True]
    Xtrain_c = train[train.columns[a]]
    Ytrain_c = train.ham
    Xtest_c = test[test.columns[a]]
    Nb3 = BernoulliNB(binarize = 0)
    Nb3.fit (Xtrain_c,Ytrain_c)
    c_val = 10
    scores = cross_val_score (Nb3, Xtrain_c, Ytrain_c, cv=c_val, scoring ='f1')
    total = 0
    for j in scores:
        total += j
    acuracia_esperada = total/c_val
    if acuracia_esperada > melhor3[0]:          
        melhor3[0] = acuracia_esperada
        melhor3[1] = i
        melhor3[2] = a    
melhor3
Xtrain_c = train[train.columns[melhor3[2]]]
Ytrain_c = train.ham
Xtest_c = test[test.columns[melhor3[2]]]
Nb3 = BernoulliNB(binarize = 0.0)
Nb3.fit (Xtrain_c,Ytrain_c)
c_val = 10
scores = cross_val_score (Nb3, Xtrain_c, Ytrain_c, cv=c_val, scoring = 'f1')
total = 0
for i in scores:
    total += i
acuracia_esperada = total/c_val
acuracia_esperada
Y_pred3 = Nb3.predict(Xtest_c)
verd3= 0
fal3= 0
for i in Y_pred3:
    if i == True:
        verd3 += 1
    else:
        fal3 += 1
dicio3 = {"Spam":fal3, "Ham": verd3}
plt.bar(range(len(dicio3)), list(dicio3.values()), align='center',color = ('blue','orange'))
plt.xticks(range(len(dicio3)), list(dicio3.keys()))
print ('Spam:', fal3)
print ('Ham:',verd3)
print ('Porcentagem de Spam: ', round(fal3/(verd3+fal3)*100,4), '%', sep = '')
result3= np.vstack((test["Id"], Y_pred3)).T
x3 = ["Id","ham"]
Resultado3 = pd.DataFrame(columns = x3, data = result3)
Resultado3.to_csv("resultados_bayes_rfe.csv", index = False)
svm = LinearSVC()
melhor4 = [0,0,0]
for i in range (1,57):
    rfe = RFE(svm,i)
    fit = rfe.fit(train.iloc[:,0:len(test.columns)-1], train.ham)
    d = fit.support_
    a = [ l for l,e in enumerate (d) if e == True ]
    Xtrain_d = train[train.columns[a]]
    Ytrain_d = train.ham
    Xtest_d = test[test.columns[a]]
    Nb4 = BernoulliNB(binarize = 0.2)
    Nb4.fit (Xtrain_d,Ytrain_d)
    c_val = 10
    scores = cross_val_score (Nb4, Xtrain_d, Ytrain_d, cv=c_val)
    total = 0
    for j in scores:
        total += j
    acuracia_esperada = total/c_val
    if acuracia_esperada > melhor4[0]:        
        melhor4[0] = acuracia_esperada
        melhor4[1] = i
        melhor4[2] = a    
melhor4
Xtrain_d = train[train.columns[melhor4[2]]]
Ytrain_d = train.ham
Xtest_d = test[test.columns[melhor4[2]]]
Nb4 = BernoulliNB(binarize = 0.2)
Nb4.fit (Xtrain_d,Ytrain_d)
c_val = 15
scores = cross_val_score (Nb4, Xtrain_d, Ytrain_d, cv=c_val,scoring = 'f1')
total = 0
for i in scores:
    total += i
acuracia_esperada = total/c_val
acuracia_esperada
Y_pred4 = Nb4.predict(Xtest_d)
verd4= 0
fal4= 0
for i in Y_pred4:
    if i == True:
        verd4 += 1
    else:
        fal4 += 1
dicio4 = {"Spam":fal4, "Ham": verd4}
plt.bar(range(len(dicio4)), list(dicio4.values()), align='center',color = ('blue','orange'))
plt.xticks(range(len(dicio4)), list(dicio4.keys()))
print ('Spam:', fal4)
print ('Ham:',verd4)
print ('Porcentagem de Spam: ', round(fal4/(verd4+fal4)*100,4), '%', sep = '')
result4= np.vstack((test["Id"], Y_pred4)).T
x4 = ["Id","ham"]
Resultado4 = pd.DataFrame(columns = x4, data = result4)
Resultado4.to_csv("resultados_bayes_rfe_mod.csv", index = False)