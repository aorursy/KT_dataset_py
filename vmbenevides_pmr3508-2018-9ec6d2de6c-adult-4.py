import scipy
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
#Algoritmos de classificação
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
train1 = pd.read_csv("../input/tarefa4/train_data.csv")
test1 = pd.read_csv("../input/tarefa4/test_data.csv")
print ("Train antes de tirar na:", train1.shape)
print ("Test antes de tirar na:", test1.shape)
train2 = train1.dropna()
test2 = test1.dropna()
print ("Train depois de tirar na:", train2.shape)
print ("Test depois de tirar na:", test2.shape)
train2['capital'] = train2['capital.gain']-train2['capital.loss']
test2['capital'] = test2['capital.gain']-train2['capital.loss']
train2['income'] = train2['income'].apply(lambda x: 1 if x == '>50K' else 0)

a = train2['income']
train2 = train2.drop (columns = ['income'])
train2['income'] = a
train2.head ()
test2.head ()
train2["native.country"].value_counts().head()
train2["native.country"].value_counts().plot('pie')
train2['native.country'] = train2['native.country'].apply(lambda x: 1 if x == 'United-States' else 0)
test2['native.country'] = test2['native.country'].apply(lambda x: 1 if x == 'United-States' else 0)
train2["native.country"].value_counts().plot('pie')
scipy.stats.pearsonr(train2['native.country'],train2['income'])
train2["education.num"].value_counts()
train2["education.num"].value_counts().plot('bar')
scipy.stats.pearsonr(train2['education.num'],train2['income'])
train2["age"].value_counts().head()
train2["age"].value_counts().plot('bar')
scipy.stats.pearsonr(train2['age'],train2['income'])
train2["race"].value_counts()
train2["race"].value_counts().plot('bar')
train2["race"] = train2['race'].apply(lambda x: 1 if x == 'White' else 0)
test2["race"] = test2['race'].apply(lambda x: 1 if x == 'White' else 0)
train2["race"].value_counts().plot('bar')
scipy.stats.pearsonr(train2['race'],train2['income'])
train2["sex"].value_counts()
train2["sex"].value_counts().plot('bar')
train2["sex"] = train2['sex'].apply(lambda x: 1 if x == 'Male' else 0)
test2["sex"] = test2['sex'].apply(lambda x: 1 if x == 'Male' else 0)
train2["sex"].value_counts().plot('bar')
scipy.stats.pearsonr(train2['sex'],train2['income'])
train2['capital.gain'].value_counts().head()
train2['capital.gain'].value_counts().plot('pie')
scipy.stats.pearsonr(train2['capital.gain'],train2['income'])
train2['capital.loss'].value_counts().head()
train2['capital.loss'].value_counts().plot('pie')
scipy.stats.pearsonr(train2['capital.loss'],train2['income'])
train2['capital'].value_counts().head()
train2['capital'].value_counts().plot('pie')
scipy.stats.pearsonr(train2['capital'],train2['income'])
train2['fnlwgt'].value_counts().head()
scipy.stats.pearsonr(train2['fnlwgt'],train2['income'])
train2['hours.per.week'].value_counts().head()
train2['hours.per.week'].value_counts ().plot('pie')
scipy.stats.pearsonr(train2['hours.per.week'],train2['income'])
train2['marital.status'].value_counts()
train2['marital.status'].value_counts().plot('bar')
train2['marital.status'] = train2['marital.status'].apply(lambda x: 1 if x == 'Married-civ-spouse' else 0)
test2['marital.status'] = test2['marital.status'].apply(lambda x: 1 if x == 'Married-civ-spouse' else 0)
train2['marital.status'].value_counts().plot('bar')
scipy.stats.pearsonr(train2['marital.status'],train2['income'])
train2['relationship'].value_counts()
train2['relationship'].value_counts().plot('bar')
train2['relationship'] = train2['relationship'].apply(lambda x: 2 if x == 'Husband' else 1 if x == "Wife" else 0)
test2['relationship'] = test2['relationship'].apply(lambda x: 2 if x == 'Husband' else 1 if x == "Wife" else 0)
train2['relationship'].value_counts().plot('bar')
scipy.stats.pearsonr(train2['relationship'],train2['income'])
train2['workclass'].value_counts()
train2['workclass'].value_counts().plot('bar')
train2['workclass'] = train2['workclass'].apply(lambda x: 1 if x in ['Self-emp-not-inc','Self-emp-inc'] else 0)
test2['workclass'] = test2['workclass'].apply(lambda x: 1 if x in ['Self-emp-not-inc','Self-emp-inc'] else 0)
train2['workclass'].value_counts().plot('bar')
scipy.stats.pearsonr(train2['workclass'],train2['income'])
train2['occupation'].value_counts()
train2['occupation'].value_counts().plot('bar')
train2['occupation'] = train2['occupation'].apply(lambda x: 1 if x in 
                                                  ['Prof-specialty','Exec-managerial','Protective-serv'] else 0)
test2['occupation'] = test2['occupation'].apply(lambda x: 1 if x in 
                                                  ['Prof-specialty','Exec-managerial','Protective-serv'] else 0)
train2['occupation'].value_counts().plot('bar')
scipy.stats.pearsonr(train2['occupation'],train2['income'])
plt.matshow(train2.corr())
train2.corr().iloc[1:-1, -1].plot('bar').grid()
Xtrain1 = train2[['age','education.num','marital.status','relationship','race','sex','capital',
                  'hours.per.week','native.country','occupation']]
Ytrain1 = train2['income']
Xtest1 = test2[['age','education.num','marital.status','relationship','race','sex','capital',
                 'hours.per.week','native.country','occupation']]
m1 = [0,0]
for i in range (20,26):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit (Xtrain1,Ytrain1)
    c_val = 10
    scores1 = cross_val_score(knn, Xtrain1, Ytrain1, cv=c_val)
    scores1 = scores1.mean()
    if scores1 > m1[1]:
        m1[1] = scores1
        m1[0] = i
print (m1)
knn = KNeighborsClassifier(n_neighbors=m1[0])
knn.fit (Xtrain1,Ytrain1)
Ytest1 = knn.predict (Xtest1)
result1 = np.vstack((test2["Id"], Ytest1)).T
x1 = ["Id","income"]
Resultado_1 = pd.DataFrame(columns = x1, data = result1)
Xtrain2 = train2[['age','education.num','marital.status','relationship','race','sex','capital',
                  'hours.per.week','native.country','occupation','workclass']]
Ytrain2 = train2['income']
Xtest2 = test2[['age','education.num','marital.status','relationship','race','sex','capital',
                 'hours.per.week','native.country','occupation','workclass']]
gnb = GaussianNB ()
gnb.fit (Xtrain2,Ytrain2)
c_val = 10
scores2 = cross_val_score (gnb, Xtrain2,Ytrain2, cv = c_val)
scores2 = scores2.mean ()
print (scores2)
Ytest2 = gnb.predict (Xtest2)
result2 = np.vstack((test2["Id"], Ytest2)).T
x2 = ["Id","income"]
Resultado_2 = pd.DataFrame(columns = x2, data = result2)
Xtrain3 = train2[['age','education.num','marital.status','relationship','race','sex','capital',
                  'hours.per.week','native.country','occupation','workclass']]
Ytrain3 = train2['income']
Xtest3 = test2[['age','education.num','marital.status','relationship','race','sex','capital',
                 'hours.per.week','native.country','occupation','workclass']]
n_est = 200
btc = BaggingClassifier (n_estimators = n_est, bootstrap_features = True)
btc.fit (Xtrain3,Ytrain3)
c_val = 10
scores3 = cross_val_score (btc,Xtrain3,Ytrain3, cv = c_val)
scores3 = scores3.mean()
print (scores3)
Ytest3 = btc.predict (Xtest3)
result3 = np.vstack((test2["Id"], Ytest3)).T
x3 = ["Id","income"]
Resultado_3 = pd.DataFrame(columns = x3, data = result3)
Xtrain4 = train2[['age','education.num','marital.status','relationship','race','sex','capital.gain','capital.loss','capital',
                  'hours.per.week','native.country','occupation','workclass']]
Ytrain4 = train2['income']
Xtest4 = test2[['age','education.num','marital.status','relationship','race','sex','capital.gain','capital.loss','capital',
                 'hours.per.week','native.country','occupation','workclass']]
n_est1 = 250
rdc = RandomForestClassifier (n_estimators = n_est1,criterion = 'gini',min_samples_split = 2,
                              min_samples_leaf = 5, max_depth = 25, max_features =  'sqrt')
rdc.fit (Xtrain4,Ytrain4)
c_val = 10
scores4 = cross_val_score (rdc,Xtrain4,Ytrain4, cv = c_val)
scores4 = scores4.mean()
print(scores4)
Ytest4 = rdc.predict (Xtest4)
result4 = np.vstack((test2["Id"], Ytest4)).T
x4 = ["Id","income"]
Resultado_4 = pd.DataFrame(columns = x4, data = result4)
Xtrain5 = train2[['age','education.num','marital.status','relationship','race','sex','capital.gain','capital.loss', 'capital',
                  'hours.per.week','native.country','occupation','workclass']]
Ytrain5 = train2['income']
Xtest5 = test2[['age','education.num','marital.status','relationship','race','sex','capital.gain','capital.loss', 'capital',
                 'hours.per.week','native.country','occupation','workclass']]
n_est2 = 200
etc = ExtraTreesClassifier (n_estimators = n_est2,criterion = 'gini',min_samples_split = 2,
                              min_samples_leaf = 5, max_depth = None)
etc.fit (Xtrain5,Ytrain5)
c_val = 10
scores5 = cross_val_score (etc,Xtrain5,Ytrain5, cv = c_val)
scores5 = scores5.mean()
print(scores5)
Ytest5 = etc.predict (Xtest5)
result5 = np.vstack((test2["Id"], Ytest5)).T
x5 = ["Id","income"]
Resultado_5 = pd.DataFrame(columns = x5, data = result5)
Resultado_1['income'] = Resultado_1['income'].apply(lambda x: '>50K' if x == 1 else '<=50K')
Resultado_2['income'] = Resultado_2['income'].apply(lambda x: '>50K' if x == 1 else '<=50K')
Resultado_3['income'] = Resultado_3['income'].apply(lambda x: '>50K' if x == 1 else '<=50K')
Resultado_4['income'] = Resultado_4['income'].apply(lambda x: '>50K' if x == 1 else '<=50K')
Resultado_5['income'] = Resultado_5['income'].apply(lambda x: '>50K' if x == 1 else '<=50K')
Resultado_1.to_csv("resultados_knn.csv", index = False)
Resultado_2.to_csv("resultados_nb.csv", index = False)
Resultado_3.to_csv("resultados_bagged_trees.csv", index = False)
Resultado_4.to_csv("resultados_random_forest.csv", index = False)
Resultado_5.to_csv("resultados_extra_trees.csv", index = False)
