import os
os.listdir('../input/costar')
import pandas as pd
import sklearn
costaRican = pd.read_csv('../input/costar/train.csv')
ncostaRican = costaRican.dropna()
testCostaRican = pd.read_csv('../input/costar/test.csv')
nTestCostaRican = testCostaRican.dropna()
from sklearn import preprocessing
numcostaRican = ncostaRican.apply(preprocessing.LabelEncoder().fit_transform)
numTestCostaRican = nTestCostaRican.apply(preprocessing.LabelEncoder().fit_transform)
XcostaRican = numcostaRican[["rooms", "escolari", "abastaguadentro", "public", "planpri", "noelec", "sanitario1", "sanitario2", "sanitario3", "sanitario5",
                           "sanitario6", "energcocinar1", "energcocinar2", "energcocinar3", "energcocinar4", "epared1", "etecho1", "eviv1", "male", "estadocivil1",
                           "estadocivil2", "parentesco1", "parentesco2", "parentesco3", "hogar_nin", "hogar_adul", "hogar_mayor", "hogar_total", "dependency",
                           "instlevel1", "instlevel2", "instlevel3", "bedrooms", "overcrowding", "tipovivi1", "tipovivi2", "computer", "television", "mobilephone",
                           "qmobilephone", "lugar1", "lugar2", "area1", "age"]]
YcostaRican = ncostaRican.Target
XTestcostaRican = numTestCostaRican[["rooms", "escolari", "abastaguadentro", "public", "planpri", "noelec", "sanitario1", "sanitario2", "sanitario3", "sanitario5",
                           "sanitario6", "energcocinar1", "energcocinar2", "energcocinar3", "energcocinar4", "epared1", "etecho1", "eviv1", "male", "estadocivil1",
                           "estadocivil2", "parentesco1", "parentesco2", "parentesco3", "hogar_nin", "hogar_adul", "hogar_mayor", "hogar_total", "dependency",
                           "instlevel1", "instlevel2", "instlevel3", "bedrooms", "overcrowding", "tipovivi1", "tipovivi2", "computer", "television", "mobilephone",
                           "qmobilephone", "lugar1", "lugar2", "area1", "age"]]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(XcostaRican, YcostaRican)
nTestCostaRican.shape
YtestPred = knn.predict(XTestcostaRican)
import numpy as np
array = np.vstack((nTestCostaRican.index.values, YtestPred)).T
final = pd.DataFrame(columns=['id', 'income'], data=array)
final.to_csv('resultsCostaRican.csv', index = False)
