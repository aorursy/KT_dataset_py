import pandas as pd
import sklearn
import numpy
cr_test = pd.read_csv("../input/test.csv")
cr_train = pd.read_csv("../input/train.csv")
ncr_test = cr_test.dropna()
ncr_train = cr_train.dropna()
cr_test.shape
ncr_test.shape
cr_train.shape
ncr_train.shape
cr_train.isnull().sum()
cr_train = cr_train.drop(["v2a1","v18q1","rez_esc"],axis = 1)
cr_train.isnull().sum()
cr_train.shape
ncr_train = cr_train.dropna()
ncr_train.shape
cr_test = cr_test.drop(["v2a1","v18q1","rez_esc"],axis = 1)
cr_test.drop(['agesq','SQBovercrowding','SQBdependency','SQBescolari','SQBage','SQBhogar_total','SQBedjefe','SQBhogar_nin'],axis=1)
cr_train.drop(['agesq','SQBovercrowding','SQBdependency','SQBescolari','SQBage','SQBhogar_total','SQBedjefe','SQBhogar_nin'],axis=1)
cr_test['dependency']
cr_test.shape
ncr_test = cr_test.dropna()
ncr_test.shape
Yncr_train = ncr_train.Target
from sklearn import preprocessing
num_train = ncr_train.apply(preprocessing.LabelEncoder().fit_transform)
num_test = ncr_test.apply(preprocessing.LabelEncoder().fit_transform)
Xncr_test = num_test.iloc[:,0:138]
Xncr_train = num_train.iloc[:,0:138]
from sklearn.neighbors import KNeighborsClassifier
for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(Xncr_train,Yncr_train)
    YtestPred = knn.predict(Xncr_test)
    a = 0
    b = 0
    c = 0
    d = 0
    for j in YtestPred:
        if j == 1:
            a += 1
        if j == 2:
            b += 1
        if j == 3:
            c += 1
        if j == 4:
            d += 1
    print('i = ', i)
    print(a/(a+b+c+d))
    print(b/(a+b+c+d))
    print(c/(a+b+c+d))
    print(d/(a+b+c+d))
a1 = 0
b1 = 0
c1 = 0
d1 = 0
for o in Yncr_train:
    if o == 1:
        a1 += 1
    if o ==2:
        b1 += 1
    if o == 3:
        c1 += 1
    if o == 4:
        d1 += 1
print(a1,b1,c1,d1)
        
print(a1/(a1+b1+c1+d1))
print(b1/(a1+b1+c1+d1))
print(c1/(a1+b1+c1+d1))
print(d1/(a1+b1+c1+d1))
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(Xncr_train,Yncr_train)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xncr_train, Yncr_train, cv=10)
scores
score_final = numpy.mean(scores)
score_final
