import os
os.listdir('../input/adult-test')
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
adult = pd.read_csv('../input/adult-train/train_data (1).csv')
adult.head()
adult["native.country"].value_counts()
adult["occupation"].value_counts().plot(kind="bar")
nadult = adult.dropna()
testAdult = pd.read_csv('../input/adult-test/test_data (1).csv')
nTestAdult = testAdult.dropna()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
Xadult = nadult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Yadult = nadult.income
XtestAdult = nTestAdult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
import numpy as np
array = np.vstack((testAdult.index.values, YtestPred)).T
first = pd.DataFrame(columns=['id', 'income'], data=array)
first.to_csv('first-results.csv', index = False)
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult[["age", "workclass", "education.num", "marital.status",
        "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss",
        "hours.per.week", "native.country"]]
Yadult = nadult.income
XtestAdult = numTestAdult[["age", "workclass", "education.num", "marital.status",
        "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss",
        "hours.per.week", "native.country"]]
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
array = np.vstack((testAdult.index.values, YtestPred)).T

final = pd.DataFrame(columns=['id', 'income'], data=array)
final.to_csv('results.csv', index = False)