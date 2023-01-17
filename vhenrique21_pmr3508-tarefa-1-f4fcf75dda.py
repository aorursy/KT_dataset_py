import pandas as pd
import sklearn
import numpy as np
import os
from sklearn import preprocessing
os.listdir("../input/adult-db")
adultOriginal = pd.read_csv("../input/adult-db/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adultOriginal.head()

adultOriginal.shape
numAdult_1=adultOriginal.fillna(method='pad')
numAdult_2=numAdult_1.fillna(method='pad')
adult = numAdult_2.apply(preprocessing.LabelEncoder().fit_transform)
adult
import matplotlib.pyplot as plt
%matplotlib inline
adult["native.country"].value_counts()
adult["age"].value_counts().plot(kind="bar")
adult["sex"].value_counts().plot(kind="bar")
adult["race"].value_counts().plot(kind="bar")
adult["education"].value_counts().plot(kind="bar")
adult["occupation"].value_counts().plot(kind="bar")
test_adult = pd.read_csv('../input/adult-db/test_data.csv',
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
numAdultTest_1=test_adult.fillna(method='pad')
numAdultTest_2=numAdultTest_1.fillna(method='pad')
adultTest = numAdultTest_2.apply(preprocessing.LabelEncoder().fit_transform)
adultTest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
Xadult = adult.drop(['workclass', 'marital.status', 'sex', 'occupation', 'relationship', 'income', 'capital.gain', 'capital.loss', 'native.country'], axis=1)
Xadult
Yadult = adultOriginal.income
XtestAdult = adultTest.drop(['workclass', 'marital.status', 'sex', 'occupation', 'relationship', 'capital.gain', 'capital.loss', 'native.country'], axis=1)
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(Xadult,Yadult)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores.mean()
YtestPred = knn.predict(XtestAdult)
YtestPred
data = pd.DataFrame(adultTest.Id)
data["income"] = YtestPred
data
data.to_csv("BaseAdult_KNN.csv", index=False)

