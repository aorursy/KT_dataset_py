import pandas as pd
import sklearn
import numpy as np
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt
%matplotlib inline
os.listdir("../input")
householdData = pd.read_csv("../input/train.csv",
        engine='python')

householdTest = pd.read_csv("../input/test.csv",
        engine='python')
numHouseHold = householdData.replace('no', 0)
numHouseHold = numHouseHold.replace('yes', 1)
numHouseHold['v2a1'] = numHouseHold['v2a1'].fillna(0)
numHouseHold['v18q1'] = numHouseHold['v18q1'].fillna(0)
householdData['rez_esc'].value_counts().idxmax()
numHouseHold['rez_esc'] = numHouseHold['rez_esc'].fillna(0)
numHouseHold = numHouseHold.fillna(-1)
numHouseHold.head()
numHouseHold["Target"].value_counts().plot(kind='pie')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
XHouseHold = numHouseHold.drop(['Id', 'idhogar', 'Target'], axis=1)
YHouseHold = numHouseHold.Target
knn = KNeighborsClassifier(n_neighbors=233)
knn.fit(XHouseHold,YHouseHold)
scores = cross_val_score(knn, XHouseHold, YHouseHold, cv=15)
scores
scores.mean()
testHouseHold = householdTest.replace('no', 0)
testHouseHold = testHouseHold.replace('yes', 1)
testHouseHold['v2a1'] = testHouseHold['v2a1'].fillna(0)
testHouseHold['v18q1'] = testHouseHold['v18q1'].fillna(0)
testHouseHold['rez_esc'] = testHouseHold['rez_esc'].fillna(0)
testHouseHold = testHouseHold.fillna(-1)
testHouseHold.head()
XTestHouseHold = testHouseHold.drop(['Id', 'idhogar'], axis=1)
YHouseHold_Pred = knn.predict(XTestHouseHold)
YHouseHold_Pred
Id = testHouseHold['Id']
idhogar = testHouseHold['idhogar']
envio = pd.DataFrame()
envio.insert(0, 'Id', Id)
envio.insert(1, 'Target', YHouseHold_Pred)
envio.head()
envio.to_csv("HouseHoldData_KNN.csv", index=False)
