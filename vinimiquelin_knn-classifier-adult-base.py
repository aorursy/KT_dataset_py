import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('../input/mydataset/train_data.csv')
df
df.shape
df["occupation"].value_counts()
df["education"].value_counts().plot(kind="bar")
df["hours.per.week"].value_counts().plot(kind="pie")
df["workclass"].value_counts().plot(kind="pie")
auxdf = df[df != '?']
nadf = auxdf.dropna()
nadf

nadf.shape
len(df) - len(nadf)
tdf = pd.read_csv('../input/mydataset/test_data.csv')
tdf
tdf.shape
auxdf = tdf[tdf != '?']
natdf = auxdf.dropna()
natdf
natdf.shape
len(tdf) - len(natdf)
nadf.dtypes
nadf["occupation"] = nadf["occupation"].astype('category')
nadf["workclass"] = nadf["workclass"].astype('category')
nadf.dtypes
nadf["occupation_cat"] = nadf["occupation"].cat.codes
nadf["workclass_cat"] = nadf["workclass"].cat.codes
nadf
nadf.shape
natdf.dtypes
natdf["occupation"] = natdf["occupation"].astype('category')
natdf["workclass"] = natdf["workclass"].astype('category')
natdf.dtypes
natdf["occupation_cat"] = natdf["occupation"].cat.codes
natdf["workclass_cat"] = natdf["workclass"].cat.codes
natdf
natdf.shape
Xdf = nadf[["occupation_cat","education.num","hours.per.week", "workclass_cat", "age", "capital.gain", "capital.loss"]]
Ydf = nadf.income

Xtdf = natdf[["occupation_cat","education.num","hours.per.week", "workclass_cat", "age", "capital.gain", "capital.loss"]]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xdf, Ydf, cv=20)
scores

np.mean(scores)
knn.fit(Xdf,Ydf)
YtPred = knn.predict(Xtdf)
YtPred
df_YtPred = pd.DataFrame(index=natdf.Id,columns=['income'])
df_YtPred['income'] = YtPred
df_YtPred
dfc_YtPred = pd.DataFrame(index=range(0, len(tdf)),columns=['income'])
dfc_YtPred['income'] = df_YtPred['income']
dfc_YtPred      
for i in range(0, len(dfc_YtPred)):
    if (dfc_YtPred['income'][i] != '<=50K' and dfc_YtPred['income'][i] != '>50K'):
        dfc_YtPred['income'][i] = '<=50K'
dfc_YtPred
dfc_YtPred.to_csv('myPred.csv')