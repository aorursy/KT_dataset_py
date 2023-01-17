import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('../input/train_data.csv')
df
df.shape
df["age"].value_counts()
df["age"].mean()
df["occupation"].value_counts()
df["education"].value_counts().plot(kind="bar")
df["hours.per.week"].value_counts().plot(kind="pie")
df["workclass"].value_counts().plot(kind="pie")
auxdf = df[df != '?']
nadf = auxdf.dropna()
nadf

nadf.shape
len(df) - len(nadf)
tdf = pd.read_csv('../input/test_data.csv')
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
Xdf = nadf[["workclass_cat", "age", "capital.gain", "capital.loss"]]
Ydf = nadf.income

Xtdf = natdf[["workclass_cat", "age", "capital.gain", "capital.loss"]]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(Xdf,Ydf)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xdf, Ydf, cv=20)
scores
np.mean(scores)
from sklearn.naive_bayes import MultinomialNB
NBM = MultinomialNB()
NBM.fit(Xdf,Ydf)
scores2 = cross_val_score(NBM, Xdf, Ydf, cv=20)
scores2
np.mean(scores2)
from sklearn.naive_bayes import BernoulliNB
NBB = BernoulliNB()
NBB.fit(Xdf,Ydf)
scores3 = cross_val_score(NBB, Xdf, Ydf, cv=20)
scores3
np.mean(scores3)
from sklearn.naive_bayes import GaussianNB
NBG = GaussianNB()
NBG.fit(Xdf,Ydf)
scores4 = cross_val_score(NBG, Xdf, Ydf, cv=20)
scores4
np.mean(scores4)
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(random_state=0)
DTC.fit(Xdf,Ydf)
scores5 = cross_val_score(DTC, Xdf, Ydf, cv=20)
scores5
np.mean(scores5)
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=200, random_state=0)
RFC.fit(Xdf,Ydf)
scores6 = cross_val_score(RFC, Xdf, Ydf, cv=20)
scores6
np.mean(scores6)
from sklearn.svm import SVC
SVMC = SVC(gamma='auto')
SVMC.fit(Xdf,Ydf)
scores7 = cross_val_score(SVMC, Xdf, Ydf, cv=10)
scores7
np.mean(scores7)
from sklearn.neural_network import MLPClassifier
MLPC = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
MLPC.fit(Xdf,Ydf)
scores8 = cross_val_score(MLPC, Xdf, Ydf, cv=10)
scores8
np.mean(scores8)
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
bagging.fit(Xdf,Ydf)
scores9 = cross_val_score(bagging, Xdf, Ydf, cv=20)
scores9
np.mean(scores9)
from sklearn.ensemble import AdaBoostClassifier
ABC = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
ABC.fit(Xdf,Ydf)
scores10 = cross_val_score(ABC, Xdf, Ydf, cv=20)
scores10
np.mean(scores10)
YtPred = knn.predict(Xtdf)
YtPred
df_YtPred = pd.DataFrame(index=natdf.Id,columns=['income'])
df_YtPred['income'] = YtPred
dfc_YtPred = pd.DataFrame(index=range(0, len(tdf)),columns=['income'])
dfc_YtPred['income'] = df_YtPred['income']      
for i in range(0, len(dfc_YtPred)):
    if (dfc_YtPred['income'][i] != '<=50K' and dfc_YtPred['income'][i] != '>50K'):
        dfc_YtPred['income'][i] = '<=50K'
dfc_YtPred
YtPred2 = NBM.predict(Xtdf)
YtPred2
df_YtPred2 = pd.DataFrame(index=natdf.Id,columns=['income'])
df_YtPred2['income'] = YtPred2
dfc_YtPred2 = pd.DataFrame(index=range(0, len(tdf)),columns=['income'])
dfc_YtPred2['income'] = df_YtPred2['income'] 
for i in range(0, len(dfc_YtPred2)):
    if (dfc_YtPred2['income'][i] != '<=50K' and dfc_YtPred2['income'][i] != '>50K'):
        dfc_YtPred2['income'][i] = '<=50K'
dfc_YtPred2
YtPred3 = NBB.predict(Xtdf)
YtPred3
df_YtPred3 = pd.DataFrame(index=natdf.Id,columns=['income'])
df_YtPred3['income'] = YtPred3
dfc_YtPred3 = pd.DataFrame(index=range(0, len(tdf)),columns=['income'])
dfc_YtPred3['income'] = df_YtPred3['income'] 
for i in range(0, len(dfc_YtPred3)):
    if (dfc_YtPred3['income'][i] != '<=50K' and dfc_YtPred3['income'][i] != '>50K'):
        dfc_YtPred3['income'][i] = '<=50K'
dfc_YtPred3
YtPred4 = NBG.predict(Xtdf)
YtPred4
df_YtPred4 = pd.DataFrame(index=natdf.Id,columns=['income'])
df_YtPred4['income'] = YtPred4
dfc_YtPred4 = pd.DataFrame(index=range(0, len(tdf)),columns=['income'])
dfc_YtPred4['income'] = df_YtPred4['income'] 
for i in range(0, len(dfc_YtPred4)):
    if (dfc_YtPred4['income'][i] != '<=50K' and dfc_YtPred4['income'][i] != '>50K'):
        dfc_YtPred4['income'][i] = '<=50K'
dfc_YtPred4
YtPred5 = DTC.predict(Xtdf)
YtPred5
df_YtPred5 = pd.DataFrame(index=natdf.Id,columns=['income'])
df_YtPred5['income'] = YtPred5
dfc_YtPred5 = pd.DataFrame(index=range(0, len(tdf)),columns=['income'])
dfc_YtPred5['income'] = df_YtPred5['income'] 
for i in range(0, len(dfc_YtPred)):
    if (dfc_YtPred5['income'][i] != '<=50K' and dfc_YtPred5['income'][i] != '>50K'):
        dfc_YtPred5['income'][i] = '<=50K'
dfc_YtPred5
YtPred6 = RFC.predict(Xtdf)
YtPred6
df_YtPred6 = pd.DataFrame(index=natdf.Id,columns=['income'])
df_YtPred6['income'] = YtPred6
dfc_YtPred6 = pd.DataFrame(index=range(0, len(tdf)),columns=['income'])
dfc_YtPred6['income'] = df_YtPred6['income'] 
for i in range(0, len(dfc_YtPred)):
    if (dfc_YtPred6['income'][i] != '<=50K' and dfc_YtPred6['income'][i] != '>50K'):
        dfc_YtPred6['income'][i] = '<=50K'
dfc_YtPred6
YtPred7 = SVMC.predict(Xtdf)
YtPred7
df_YtPred7 = pd.DataFrame(index=natdf.Id,columns=['income'])
df_YtPred7['income'] = YtPred7
dfc_YtPred7 = pd.DataFrame(index=range(0, len(tdf)),columns=['income'])
dfc_YtPred7['income'] = df_YtPred7['income'] 
for i in range(0, len(dfc_YtPred)):
    if (dfc_YtPred7['income'][i] != '<=50K' and dfc_YtPred7['income'][i] != '>50K'):
        dfc_YtPred7['income'][i] = '<=50K'
dfc_YtPred7
YtPred8 = MLPC.predict(Xtdf)
YtPred8
df_YtPred8 = pd.DataFrame(index=natdf.Id,columns=['income'])
df_YtPred8['income'] = YtPred8
dfc_YtPred8 = pd.DataFrame(index=range(0, len(tdf)),columns=['income'])
dfc_YtPred8['income'] = df_YtPred8['income'] 
for i in range(0, len(dfc_YtPred)):
    if (dfc_YtPred8['income'][i] != '<=50K' and dfc_YtPred8['income'][i] != '>50K'):
        dfc_YtPred8['income'][i] = '<=50K'
dfc_YtPred8
YtPred9 = bagging.predict(Xtdf)
YtPred9
df_YtPred9 = pd.DataFrame(index=natdf.Id,columns=['income'])
df_YtPred9['income'] = YtPred9
dfc_YtPred9 = pd.DataFrame(index=range(0, len(tdf)),columns=['income'])
dfc_YtPred9['income'] = df_YtPred9['income'] 
for i in range(0, len(dfc_YtPred)):
    if (dfc_YtPred9['income'][i] != '<=50K' and dfc_YtPred9['income'][i] != '>50K'):
        dfc_YtPred9['income'][i] = '<=50K'
dfc_YtPred9
YtPred10 = ABC.predict(Xtdf)
YtPred10
df_YtPred10 = pd.DataFrame(index=natdf.Id,columns=['income'])
df_YtPred10['income'] = YtPred10
dfc_YtPred10 = pd.DataFrame(index=range(0, len(tdf)),columns=['income'])
dfc_YtPred10['income'] = df_YtPred10['income'] 
for i in range(0, len(dfc_YtPred)):
    if (dfc_YtPred10['income'][i] != '<=50K' and dfc_YtPred10['income'][i] != '>50K'):
        dfc_YtPred10['income'][i] = '<=50K'
dfc_YtPred10
dfc_YtPred.to_csv('KNNPred.csv')
dfc_YtPred2.to_csv('NBMPred.csv')
dfc_YtPred3.to_csv('NBBPred.csv')
dfc_YtPred4.to_csv('NBGPred.csv')
dfc_YtPred5.to_csv('DTCPred.csv')
dfc_YtPred6.to_csv('RFCPred.csv')
dfc_YtPred7.to_csv('SVMCPred.csv')
dfc_YtPred8.to_csv('MLPCPred.csv')
dfc_YtPred9.to_csv('baggingPred.csv')
dfc_YtPred10.to_csv('ABCPred.csv')