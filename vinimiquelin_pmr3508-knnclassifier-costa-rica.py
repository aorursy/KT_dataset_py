import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('../input/train.csv')
pd.set_option('display.max_rows', 200)  # para visualizar todas as linhas das análises
pd.set_option('display.max_columns', 200) # para visualizar todas as colunas da tabela 

df
df.shape
df.columns[df.isna().any()].tolist()
df.dtypes == object
df["rooms"].value_counts().plot(kind="bar")
df["estadocivil1"].value_counts().plot(kind="pie")
df["parentesco1"].value_counts().plot(kind="pie")
df["overcrowding"].value_counts().plot(kind="pie")
df["mobilephone"].value_counts().plot(kind="bar")
df["SQBescolari"].value_counts().plot(kind="bar")
df["SQBhogar_total"].value_counts().plot(kind="bar")
df["SQBdependency"].value_counts().plot(kind="pie")
nadf = df.dropna()
nadf

nadf.shape
len(df) - len(nadf)
tdf = pd.read_csv('../input/test.csv')
tdf
tdf.shape
natdf = tdf.dropna()
natdf
natdf.shape
len(tdf) - len(natdf)
Xdf = df[["hacdor", "rooms", "hacapo", "v14a", "refrig", "r4h1", "r4h2", "r4h3", "r4m1", "r4m2", "r4m3", "r4t1", "r4t2", "r4t3", "tamhog", "tamviv", "escolari", "hhsize", "paredblolad", "paredzocalo", "paredpreb", "pareddes", "paredmad", "paredzinc", "paredfibras", "paredother", "pisomoscer", "pisocemento", "pisoother", "pisonatur", "pisonotiene", "pisomadera", "techozinc", "techoentrepiso", "techocane", "techootro", "cielorazo", "abastaguadentro", "abastaguafuera", "abastaguano", "public", "planpri", "noelec", "coopele", "sanitario1", "sanitario2", "sanitario3", "sanitario5", "sanitario6", "energcocinar1", "energcocinar2", "energcocinar3", "energcocinar4", "elimbasu1", "elimbasu2", "elimbasu3", "elimbasu4", "elimbasu5", "elimbasu6", "epared1", "epared2", "epared3", "etecho1", "etecho2", "etecho3", "eviv1", "eviv2", "eviv3", "dis", "male", "female", "estadocivil1", "estadocivil2", "estadocivil3", "estadocivil4", "estadocivil5", "estadocivil6", "estadocivil7", "parentesco1", "parentesco2", "parentesco3", "parentesco4", "parentesco5", "parentesco6", "parentesco7", "parentesco8", "parentesco9", "parentesco10", "parentesco11", "parentesco12", "hogar_nin", "hogar_adul", "hogar_mayor", "hogar_total", "instlevel1", "instlevel2", "instlevel3", "instlevel4", "instlevel5", "instlevel6", "instlevel7", "instlevel8", "instlevel9", "bedrooms", "overcrowding", "tipovivi1", "tipovivi2", "tipovivi3", "tipovivi4", "tipovivi5", "computer", "television", "mobilephone", "qmobilephone", "lugar1", "lugar2", "lugar3", "lugar4", "lugar5", "lugar6", "area1", "area2", "age", "SQBescolari", "SQBage", "SQBhogar_total", "SQBedjefe", "SQBhogar_nin", "SQBovercrowding", "SQBdependency", "agesq"]]
Ydf = df.Target

Xtdf = tdf[["hacdor", "rooms", "hacapo", "v14a", "refrig", "r4h1", "r4h2", "r4h3", "r4m1", "r4m2", "r4m3", "r4t1", "r4t2", "r4t3", "tamhog", "tamviv", "escolari", "hhsize", "paredblolad", "paredzocalo", "paredpreb", "pareddes", "paredmad", "paredzinc", "paredfibras", "paredother", "pisomoscer", "pisocemento", "pisoother", "pisonatur", "pisonotiene", "pisomadera", "techozinc", "techoentrepiso", "techocane", "techootro", "cielorazo", "abastaguadentro", "abastaguafuera", "abastaguano", "public", "planpri", "noelec", "coopele", "sanitario1", "sanitario2", "sanitario3", "sanitario5", "sanitario6", "energcocinar1", "energcocinar2", "energcocinar3", "energcocinar4", "elimbasu1", "elimbasu2", "elimbasu3", "elimbasu4", "elimbasu5", "elimbasu6", "epared1", "epared2", "epared3", "etecho1", "etecho2", "etecho3", "eviv1", "eviv2", "eviv3", "dis", "male", "female", "estadocivil1", "estadocivil2", "estadocivil3", "estadocivil4", "estadocivil5", "estadocivil6", "estadocivil7", "parentesco1", "parentesco2", "parentesco3", "parentesco4", "parentesco5", "parentesco6", "parentesco7", "parentesco8", "parentesco9", "parentesco10", "parentesco11", "parentesco12", "hogar_nin", "hogar_adul", "hogar_mayor", "hogar_total", "instlevel1", "instlevel2", "instlevel3", "instlevel4", "instlevel5", "instlevel6", "instlevel7", "instlevel8", "instlevel9", "bedrooms", "overcrowding", "tipovivi1", "tipovivi2", "tipovivi3", "tipovivi4", "tipovivi5", "computer", "television", "mobilephone", "qmobilephone", "lugar1", "lugar2", "lugar3", "lugar4", "lugar5", "lugar6", "area1", "area2", "age", "SQBescolari", "SQBage", "SQBhogar_total", "SQBedjefe", "SQBhogar_nin", "SQBovercrowding", "SQBdependency", "agesq"]]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xdf, Ydf, cv=20)
scores

np.mean(scores)
knn.fit(Xdf,Ydf)
YtPred = knn.predict(Xtdf)
YtPred
df_YtPred = pd.DataFrame(index=tdf.Id,columns=['Target'])
df_YtPred['Target'] = YtPred
df_YtPred
df_YtPred.shape
df_YtPred.to_csv('myPred2.csv')