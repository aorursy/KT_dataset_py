import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn import preprocessing
adult_data = pd.read_csv("/kaggle/input/ucirepository/adult_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
adult_test = pd.read_csv("/kaggle/input/ucirepository/adult_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
adult_data.shape
adult_test.shape
adult_data["Country"].value_counts()
adult_data["Age"].plot(kind="hist",bins=73)
adult_data["Sex"].value_counts().plot(kind="bar")
adult_data["Target"].value_counts().plot(kind="bar")
adult_data["Race"].value_counts().plot(kind="bar")
adult_data
adult_test
adult_data.isnull().sum()
adult_test.isnull().sum()
adult_dt = adult_data.dropna()
adult_tst = adult_test.dropna()
adult_dt.shape
adult_tst.shape
adult_dt.describe()
cols = ['Age', 'Capital Loss', 'Capital Gain', 'Education-Num', 'fnlwgt', 'Hours per week']

sns.set()

sns.pairplot(adult_dt, vars = cols, hue = 'Target')
def freq(column):

    return column*100//float(column[-1])
Income_Country = pd.crosstab(adult_dt.Target, adult_dt.Country, margins=True)

Income_Country.apply(freq, axis=0)
Income_Sex = pd.crosstab(adult_dt.Target, adult_dt.Sex, margins=True)

Income_Sex.apply(freq, axis=0)
Income_Race = pd.crosstab(adult_dt.Target, adult_dt.Race, margins=True)

Income_Race.apply(freq, axis=0)
Income_Sex.apply(freq, axis=0).plot(kind="bar")
Income_Race.apply(freq, axis=0).plot(kind="bar")
Xadult = adult_dt[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
Yadult = adult_dt.Target
XtestAdult = adult_tst[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
YtestAdult = adult_tst.Target
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
YtestPred
accuracy_score(YtestAdult,YtestPred)
nAdult_dt = adult_dt.apply(preprocessing.LabelEncoder().fit_transform)
nAdult_tst = adult_tst.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = nAdult_dt.iloc[:,0:14]
Yadult = nAdult_dt.Target
XtestAdult = nAdult_tst.iloc[:,0:14]
YtestAdult = nAdult_tst.Target
knn = KNeighborsClassifier(n_neighbors=30)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
YtestPred
accuracy_score(YtestAdult,YtestPred)
Xadult = nAdult_dt[["Age", "Workclass", "Education-Num", 

        "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week"]]
XtestAdult = nAdult_tst[["Age", "Workclass", "Education-Num", 

        "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week"]]
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(Xadult,Yadult)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

           metric_params=None, n_jobs=1, n_neighbors=30, p=2,

           weights='uniform')
YtestPred = knn.predict(XtestAdult)
accuracy_score(YtestAdult,YtestPred)
df_out = pd.DataFrame({'income':YtestPred})
df_out.head()
df_out.to_csv("submission.csv", index = True, index_label = 'Id')