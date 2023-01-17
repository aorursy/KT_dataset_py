import pandas as pd
import numpy as np
import sklearn
import os
from matplotlib import pyplot as plt
os.listdir("../input/")
#os.listdir("../input/adult-dataset")
filepath = "../input/adult-dataset/train_data.csv"
adult = pd.read_csv(filepath,
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult
listNull = [x for x in [(row,adult[row].isnull().sum()) for row in adult] if x[1]!=0]
print(listNull)
adult.drop(adult.index[0], inplace=True)
#adult["Country"].value_counts()
adult["Workclass"].value_counts()
adult["Occupation"].value_counts()
#adult["Age"].value_counts()
#adult["Sex"].value_counts()
#adult["Race"].value_counts()
firstW = ["United-States",
"Germany",
"Canada",
"England",
"Italy",
"Japan",
"Poland",
"Portugal",
"Taiwan",
"France",
"Greece",
"Ireland",
"Hong",
"Yugoslavia",
"Hungary",
"Scotland",
"Holand-Netherlands"]
adult["Race"] = adult["Race"].transform(lambda x: 1 if x=="White" else 0 if x==x else x)
adult["Country"] = adult["Country"].transform(lambda x: 1 if x in firstW else -1 if x == "Mexico" else 0 if x==x else x)
#adult["Sex"] = adult["Sex"].transform(lambda x: 1 if x=="Male" else 0 if x==x else x)
adult["Sex"].replace({"Male":1,"Female":0}, inplace=True)
adult["Age"] = adult["Age"].transform(lambda x: int(x))
#print(adult["Age"].mean())
#print(adult["Age"].std())
adult["Age"] = adult["Age"].transform(lambda x: 2*(x - x.mean()) / x.std())
#adult["Capital Gain"] = adult["Capital Gain"].transform(lambda x: int(x))
#adult["Capital Gain"] = adult["Capital Gain"].transform(lambda x: (x-x.mean()) / x.std())
#adult["Capital Loss"] = adult["Capital Loss"].transform(lambda x: int(x))
#adult["Capital Loss"] = adult["Capital Loss"].transform(lambda x: (x-x.mean()) / x.std())
adult["Workclass"] = adult["Workclass"].transform(lambda x: 1 if x in ["Without-pay", "Never-worked"] else 0 if x==x else x)
adult["Occupation"] = adult["Occupation"].transform(lambda x: 1 if x in ["Priv-house-serv","Handlers-cleaners","Transport-moving"] else 0 if x==x else x)
#adult.drop(["Occupation","Workclass"], axis='columns', inplace=True)
adult["MissingCountry"] = adult["Country"].transform(lambda x: 1 if x==x else 0)
adult["Country"].fillna(adult["Country"].mean(), inplace=True)
adult["MissingWorkclass"] = adult["Workclass"].transform(lambda x: 1 if x==x else 0)
adult["Workclass"].fillna(adult["Workclass"].mean(), inplace=True)
adult["MissingOccupation"] = adult["Occupation"].transform(lambda x:1 if x==x else 0)
adult["Occupation"].fillna(adult["Occupation"].mean(),inplace=True)
nadult = adult.dropna()
nadult["MissingCountry"].value_counts()
testfilepath = "../input/adult-dataset/test_data.csv"
testAdult = pd.read_csv(testfilepath,
        names=[
        "Id","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
testAdult
testAdult["Workclass"].value_counts()
testAdult.drop(testAdult.index[0], inplace=True)
testAdult.drop("Id",axis=1, inplace=True)
testAdult.drop("Target",axis=1, inplace=True)
listNull = [x for x in [(row,testAdult[row].isnull().sum()) for row in testAdult] if x[1]!=0]
print(listNull)
testAdult["Race"] = testAdult["Race"].transform(lambda x: 1 if x=="White" else 0)
testAdult["Country"] = testAdult["Country"].transform(lambda x: 1 if x in firstW else -1 if x=="Mexico" else 0)
testAdult["Sex"] = testAdult["Sex"].transform(lambda x: 1 if x=="Male" else 0)
testAdult["Age"] = testAdult["Age"].transform(int)
testAdult["Age"] = testAdult["Age"].transform(lambda x: 2*(x - x.mean()) / x.std())
#testAdult["Capital Gain"] = testAdult["Capital Gain"].transform(lambda x: int(x))
#testAdult["Capital Gain"] = testAdult["Capital Gain"].transform(lambda x: (x-x.mean()) / x.std())
#testAdult["Capital Loss"] = testAdult["Capital Loss"].transform(lambda x: int(x))
#testAdult["Capital Loss"] = testAdult["Capital Loss"].transform(lambda x: (x-x.mean()) / x.std())
testAdult["Workclass"] = testAdult["Workclass"].transform(lambda x: 1 if x in ["Without-pay", "Never-worked"] else 0)
testAdult["Occupation"] = testAdult["Occupation"].transform(lambda x: 1 if x in ["Priv-house-serv","Handlers-cleaners","Transport-moving"] else 0)
#testAdult.drop("Workclass", axis='columns', inplace=True)
testAdult["MissingCountry"] = testAdult["Country"].transform(lambda x: 1 if x==x else 0)
testAdult["Country"].fillna(testAdult["Country"].mean(), inplace=True)
testAdult["MissingWorkclass"] = testAdult["Workclass"].transform(lambda x: 1 if x==x else 0)
testAdult["Workclass"].fillna(testAdult["Workclass"].mean(), inplace=True)
testAdult["MissingOccupation"] = testAdult["Occupation"].transform(lambda x:1 if x==x else 0)
testAdult["Occupation"].fillna(testAdult["Occupation"].mean(),inplace=True)
#from sklearn.preprocessing import Imputer
#values = mydata.values
#imputer = Imputer(missing_values=’NaN’, strategy=’mean’)
#transformed_values = imputer.fit_transform(values)
testAdult["Country"].value_counts()
#testAdult["Sex"].value_counts()
#testAdult["Race"].value_counts()
#nTestAdult = testAdult.dropna()
nTestAdult = testAdult
nTestAdult
#Xadult = nadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
#Xadult = nadult[["Age","Education-Num","Race", "Sex","Capital Gain", "Capital Loss", "Hours per week", "Country"]]
Xadult = nadult[["Age", "Workclass","Education-Num","Occupation","Race", "Sex","Capital Gain", "Capital Loss", "Hours per week", "Country"]]
#Xadult = nadult[["Age", "Workclass","Education-Num","Occupation","Race", "Sex","Capital Gain", "Capital Loss", "Hours per week", "Country","MissingCountry", "MissingOccupation", "MissingWorkclass"]]
Yadult = nadult.Target
#XtestAdult = nTestAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
#XtestAdult = nTestAdult[["Age", "Education-Num","Race", "Sex","Capital Gain", "Capital Loss", "Hours per week", "Country"]]
XtestAdult = nTestAdult[["Age", "Workclass","Education-Num","Occupation","Race", "Sex","Capital Gain", "Capital Loss", "Hours per week", "Country"]]
#XtestAdult = nTestAdult[["Age", "Workclass","Education-Num","Occupation","Race", "Sex","Capital Gain", "Capital Loss", "Hours per week", "Country", "MissingCountry", "MissingOccupation", "MissingWorkclass"]]
XtestAdult
XtestAdult["Country"].value_counts()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=25)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
np.mean(scores)
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
YtestPred
savepath = "results.csv"
#YtestPred.to_csv(savepath)
#import numpy as np
#np.savetxt("foo.csv", YtestPred, delimiter=",")
ypanda = pd.DataFrame(YtestPred, columns = ["income"])
ypanda.to_csv(savepath, index_label="Id")
ypanda