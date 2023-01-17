import pandas as pd
import sklearn
adult = pd.read_csv("../input/adult-dataset/train_data.csv",names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],sep=r'\s*,\s*',
        engine='python', na_values="?")
  
adult.shape
adult.head()
adult["Education"].value_counts()
nadult = adult.dropna().drop(adult.index[0])
nadult.head()
adult_test = pd.read_csv("../input/adultdb/test_data.csv",names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],sep=r'\s*,\s*',
        engine='python', na_values="?")
adult_test.index[0]
adult_test = adult_test.drop(adult_test.index[0])
nadult_test = adult_test.dropna()
nadult_test.index[0]
Xadult = nadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
Yadult = nadult.Target
nadult[["Hours per week"]].head
XtestAdult = nadult_test[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
YtestAdult = nadult_test.Target
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
YtestPred
from sklearn.metrics import accuracy_score
accuracy_score(YtestAdult,YtestPred)
knn30 = KNeighborsClassifier(n_neighbors=30)
scores30 = cross_val_score(knn30, Xadult, Yadult, cv=10)
scores30
knn30.fit(Xadult,Yadult)
YtestPred30 = knn30.predict(XtestAdult)
YtestPred30
accuracy_score(YtestAdult,YtestPred30)

