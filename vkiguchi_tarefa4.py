import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from scipy import stats as st
import os
import matplotlib.colors as mcolors
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
os.listdir("../input/")
trainfilepath = "../input/train_data.csv"
testfilepath = "../input/test_data.csv"
#adult = pd.read_csv(trainfilepath, sep=r'\s*,\s*', engine='python', na_values='?')
#testAdult = pd.read_csv(testfilepath, sep=r'\s*,\s*', engine='python', na_values='?')
adult = pd.read_csv(trainfilepath,
        names=[
        "Id", "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.drop(adult.index[0], inplace=True)
testAdult = pd.read_csv(testfilepath,
        names=[
        "Id", "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
testAdult.drop(testAdult.index[0], inplace=True)

transformed = False
adult.shape
adult.head()
adult.Target.value_counts()
adult.Sex.value_counts()
adult.Race.value_counts()
numLabels = ["Age", "fnlwgt", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week"]
for label in numLabels:
    adult[label] = adult[label].transform(int)
    testAdult[label] = testAdult[label].transform(int)
adult["White"] = adult.Race.transform(lambda x: True if x=="White" else False if x==x else x)
#adult["Black"] = adult.Race.transform(lambda x: x=="Black")
testAdult["White"] = testAdult.Race.transform(lambda x: True if x=="White" else False if x==x else x)
#testAdult["Black"] = testAdult.Race.transform(lambda x: x=="Black")
if(transformed == False):
    adult.Target = adult.Target.transform(lambda x: x==">50K")
    adult.Sex = adult.Sex.transform(lambda x: True if x=="Male" else False if x==x else x)
    testAdult.Sex = testAdult.Sex.transform(lambda x: True if x=="Male" else False if x==x else x)
    transformed = True
adult.head()
adult["Country"].value_counts()
adult["Martial Status"].value_counts()
adult["Occupation"].value_counts()
adult.corr()["Target"]
Xadult = adult.drop(["Id", "Target"], axis="columns")
Yadult = adult.Target

XtestAdult = testAdult.drop("Id", axis="columns")
Xadult.head()
XtestAdult.head()
import category_encoders as ce

ce_binary = ce.BinaryEncoder(cols = ['Country', "Martial Status", "Occupation", "Workclass", "Relationship"])

binary_encoder = ce_binary.fit(Xadult, Yadult)

XencodedAdult = binary_encoder.transform(Xadult)
XencodedTestAdult = binary_encoder.transform(XtestAdult)
#Xadult["Country"] = Xadult["Country"].transform(lambda x: x=="United-States")
#XnumAdult = Xadult.drop(["Workclass", "Education", "Martial Status", "Occupation", "Relationship", "Race"], axis="columns")
XnumAdult = Xadult.drop(["Workclass", "Education", "Martial Status", "Occupation", "Relationship", "Race", "Country"], axis="columns")

#XnumEncodedAdult = XencodedAdult.drop(["Workclass", "Education", "Martial Status", "Occupation", "Relationship", "Race"], axis="columns")
#XnumEncodedAdult = XencodedAdult.drop(["Workclass", "Education", "Relationship", "Race"], axis="columns")
XnumEncodedAdult = XencodedAdult.drop(["Education", "Race"], axis="columns")
#XnumEncodedTestAdult = XencodedTestAdult.drop(["Workclass", "Education", "Martial Status", "Occupation", "Relationship", "Race"], axis="columns")
#XnumEncodedTestAdult = XencodedTestAdult.drop(["Workclass", "Education", "Relationship", "Race"], axis="columns")
XnumEncodedTestAdult = XencodedTestAdult.drop(["Education", "Race"], axis="columns")
from sklearn.decomposition import PCA
pca = PCA(n_components = 10)
pca.fit(XnumEncodedAdult)
print(pca.explained_variance_ratio_)
XprincipalAdult = pca.transform(XnumEncodedAdult)
knn = KNeighborsClassifier(n_neighbors=25)

scores = cross_val_score(knn, XnumAdult, Yadult, cv=10)
print(scores.mean())

scores = cross_val_score(knn, XnumEncodedAdult, Yadult, cv=10)
print(scores.mean())

scores = cross_val_score(knn, XprincipalAdult, Yadult, cv=10)
print(scores.mean())
tree = DecisionTreeClassifier(max_depth=80, min_samples_split=12, min_samples_leaf=2)
model = BaggingClassifier(tree, n_estimators=100, bootstrap=True, bootstrap_features=True, max_samples=1.0, max_features=1.0, random_state=0)


scores = cross_val_score(model, XnumAdult, Yadult, cv=10)
print(scores.mean())

scores = cross_val_score(model, XnumEncodedAdult, Yadult, cv=10)
print(scores.mean())

#scores = cross_val_score(model, XnumEncodedAdult.drop("White", axis="columns"), Yadult, cv=10)
#print(scores.mean())

#scores = cross_val_score(model, XprincipalAdult, Yadult, cv=10)
#print(scores.mean())

#from sklearn.decomposition import FastICA
#transformer = FastICA(n_components=15, random_state=0)
#scores = cross_val_score(model, transformer.fit_transform(XnumEncodedAdult), Yadult, cv=10)
#print(scores.mean())

model.fit(XnumEncodedAdult, Yadult)

adultPredict = model.predict(XnumEncodedTestAdult)

predictAdult = testAdult.apply(lambda x:x)
predictAdult["income"] = adultPredict
predictAdult["income"] = predictAdult["income"].transform(lambda x: ">50K" if x else "<=50K")

predictAdult.head()

savepath = "adultBaggedTreePredict.csv"
predictAdult.to_csv(savepath, index=False, columns=['Id', 'income'])
forest = RandomForestClassifier(n_estimators=100, max_depth=80, min_samples_split=12, random_state=0, min_samples_leaf=2)
#forest = RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_split=12, random_state=0, max_features="sqrt", min_samples_leaf=2)


scores = cross_val_score(forest, XnumAdult, Yadult, cv=10)
print(scores.mean())

scores = cross_val_score(forest, XnumEncodedAdult, Yadult, cv=10)
print(scores.mean())

scores = cross_val_score(forest, XprincipalAdult, Yadult, cv=10)
print(scores.mean())


forest.fit(XnumEncodedAdult, Yadult)

adultPredict = forest.predict(XnumEncodedTestAdult)

predictAdult = testAdult.apply(lambda x:x)
predictAdult["income"] = adultPredict
predictAdult["income"] = predictAdult["income"].transform(lambda x: ">50K" if x else "<=50K")

predictAdult.head()

savepath = "adultForestPredict.csv"
predictAdult.to_csv(savepath, index=False, columns=['Id', 'income'])
from sklearn.ensemble import ExtraTreesClassifier

extraT = ExtraTreesClassifier(n_estimators=50, max_depth=20, min_samples_split=12, random_state=0, min_samples_leaf=2, max_features="sqrt")

scores = cross_val_score(extraT, XnumAdult, Yadult, cv=10)
print(scores.mean())

scores = cross_val_score(extraT, XnumEncodedAdult, Yadult, cv=10)
print(scores.mean())

#extraT.fit(XculledTrainHouses, YculledTrainHouses)

#housePredict = extraT.predict(XtestHouses)

#predictHouses = testHouses.apply(lambda x:x)
#predictHouses["median_house_value"] = housePredict

#savepath = "houseExtraTreesPredict.csv"
#predictHouses.to_csv(savepath, index=False, columns=['Id', 'median_house_value'])
boosted = GradientBoostingClassifier(subsample=0.85, n_estimators=50, max_depth=2)
#boosted = GradientBoostingRegressor(loss='huber', subsample=0.85, n_estimators=5250, max_depth=2, alpha=0.25)
#boosted = GradientBoostingRegressor(loss='lad', subsample=0.85, n_estimators=320, max_depth=2)
#boosted = GradientBoostingRegressor(loss='quantile', subsample=0.85, n_estimators=320, max_depth=2, alpha=0.9)


scores = cross_val_score(boosted, XnumAdult, Yadult, cv=10)
print(scores.mean())

scores = cross_val_score(boosted, XnumEncodedAdult, Yadult, cv=10)
print(scores.mean())


#boosted.fit(XculledTrainHouses, YculledTrainHouses)

#housePredict = boosted.predict(XtestHouses)

#predictHouses = testHouses.apply(lambda x:x)
#predictHouses["median_house_value"] = housePredict

#savepath = "GradientForestPredict.csv"
#predictHouses.to_csv(savepath, index=False, columns=['Id', 'median_house_value'])
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

svm = SVC(gamma='auto', max_iter = 50, random_state=0)
scaler = StandardScaler()

scaler.fit(XnumAdult)
XscaledAdult = scaler.transform(XnumAdult)

scaler2 = StandardScaler()
scaler2.fit(XnumEncodedAdult)
XscaledEncodedAdult = scaler2.transform(XnumEncodedAdult)

#scores = cross_val_score(svm, XnumAdult, Yadult, cv=10)
#print(scores.mean())

scores = cross_val_score(svm, XscaledAdult, Yadult, cv=10)
print(scores.mean())

scores = cross_val_score(svm, XscaledEncodedAdult, Yadult, cv=10)
print(scores.mean())

#scores = cross_val_score(svm, XnumEncodedAdult, Yadult, cv=10)
#print(scores.mean())
