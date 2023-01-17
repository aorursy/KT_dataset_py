import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

from sklearn import neighbors, linear_model, svm, ensemble, preprocessing, cross_validation



%matplotlib inline
train = pd.read_csv("../input/Iris.csv")

train.head()
train.drop(["Id"], axis=1, inplace = True)
pd.DataFrame(train.isnull().sum(), columns=["Train Missing values"])
enc = preprocessing.LabelEncoder()

train["Species_enc"] = enc.fit_transform(train.Species)
train.head()
plt.figure(figsize=(9,2))

sns.heatmap(train.corr(), cmap="RdYlGn", annot=True)
plt.figure(figsize=(20,7))



plt.subplot2grid((1,2), (0,0))

sns.set_style("whitegrid")

sns.boxplot(data = train.drop(["Species_enc"], 1))

plt.title("Box plot of all features")



plt.subplot2grid((1,2), (0,1))

sns.violinplot(data = train.drop(["Species_enc"], 1))

plt.title("Violin plot of features")
plt.figure(figsize=(12,9))



plt.subplot2grid((2,2),(0,0))

sns.distplot(a = train.PetalWidthCm)

plt.title("Distribution of Petal width")



plt.subplot2grid((2,2),(0,1))

sns.distplot(train.PetalLengthCm)

plt.title("Distribution of Petal lenght")



plt.subplot2grid((2,2),(1,0))

sns.distplot(train.SepalLengthCm)

plt.title("Distribution of Sepal lenght")



plt.subplot2grid((2,2),(1,1))

sns.distplot(train.SepalWidthCm)

plt.title("Distribution of Sepal width")
plt.figure(figsize=(10,8))

g = sns.lmplot(data = train, x="SepalLengthCm", y = "SepalWidthCm", hue = "Species")

g.set(ylim=(1.8,5))

plt.title("Relationship between Sepal Length and Sepal Width and Species")
plt.figure(figsize=(15,7))

sns.lmplot(data = train, x="PetalLengthCm", y = "PetalWidthCm", hue = "Species")

plt.title("Relationship between Petal Length and Petal Width and Species")
plt.figure(figsize=(17,8))





plt.subplot2grid((2,2),(0,0))

sns.violinplot(x= train.Species, y = train.PetalLengthCm)

plt.title("relationship between species and Petal Lenght")





plt.subplot2grid((2,2),(0,1))

sns.violinplot(x= train.Species, y = train.PetalWidthCm)

plt.title("relationship between species and Petal Width")





plt.subplot2grid((2,2),(1,0))

sns.violinplot(x= train.Species, y = train.SepalLengthCm)

plt.title("relationship between species and Sepal Lenght")





plt.subplot2grid((2,2),(1,1))

sns.violinplot(x= train.Species, y = train.SepalWidthCm)

plt.title("relationship between species and Sepal Width")
x = train.drop(["Species", "Species_enc"], axis=1)

y = train["Species_enc"]

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y)









clf_knn = neighbors.KNeighborsClassifier()



clf_knn.fit(x_train, y_train)



score_knn = clf_knn.score(x_test, y_test)







clf_svc = svm.SVC()



clf_svc.fit(x_train, y_train)



score_svc = clf_svc.score(x_test, y_test)







clf_lgr = linear_model.LogisticRegression()



clf_lgr.fit(x_train, y_train)



score_lgr = clf_lgr.score(x_test, y_test)







clf_rdmfrst = ensemble.RandomForestClassifier()



clf_rdmfrst.fit(x_train, y_train)



score_rdmfrst = clf_rdmfrst.score(x_test, y_test)







clf_sgd = linear_model.SGDClassifier()



clf_sgd.fit(x_train, y_train)



score_sgd = clf_sgd.score(x_test, y_test)











scores = [score_knn, score_lgr, score_rdmfrst, score_sgd, score_svc]



clf_names = ["K nearest neighbors", "Logistic regression", "Random forrest", "Stochastic gradient descent", 

             "Support vector classifier"]



clf_scores = pd.DataFrame({"Algorithms": clf_names, "Scores": scores} )



clf_scores.sort_values(by=["Scores"], ascending=False, inplace=True)



clf_scores