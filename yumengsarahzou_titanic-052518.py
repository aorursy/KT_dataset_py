# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# import matplotlib as mpl
# mpl.rcParams['figure.figsize'] = (12,8)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
combine = pd.concat([train.drop(["Survived"], axis=1), test], axis=0)
train.head()
train.info()
train = train.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
train.head()
train["Sex"] = np.where(train["Sex"]=="female", 1, 0)
train.head()
corr = train.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap="coolwarm")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
sns.pointplot(x="Pclass", y="Fare", hue="Embarked", data=train, ax=axes[0])
sns.pointplot(x="Sex", y="Age", hue="Embarked", data=train, ax=axes[1])
sns.pointplot(x="Embarked", y="Pclass", hue="Sex", data=train, ax=axes[2])
print("{}{}".format("survival rate: \n", train.groupby("Embarked")["Survived"].mean()))
print("{}{}".format("count: \n", train.groupby(["Pclass", "Sex"])["Embarked"].value_counts().unstack()))
train.loc[train["Embarked"].isnull()==True, :]
combine.query("Sex=='1' & Pclass==1 & 70 < Fare < 90")["Embarked"].mode()
train.loc[train["Embarked"].isnull()==True, "Embarked"] = "C"
train.loc[train["Fare"]==80, :]
train.loc[train["Age"].isnull()==True, :][:5]
g = sns.FacetGrid(train, col="Sex", row="Pclass", hue="Embarked")
g.map(sns.kdeplot, "Age").add_legend()
combine["Sex"] = np.where(combine["Sex"]=="female", 1, 0)
def fill_age(df):
    for pclass in df["Pclass"].unique():
        for sex in df["Sex"].unique():
            for embark in df["Embarked"].unique():
                query_str = "Pclass=={} & Sex=={} & Embarked=='{}'".format(pclass, sex, embark)
                median_age = combine.query(query_str)["Age"].median()
                print("Pclass={}, Sex={}, Embarked={}: median age {}".format(pclass, sex, embark, median_age))
                df.loc[df.query(query_str + "& Age=='NaN'").index.values, "Age"] = median_age
fill_age(train)
train.loc[train["Age"].isnull()==True, :]
train["Family"] = train["SibSp"] + train["Parch"]
sns.barplot(x="Family", y="Survived", data=train)
train = pd.get_dummies(train, drop_first=True)
train.head()
y_train = train["Survived"].values
X_train = train.drop(["Survived"], axis=1).values
train_scaled = train.copy()
def scale(series):
    min_ = series.min()
    range_ = (series - min_).max()
    return (series - min_)/range_
train_scaled["Age"] = scale(train_scaled["Age"])
train_scaled["Fare"] = scale(train_scaled["Fare"])
train_scaled.head()
sns.jointplot("Age", "Fare", train_scaled)
corr = train.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap="coolwarm")
X_train_scaled = train_scaled.drop(["Survived"], axis=1).values
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
def print_feature_weight(feature_importances_):
    f_weight = ["{:.0%}".format(f) for f in feature_importances_]
    for line in (train.columns[1:], f_weight):
        print("{:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>12} {:>12}".format(*line))
Cs = np.logspace(-3, 3, 7)
logreg_cv = LogisticRegressionCV(Cs, cv=5, max_iter=1000).fit(X_train, y_train)
print("C: ", logreg_cv.C_)
print("{:.1%}".format(logreg_cv.score(X_train, y_train)))
nb_cv_score = cross_val_score(GaussianNB(), X_train, y_train, cv=5)
print(nb_cv_score)
print("{:.1%}".format(nb_cv_score.mean()))
params = {"max_depth":np.linspace(1, 20, 20)}
tree_cv = GridSearchCV(DecisionTreeClassifier(), params, cv=5)
tree_cv.fit(X_train, y_train)
print(tree_cv.best_params_)
print("{:.1%}".format(tree_cv.best_score_))
params = {"n_estimators": np.linspace(10, 30, 5, dtype="int_"),
          "max_depth": np.linspace(1, 20, 8, dtype="int_")}
rf_cv = GridSearchCV(RandomForestClassifier(), params, cv=5)
rf_cv.fit(X_train, y_train)
print(rf_cv.best_params_)
print("{:.1%}".format(rf_cv.best_score_))
params = {"n_estimators": np.logspace(1, 2.5, 5, dtype="int_"),
          "max_depth": np.linspace(1, 6, 3, dtype="int_"),
          "learning_rate": np.logspace(-2, 0, 5)}
gb_cv = GridSearchCV(GradientBoostingClassifier(), params, cv=5)
gb_cv.fit(X_train, y_train)
print(gb_cv.best_params_)
print("{:.1%}".format(gb_cv.best_score_))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
params = {"n_neighbors": np.linspace(4, 26, 11, dtype="int_")}
knn_cv = GridSearchCV(KNeighborsClassifier(), params, cv=5)
knn_cv.fit(X_train_scaled, y_train)
print(knn_cv.best_params_)
print("{:.1%}".format(knn_cv.best_score_))
params = {"C":np.logspace(-2, 4, 7),
          "gamma": ("auto", 0.1, 0.01)}
svc_cv = GridSearchCV(SVC(), params, cv=5)
svc_cv.fit(X_train_scaled, y_train)
print(svc_cv.best_params_)
print("{:.1%}".format(svc_cv.best_score_))
mlp = MLPClassifier(max_iter=1000)
mlp_cv_score = cross_val_score(mlp, X_train_scaled, y_train, cv=5)
print(mlp_cv_score)
print("{:.1%}".format(mlp_cv_score.mean()))
clf = gb_cv.best_estimator_
print_feature_weight(clf.feature_importances_)
test = test.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
test["Sex"] = np.where(test["Sex"]=="female", 1, 0)
test.info()
fill_age(test)
test.loc[test["Age"].isnull()==True, :]
g = sns.FacetGrid(combine, col="Pclass", hue="Embarked", size=5)
g.map(sns.kdeplot, "Fare").add_legend()
test.loc[test["Fare"].isnull()==True, :]
test.query("Pclass==3 & Embarked=='S'")["Fare"].median()
test.loc[test["Fare"].isnull()==True, "Fare"] = test.query("Pclass==3 & Embarked=='S'")["Fare"].median()
test["Family"] = test["SibSp"] + test["Parch"]
test = pd.get_dummies(test, drop_first=True)
test.head()
X_test = test.values
y_pred = clf.predict(X_test)
submission = pd.DataFrame({"PassengerId": np.arange(len(y_pred))+892, "Survived": y_pred})
submission.to_csv("titanic_submission.csv", index=False)
submission.head()
