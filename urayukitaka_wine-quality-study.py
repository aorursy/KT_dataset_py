# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebraa

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## Libraries

import os

from matplotlib import pyplot as plt

import seaborn as sns

sns.set()



# For clustering

from sklearn import cluster

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import Isomap



# Data preprocessing library

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



# Madhine learning

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import RandomOverSampler



# Evaluation library

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve, auc
df = pd.read_csv("../input/wine-quality/winequalityN.csv", header=0)

df.head()
# Data size

print("Data size:{}".format(df.shape))
# Data info

df.info()
# "type" values

df["type"].value_counts()
# Overview, Basic features

df.describe()
df.dropna(inplace=True)

df.shape
# Wine type and Wine quality data count

fig, ax = plt.subplots(1,2,figsize=(15,6))



# Wine type

sns.countplot(df["type"], ax=ax[0])

ax[0].set_title("Wine type")



# Wine quality

sns.countplot(df["quality"], ax=ax[1])

ax[1].set_title("Wine type")
# Each wine type, plotting check of whole data, sample=1000.

sns.pairplot(df.sample(n=1000, random_state=10), hue="type", hue_order=['white', 'red'])

plt.legend()
# Each wine type, plotting check of whole data, sample=1000.

sns.pairplot(df.sample(n=1000, random_state=10), hue="quality", hue_order=[3,4,5,6,7,8,9])

plt.legend()
corr_values = df.iloc[:,1:]



# Heatmap

plt.figure(figsize=(15,15))

hm = sns.heatmap(corr_values.corr(),

                cbar=True,

                annot=True,

                square=True,

                cmap="RdBu_r",

                fmt=".2f",

                annot_kws={"size":10},

                yticklabels=corr_values.columns,

                vmax=1,

                vmin=-1,

                center=0)

plt.xlabel("Variables")

plt.ylabel("Variables")
# Cluster data set

cluster_params = df.iloc[:,1:-1]



# Standarized 

sc = StandardScaler()

sc.fit(cluster_params)

params = sc.transform(cluster_params)



# Create 5 clusters

kmeans = cluster.KMeans(n_clusters=5, max_iter=30, init="random", random_state=0)

kmeans.fit(params)

labels = kmeans.labels_
# Isomap, Compress to 2D.

iso = Isomap(n_components=2)

iso.fit(params)

data_projected = iso.transform(params)

data_projected.shape
plt.figure(figsize=(13,10))

plt.scatter(data_projected[:,0], data_projected[:,1], c=df["quality"], edgecolor='none', alpha=0.7, cmap=plt.cm.get_cmap('hsv', 6))

plt.colorbar(label="quality", ticks=range(6))
plt.figure(figsize=(13,10))

plt.scatter(data_projected[:,0], data_projected[:,1], c=labels, edgecolor='none', alpha=0.8, cmap=plt.cm.get_cmap('nipy_spectral', 5))

plt.colorbar(label="cluster", ticks=range(5))
df["Cluster"] = labels



# group by Cluster, confirm with mean value.

df.groupby("Cluster").mean().reset_index()
# To make the dataframe

df_1st_route = df.groupby("Cluster").mean().reset_index().query("Cluster==2 | Cluster==1 | Cluster==4").sort_values(by="quality")



# sort index by cluster route

def cluster_route_flg(x):

    if x["Cluster"] == 2:

        res=1

    elif x["Cluster"] == 1:

        res=2

    else:

        res=3

    return res



df_1st_route["Cluster_route"] = df_1st_route.apply(cluster_route_flg, axis=1)



df_1st_route.reset_index(inplace=True)

df_1st_route.head()
# Preparing plot values

x = df_1st_route.iloc[:,1:-1]

y = df_1st_route["quality"]



# Visualization

fig, ax = plt.subplots(1,5, figsize=(25,4))

plt.subplots_adjust(wspace=0.3, hspace=0.3)



# residual sugar

ax[0].plot(x["residual sugar"], y, 'o',markersize=5)

ax[0].set_xlabel("residual sugar")

ax[0].set_ylabel("quality")



# chlorides

ax[1].plot(x["chlorides"], y, 'o',markersize=5)

ax[1].set_xlabel("chlorides")

ax[1].set_ylabel("quality")



# free sulfur dioxide

ax[2].plot(x["free sulfur dioxide"], y, 'o',markersize=5)

ax[2].set_xlabel("free sulfur dioxide")

ax[2].set_ylabel("quality")



# total sulfur dioxide

ax[3].plot(x["total sulfur dioxide"], y, 'o',markersize=5)

ax[3].set_xlabel("total sulfur dioxide")

ax[3].set_ylabel("quality")



# alcohol

ax[4].plot(x["alcohol"], y, 'o',markersize=5)

ax[4].set_xlabel("alcohol")

ax[4].set_ylabel("quality")
# To make the dataframe

df_2nd_route = df.groupby("Cluster").mean().reset_index().query("Cluster==3 | Cluster==0 | Cluster==4")



# sort index by cluster route

def cluster_route_flg(x):

    if x["Cluster"] == 3:

        res=1

    elif x["Cluster"] == 0:

        res=2

    else:

        res=3

    return res



df_2nd_route["Cluster_route"] = df_2nd_route.apply(cluster_route_flg, axis=1)



df_2nd_route.reset_index(inplace=True)

df_2nd_route.head()



df_2nd_route
# Preparing plot values

x = df_2nd_route.iloc[:,1:-1]

y = df_2nd_route["quality"]



# Visualization

fig, ax = plt.subplots(1,5, figsize=(25,4))

plt.subplots_adjust(wspace=0.3, hspace=0.3)



# volatile acidity

ax[0].scatter(x["volatile acidity"], y, s=40)

ax[0].set_xlabel("volatile acidity")

ax[0].set_ylabel("quality")



# residual sugar

ax[1].scatter(x["residual sugar"], y, s=40)

ax[1].set_xlabel("residual sugar")

ax[1].set_ylabel("quality")



# free sulfur dioxide

ax[2].scatter(x["free sulfur dioxide"], y, s=40)

ax[2].set_xlabel("free sulfur dioxide")

ax[2].set_ylabel("quality")



# total sulfur dioxide

ax[3].scatter(x["total sulfur dioxide"], y, s=40)

ax[3].set_xlabel("total sulfur dioxide")

ax[3].set_ylabel("quality")



# alcohol

ax[4].scatter(x["alcohol"], y, s=40)

ax[4].set_xlabel("alcohol")

ax[4].set_ylabel("quality")
# Making the flag of wine type

mapping = {"white":0, "red":1}

df["type_flg"] = df['type'].map(mapping)

df['type_flg'].value_counts()
# Data preparing

X = df.iloc[:,1:12]

y = df["type_flg"]



# Data splitting to make the training data and validation data

# training data :70%, validation(test data) :30%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



# Taking veryfing to Standarlized data

sc = StandardScaler()

sc.fit(X_train)



X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
# Logistic Regression

lr = LogisticRegression()



param_range = [0.001, 0.01, 0.1, 1.0]

penalty = ['l1', 'l2']

param_grid = [{"C":param_range, "penalty":penalty}]



gs_lr = GridSearchCV(estimator=lr, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=-1)

gs_lr = gs_lr.fit(X_train_std, y_train)



print(gs_lr.best_score_.round(3))

print(gs_lr.best_params_)
# Decision tree

tree = DecisionTreeClassifier(max_depth=4, random_state=10)



param_range = [3, 6, 9, 12]

leaf = [10, 15, 20]

criterion = ["entropy", "gini", "error"]

param_grid = [{"max_depth":param_range, "criterion":criterion, "max_leaf_nodes":leaf}]



gs_tree = GridSearchCV(estimator=tree, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=-1)

gs_tree = gs_tree.fit(X_train, y_train)



print(gs_tree.best_score_.round(3))

print(gs_tree.best_params_)
print("-"*50)

# Logistic Regression Result

y_pred = gs_lr.best_estimator_.predict(X_test_std)

print("Logistic Regression Result")

print("confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))

print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))

print("-"*50)



# Decision tree

y_pred = gs_tree.best_estimator_.predict(X_test)

print("Decision tree")

print("confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))

print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))

print("-"*50)
def quality_flag(x):

    if x["quality"] >= 7:

        res = 1

    else:

        res = 0

    return res



df["quality_flg"] = df.apply(quality_flag, axis=1)

df["quality_flg"].value_counts()
# Data preparing

X = df.iloc[:,1:12]

y = df["quality_flg"]



# Data splitting to make the training data and validation data

# training data :70%, validation(test data) :30%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



# Taking veryfing to Standarlized data

sc = StandardScaler()

sc.fit(X_train)



X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
# Logistic Regression

lr = LogisticRegression()



param_range = [0.001, 0.01, 0.1, 1.0]

penalty = ['l1', 'l2']

param_grid = [{"C":param_range, "penalty":penalty}]



gs_lr = GridSearchCV(estimator=lr, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=-1)

gs_lr = gs_lr.fit(X_train_std, y_train)



print(gs_lr.best_score_.round(3))

print(gs_lr.best_params_)
# Decision tree

tree = DecisionTreeClassifier(max_depth=4, random_state=10)



param_range = [3, 6, 9, 12]

leaf = [10, 15, 20]

criterion = ["entropy", "gini", "error"]

param_grid = [{"max_depth":param_range, "criterion":criterion, "max_leaf_nodes":leaf}]



gs_tree = GridSearchCV(estimator=tree, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=-1)

gs_tree = gs_tree.fit(X_train, y_train)



print(gs_tree.best_score_.round(3))

print(gs_tree.best_params_)
print("-"*50)

# Logistic Regression Result

y_pred = gs_lr.best_estimator_.predict(X_test_std)

print("Logistic Regression Result")

print("confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))

print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))

print("-"*50)



# Decision tree

y_pred = gs_tree.best_estimator_.predict(X_test)

print("Decision tree")

print("confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))

print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))

print("-"*50)
# Set a RandomOverSampler

ros = RandomOverSampler(sampling_strategy = 'auto', random_state=10)
# Making the training data

X_train_resampled, y_train_resampled = ros.fit_sample(X_train_std, y_train)



# Logistic Regression

lr = LogisticRegression()



param_range = [0.001, 0.01, 0.1, 1.0]

penalty = ['l1', 'l2']

param_grid = [{"C":param_range, "penalty":penalty}]



gs_lr = GridSearchCV(estimator=lr, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=-1)

gs_lr = gs_lr.fit(X_train_resampled, y_train_resampled)



print(gs_lr.best_score_.round(3))

print(gs_lr.best_params_)
# Making the training data

X_train_resampled, y_train_resampled = ros.fit_sample(X_train, y_train)



# Decision tree

tree = DecisionTreeClassifier(max_depth=4, random_state=10)



param_range = [3, 6, 9, 12]

leaf = [10, 15, 20]

criterion = ["entropy", "gini", "error"]

param_grid = [{"max_depth":param_range, "criterion":criterion, "max_leaf_nodes":leaf}]



gs_tree = GridSearchCV(estimator=tree, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=-1)

gs_tree = gs_tree.fit(X_train_resampled, y_train_resampled)



print(gs_tree.best_score_.round(3))

print(gs_tree.best_params_)
print("-"*50)

# Logistic Regression Result

y_pred = gs_lr.best_estimator_.predict(X_test_std)

print("Logistic Regression Result")

print("confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))

print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))

print("-"*50)



# Decision tree

y_pred = gs_tree.best_estimator_.predict(X_test)

print("Decision tree")

print("confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))

print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))

print("-"*50)
# Data preparing

X = df.query("type=='red'").iloc[:,1:12]

y = df.query("type=='red'")["quality_flg"]



# Data splitting to make the training data and validation data

# training data :70%, validation(test data) :30%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



# Taking veryfing to Standarlized data

sc = StandardScaler()

sc.fit(X_train)



X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
# Logistic Regression

lr = LogisticRegression()



param_range = [0.001, 0.01, 0.1, 1.0]

penalty = ['l1', 'l2']

param_grid = [{"C":param_range, "penalty":penalty}]



gs_lr = GridSearchCV(estimator=lr, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=-1)

gs_lr = gs_lr.fit(X_train_std, y_train)



print(gs_lr.best_score_.round(3))

print(gs_lr.best_params_)
# Decision tree

tree = DecisionTreeClassifier(max_depth=4, random_state=10)



param_range = [3, 6, 9, 12]

leaf = [10, 15, 20]

criterion = ["entropy", "gini", "error"]

param_grid = [{"max_depth":param_range, "criterion":criterion, "max_leaf_nodes":leaf}]



gs_tree = GridSearchCV(estimator=tree, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=-1)

gs_tree = gs_tree.fit(X_train, y_train)



print(gs_tree.best_score_.round(3))

print(gs_tree.best_params_)
print("-"*50)

# Logistic Regression Result

y_pred = gs_lr.best_estimator_.predict(X_test_std)

print("Logistic Regression Result")

print("confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))

print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))

print("-"*50)



# Decision tree

y_pred = gs_tree.best_estimator_.predict(X_test)

print("Decision tree")

print("confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))

print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))

print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))

print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))

print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))

print("-"*50)