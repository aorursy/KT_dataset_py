# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv("../input/Iris.csv")
import seaborn as sns



_ = sns.pairplot(

    data = iris,

    vars=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],

    hue="Species"

)
data = iris[iris.Species != "Iris-setosa"][["SepalLengthCm", "PetalLengthCm"]]



_ = data.plot.scatter(x="SepalLengthCm", y="PetalLengthCm", figsize=(6, 6), xlim=[3.5, 9.5], ylim=[2, 8])
import matplotlib.pyplot as plt



_, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

_ = sns.heatmap(

    data.corr(), 

    cmap=sns.diverging_palette(220, 10, as_cmap=True), 

    square=True, 

    cbar=False, 

    ax=ax,

    annot = True, 

    annot_kws = {"fontsize": 8}

)
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

sns.regplot(data=data, x="SepalLengthCm", y="PetalLengthCm", ax=ax)

ax.set_xlim([3.5, 9.5])

_ = ax.set_ylim([2, 8])
data_centered = pd.DataFrame(data.values - data.mean(axis=0).values, columns=["SepalLengthCm", "PetalLengthCm"])



_ = data_centered.plot.scatter(x="SepalLengthCm", y="PetalLengthCm", figsize=(6, 6), xlim=[-3, 3], ylim=[-3, 3])
X = np.mat(data_centered.values)

n = X.shape[0]

m = X.shape[1]

Xt = X.T



S = Xt * X / (n + 1) # matrix product between Xt and X



_, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

_ = sns.heatmap(

    S, 

    cmap=sns.diverging_palette(220, 10, as_cmap=True), 

    square=True, 

    cbar=False, 

    ax=ax,

    annot = True, 

    annot_kws = {"fontsize": 8}

)
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

_ = sns.heatmap(

    data.cov(), 

    cmap=sns.diverging_palette(220, 10, as_cmap=True), 

    square=True, 

    cbar=False, 

    ax=ax,

    annot = True, 

    annot_kws = {"fontsize": 8}

)
w, Vt = np.linalg.eigh(S)

A = np.diag(w)



_, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

_ = sns.heatmap(

    Vt * A * Vt.T, 

    cmap=sns.diverging_palette(220, 10, as_cmap=True), 

    square=True, 

    cbar=False, 

    ax=ax,

    annot = True, 

    annot_kws = {"fontsize": 8}

)
Y = X * Vt



_, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

_ = sns.heatmap(

    pd.DataFrame(Y, columns=[r"$Y_1$", r"$Y_2$"]).cov(), 

    cmap=sns.diverging_palette(220, 10, as_cmap=True), 

    square=True, 

    cbar=False, 

    ax=ax,

    annot = True, 

    annot_kws = {"fontsize": 8}

)
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

_ = sns.heatmap(

    A, 

    cmap=sns.diverging_palette(220, 10, as_cmap=True), 

    square=True, 

    cbar=False, 

    ax=ax,

    annot = True, 

    annot_kws = {"fontsize": 8}

)
ax = data_centered.plot.scatter(x="SepalLengthCm", y="PetalLengthCm", figsize=(6, 6), xlim=[-3,3], ylim=[-3, 3], alpha=0.4)

ax.annotate(r"$V^t_1$", xy=[0, 0], xytext=Vt[:,1], arrowprops={"arrowstyle": "<|-"})

_ = ax.annotate(r"$V^t_0$", xy=[0, 0], xytext=Vt[:,0], arrowprops={"arrowstyle": "<|-"})
print("The total variance of original data is {:.3f}. The total variance of new data is {:.3f}".format(

    data.cov().values.trace(),

    pd.DataFrame(Y, columns=["Y1", "Y2"]).cov().values.trace()

    )

)
from sklearn.decomposition import PCA



pca = PCA()

pca.fit(data)



explained_variance = pd.DataFrame({"evr": pca.explained_variance_ratio_, "evrc": pca.explained_variance_ratio_.cumsum()}, 

                                  index=pd.Index(["pc{:d}".format(i) for i in np.arange(1, 3)], name="principle components"))



_, ax = plt.subplots(figsize=(8, 4))

_ = explained_variance.evrc.plot(kind="line", color="#ee7621", ax=ax, linestyle="-", marker="h")

_ = explained_variance.evr.plot(kind="bar", ax=ax, color="#00304e", alpha=0.8)

_ = ax.set_title("Explained Variance Ratio of Principle Components", fontsize=10)

_ = ax.set_ylim([0.0, 1.1])



for x, y in zip(np.arange(0, len(explained_variance.evrc)), explained_variance.evrc):

    _ = ax.annotate("{:.1f}%".format(y * 100.0), xy=(x-0.05, y+0.03), fontsize=7)



for x, y in zip(np.arange(1, len(explained_variance.evr)), explained_variance.evr[1:]):

    _ = ax.annotate("{:.1f}%".format(y * 100.0), xy=(x-0.05, y+0.02), fontsize=7)
from sklearn.decomposition import PCA



pca = PCA()

pca.fit(iris.loc[iris.Species != "Iris-setosa", ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]])





explained_variance = pd.DataFrame({"evr": pca.explained_variance_ratio_, "evrc": pca.explained_variance_ratio_.cumsum()}, 

                                  index=pd.Index(["pc{:d}".format(i) for i in np.arange(1, 5)], name="principle components"))





_, ax = plt.subplots(figsize=(8, 4))

_ = explained_variance.evrc.plot(kind="line", color="#ee7621", ax=ax, linestyle="-", marker="h")

_ = explained_variance.evr.plot(kind="bar", ax=ax, color="#00304e", alpha=0.8)

_ = ax.set_title("Explained Variance Ratio of Principle Components", fontsize=10)

_ = ax.set_ylim([0.0, 1.1])



for x, y in zip(np.arange(0, len(explained_variance.evrc)), explained_variance.evrc):

    _ = ax.annotate("{:.1f}%".format(y * 100.0), xy=(x-0.07, y+0.03), fontsize=7)



for x, y in zip(np.arange(1, len(explained_variance.evr)), explained_variance.evr[1:]):

    _ = ax.annotate("{:.1f}%".format(y * 100.0), xy=(x-0.05, y+0.02), fontsize=7)
