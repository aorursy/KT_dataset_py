# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import datasets

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
iris = datasets.load_iris()
In:print (iris.DESCR)

print (iris.data)

print (iris.data.shape)

print (iris.feature_names)

print (iris.target)

print (iris.target.shape)

print (iris.target_names)
print (type(iris.data))
import pandas as pd

import numpy as np

from pandas.plotting import scatter_matrix

colors = list()

palette = {0: "red", 1: "green", 2: "blue"}

for c in np.nditer(iris.target): colors.append(palette[int(c)])

# using the palette dictionary, we convert

# each numeric class into a color string

dataframe = pd.DataFrame(iris.data,  columns=iris.feature_names)

sc = pd.plotting.scatter_matrix(dataframe, alpha=0.3, figsize=(10, 10), diagonal='hist', color=colors, marker='o', grid=True)
pd.__version__
print(__doc__)





# Code source: Gaël Varoquaux

# Modified for documentation by Jaques Grobler

# License: BSD 3 clause



import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets

from sklearn.decomposition import PCA



# import some data to play with

iris = datasets.load_iris()

X = iris.data[:, :2]  # we only take the first two features.

y = iris.target



x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5

y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5



plt.figure(2, figsize=(8, 6))

plt.clf()



# Plot the training points

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,

            edgecolor='k')

plt.xlabel('Sepal length')

plt.ylabel('Sepal width')



plt.xlim(x_min, x_max)

plt.ylim(y_min, y_max)

plt.xticks(())

plt.yticks(())



# To getter a better understanding of interaction of the dimensions

# plot the first three PCA dimensions

fig = plt.figure(1, figsize=(8, 6))

ax = Axes3D(fig, elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(iris.data)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,

           cmap=plt.cm.Set1, edgecolor='k', s=40)

ax.set_title("First three PCA directions")

ax.set_xlabel("1st eigenvector")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("2nd eigenvector")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("3rd eigenvector")

ax.w_zaxis.set_ticklabels([])



plt.show()