# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

from pandas import DataFrame

from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv("../input/iris/Iris.csv")

iris.info()

iris.head()
sns.pairplot(iris,hue="Species")

plt.show()
sns.heatmap(iris.corr(),annot=True,cmap='cubehelix_r')

plt.show()
kume1 = DataFrame(iris,columns=['PetalLengthCm','PetalWidthCm'])

kume1.head()
kmeans = KMeans(n_clusters=3).fit(kume1)

centroids = kmeans.cluster_centers_

print(centroids)

sns.scatterplot(x="PetalLengthCm",y="PetalWidthCm",data=iris,hue="Species")

plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100)

plt.show()