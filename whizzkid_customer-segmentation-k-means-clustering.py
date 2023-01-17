import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
cust=pd.read_csv("../input/Mall_Customers.csv")

cust.columns=["id","gender","age","income","score"]

cust.index=cust.id.values

cust=cust.drop(["id","gender"],axis=1)

cust.head()
cust.describe()
fig=plt.figure()

ax=fig.add_subplot(111,projection="3d")

ax.scatter(cust.age,cust.income,cust.score)
from sklearn.cluster import KMeans

kmean=KMeans(n_clusters=3).fit_predict(cust.values)
fig=plt.figure()

ax=fig.add_subplot(111,projection="3d")

ax.scatter(cust.age,cust.income,cust.score,c=kmean)