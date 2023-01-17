import numpy  as np

import pandas as pd

import matplotlib.pyplot as pl

import seaborn as sb



%matplotlib inline

from sklearn.cluster import KMeans









dataframe=pd.read_csv('../input/Iris.csv',index_col=0)
sb.pairplot(dataframe)
x = np.array(dataframe.drop('Species',axis = 1))

x=np.round(dataframe.drop('Species',axis = 1),2)

kmeans = KMeans(n_clusters=3, random_state=0)

kmeans.fit(x)

kmeans.labels_

dataframe['classes'] = kmeans.labels_
sb.pairplot(dataframe,vars=dataframe.columns[:4],hue="classes")