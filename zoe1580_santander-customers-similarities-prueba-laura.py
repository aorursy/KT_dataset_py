# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in. 



import numpy as np # linear algebra.

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
N_ROWS = 1000

filename = "../input/train_ver2.csv"



#n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)

n = 13647309 #number of records in train_ver2.csv

print(n)

skip = sorted(random.sample(range(1,n+1),n-N_ROWS)) #the 0-indexed header will not be included in the skip list
df = pd.read_csv(filename, skiprows=skip)

print(df.shape)
df['fecha_dato'] = pd.to_datetime(df['fecha_dato'])

df['fecha_alta'] = pd.to_datetime(df['fecha_alta'])
import re 



pattern = re.compile("ind_.*_ult1")



prod_cols = [ x for x in df.columns if re.match(pattern,x) ]

prod_cols.append("ncodpers")

print(prod_cols)
df_dummiesed = pd.get_dummies(df)

df_dummiesed = df_dummiesed.drop_duplicates(subset="ncodpers",keep="last")

df_dummiesed["fecha_alta"] = pd.to_numeric(df_dummiesed["fecha_alta"])

df_dummiesed["fecha_dato"] = pd.to_numeric(df_dummiesed["fecha_dato"])

#df_dummiesed = df_dummiesed.drop("conyuemp",axis=1).fillna(df_dummiesed.mean())

df_dummiesed = df_dummiesed.drop("conyuemp",axis=1).dropna()



print(df_dummiesed.describe())
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

import seaborn as sns

import matplotlib.pyplot as plt



s = []

for n_clusters in range(2,20):

    kmeans = KMeans(n_clusters=n_clusters)

    kmeans.fit(df_dummiesed)



    labels = kmeans.labels_

    centroids = kmeans.cluster_centers_



    s.append(silhouette_score(df_dummiesed, labels, metric='euclidean'))



plt.plot(s)

plt.ylabel("Silouette")

plt.xlabel("k")

plt.title("Silouette for K-means cell's behaviour")

sns.despine()
N_CLUSTERS = 11

kmeans = KMeans(n_clusters=N_CLUSTERS)

kmeans.fit(df_dummiesed)



print(kmeans.labels_)

plt.hist(kmeans.labels_,bins=N_CLUSTERS-1)
from sklearn.metrics.pairwise import cosine_similarity



sim = cosine_similarity(df_dummiesed,dense_output=False)

df_sim = pd.DataFrame(sim,columns=df_dummiesed['ncodpers'],index=df_dummiesed['ncodpers'])



print(df_sim)