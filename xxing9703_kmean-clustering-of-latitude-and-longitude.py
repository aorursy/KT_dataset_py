import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from collections import Counter
#load properties

#replace 'NaN' in regionidzip with 97000

#drop rows with 'NaN' in latitude & longitude



df = pd.read_csv('../input/properties_2016.csv')

df['regionidzip']=df['regionidzip'].fillna(97000)

df.dropna(axis=0,how='any',subset=['latitude','longitude'],inplace=True)

X=df.loc[:,['latitude','longitude']]

zp=df.regionidzip
#run KMeans

id_n=8

kmeans = KMeans(n_clusters=id_n, random_state=0).fit(X)

id_label=kmeans.labels_
#plot result

ptsymb = np.array(['b.','r.','m.','g.','c.','k.','b*','r*','m*','r^']);

plt.figure(figsize=(12,12))

plt.ylabel('Longitude', fontsize=12)

plt.xlabel('Latitude', fontsize=12)

for i in range(id_n):

    cluster=np.where(id_label==i)[0]

    plt.plot(X.latitude[cluster].values,X.longitude[cluster].values,ptsymb[i])

plt.show()
#revise the clustering based on zipcode

uniq_zp=np.unique(zp)

for i in uniq_zp:

    a=np.where(zp==i)[0]

    c = Counter(id_label[a])

    c.most_common(1)[0][0]

    id_label[a]=c.most_common(1)[0][0]
#plot result (revised)

plt.figure(figsize=(12,12))

plt.ylabel('Longitude', fontsize=12)

plt.xlabel('Latitude', fontsize=12)

for i in range(id_n):

    cluster=np.where(id_label==i)[0]

    plt.plot(X.latitude[cluster].values,X.longitude[cluster].values,ptsymb[i])

plt.show()