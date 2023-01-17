import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from functools import reduce

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score
df1=pd.read_csv("../input/europe-datasets/crime_2016.csv")

df2=pd.read_csv("../input/europe-datasets/pollution_2016.csv")

df3=pd.read_csv("../input/europe-datasets/trust_in_legal_2013.csv")

df4=pd.read_csv("../input/europe-datasets/trust_in_police_2013.csv")

df5=pd.read_csv("../input/europe-datasets/trust_in_politics_2013.csv")

df6=pd.read_csv("../input/europe-datasets/gdp_2016.csv")

df7=pd.read_csv("../input/europe-datasets/perceived_health_2016.csv")

df8=pd.read_csv("../input/europe-datasets/population_2011.csv")
# consolidate data with the country as the unique identifier

data=reduce(lambda x,y: pd.merge(x,y, on='country', how='inner'), 

            [df1, df2, df3, df4, df5, df6, df7, df8])



# matrix without the unique identifier for fitting later

x=data

x=x.drop('country',1)
# Elbow Method

sum_of_sq_dist=[] # initialize

K=range(1,10) # check up to 10 clusters

for k in K:

    km=KMeans(n_clusters=k)

    km=km.fit(x)

    sum_of_sq_dist.append(km.inertia_)



# plot the result

plt.plot(K,sum_of_sq_dist,'bx-')

plt.xlabel('k clusters')

plt.ylabel('Sum of Squared Distances')

plt.title('Elbow Method for Best k')

plt.show()
# Print results from Avg Silhouette Method

for n_clusters in range(2,11): # check up to 10 clusters

    clusterer=KMeans(n_clusters=n_clusters,random_state=10) # seed 10

    cluster_labels=clusterer.fit_predict(x)

    silhouette_avg=silhouette_score(x,cluster_labels)

    print("When we use",n_clusters,"clusters,",

          "the average silhouette score is",

          round(silhouette_avg,3)) # round to 3 decimals for readable comparison

    sample_silhouette_values=silhouette_samples(x,cluster_labels)
# Fit with k-means clustering

kmeans=KMeans(n_clusters=2)

kmeans=kmeans.fit(x)



col=list(x.columns.values) # original variables

count=data.iloc[:,0] # original countries



ctr=kmeans.cluster_centers_ # cluster coordinates

count_label=kmeans.labels_+1 # assign cluster numbers as cluster 1 and 2 for intuition



# Show results

ctr=pd.DataFrame(ctr,columns=col)

ctr.insert(0,"cluster",[1,2]) # label countries

(ctr.T).round(2)
# show the clustered countries

ctry_sum={'country': count,

          'cluster': count_label} 

ctry_sum=pd.DataFrame(ctry_sum,columns=['country','cluster'])

ctry_sum
result = pd.concat([ctry_sum.reset_index(drop=True), x], axis=1) # original data and cluster results

cluster_1=result[result.cluster==1]

cluster_2=result[result.cluster==2]

plt.scatter(cluster_1.iloc[:,2],cluster_1.iloc[:,14],c="green",label="Cluster 1")

plt.scatter(cluster_2.iloc[:,2],cluster_2.iloc[:,14],c="red",label="Cluster 2")

plt.xlabel('% Reported Crime')

plt.ylabel('% Young Adult Pop')

plt.title('% Reported Crime vs. % Young Adult Pop')

plt.legend()

plt.show()
plt.scatter(cluster_1.iloc[:,12],cluster_1.iloc[:,13],c="green",label="Cluster 1")

plt.scatter(cluster_2.iloc[:,12],cluster_2.iloc[:,13],c="red",label="Cluster 2")

plt.xlabel('% Pop in Very Bad Health')

plt.ylabel('Pop (10s of millions)')

plt.title('% Pop in Very Bad Health vs. Pop (10 millions)')

plt.legend()

plt.show()
plt.scatter(cluster_1.iloc[:,3],cluster_1.iloc[:,7],c="green",label="Cluster 1")

plt.scatter(cluster_2.iloc[:,3],cluster_2.iloc[:,7],c="red",label="Cluster 2")

plt.xlabel('% Pollution')

plt.ylabel('GDP')

plt.title('% Pollution vs. GDP')

plt.legend()

plt.show()