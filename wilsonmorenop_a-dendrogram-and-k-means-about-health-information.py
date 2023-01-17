import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

import scipy.stats                # statistics

from scipy.spatial.distance import pdist, squareform

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import linkage

from sklearn import preprocessing

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA



import os
df = pd.read_csv('../input/Health_data_ram.csv')

df = df.drop(['State','special_assitance_area'],axis=1)

df = df.fillna(0)
# Print head of df

df.head()



# Print the info of df

print(df.info())



# Print the shape of df

print(df.shape)#53 filas por 8 variables
#Covariance matrix with five variables

corr = df.iloc[:,1:].corr()
mask = np.zeros_like(corr, dtype=np.bool) 

mask[np.triu_indices_from(mask)] = True 



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9)) 



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True) 



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
df_scale = df.copy() 

scaler = preprocessing.StandardScaler() 

columns = df.columns[1:6] 

df_scale[columns] = scaler.fit_transform(df_scale[columns]) 

df_scale.head()  
row_dist = pd.DataFrame(squareform(pdist(df_scale,metric='euclidean')),columns=df['City ID'],index=df['City ID'])

row_dist.head()
row_clusters = linkage(df_scale.values,metric='euclidean',method='complete')

row_dist.head()
row_dist.shape
fig = plt.figure(figsize=(12,8))

row_render = dendrogram(row_clusters,orientation='left')



plt.tight_layout()

plt.ylabel('Similarity')



plt.show()
#Se trata de aplicar cluster method con k-means



ks = range(1, 6) # pone un intervalo de 1 a 5

inertias = []



for k in ks:

    # Create a KMeans instance with k clusters: model

    model = KMeans(n_clusters=k) #invoca el model de k means evaluando con keyword = n_clusters = k, en la variable model

    

    # Fit model to samples

    model.fit(df_scale.iloc[:,1:]) #Ajusta el model k means al df_scale sin la columna de serial

    

    # Append the inertia to the list of inertias

    inertias.append(model.inertia_) #Suma de las distancias cuadradas de las muestras a sus centros de cluster más cercanos.

    

# Plot ks vs inertias

plt.plot(ks, inertias, '-o')

plt.xlabel('number of clusters, k')

plt.ylabel('inertia')

plt.xticks(ks)

plt.show()
# Create a KMeans instance with 3 clusters: model

model = KMeans(n_clusters=3) 



# Fit model to points

model.fit(df_scale.iloc[:,1:6])



# Determine the cluster labels of new_points: labels

df_scale['cluster'] = model.predict(df_scale.iloc[:,1:6]) 



df_scale.head()
#Componentes Principales PCA en python

# Create PCA instance: model

model_pca = PCA() 



# Apply the fit_transform method of model to grains: pca_features

pca_features = model_pca.fit_transform(df_scale.iloc[:,1:6]) 

# Assign 0th column of pca_features: xs

xs = pca_features[:,0] 

# Assign 1st column of pca_features: ys

ys = pca_features[:,1] # A la variable ys se le asigna las segundas posiciones de pca_features



# Scatter plot xs vs ys

sns.scatterplot(x=xs, y=ys, hue="cluster", data=df_scale)
centroids = model.cluster_centers_ # A model de k-means se le aplica los centros de cada cluster ¿Es necesario?

df_scale.iloc[:,1:10].groupby(['cluster']).mean() # df_scale 
sns.heatmap(df_scale.iloc[:,1:7].groupby(['cluster']).mean(), cmap="YlGnBu")
pd.DataFrame(df_scale['cluster'].value_counts(dropna=False))