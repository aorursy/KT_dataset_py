import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.cluster import AgglomerativeClustering 

from sklearn.preprocessing import StandardScaler, normalize

from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score

import scipy.cluster.hierarchy as shc
raw_df = pd.read_csv('../input/ccdata/CC GENERAL.csv')

raw_df = raw_df.drop('CUST_ID', axis = 1) 

raw_df.fillna(method ='ffill', inplace = True) 

raw_df.head(2)
# Standardize data

scaler = StandardScaler() 

scaled_df = scaler.fit_transform(raw_df) 

  

# Normalizing the Data 

normalized_df = normalize(scaled_df) 

  

# Converting the numpy array into a pandas DataFrame 

normalized_df = pd.DataFrame(normalized_df) 

  

# Reducing the dimensions of the data 

pca = PCA(n_components = 2) 

X_principal = pca.fit_transform(normalized_df) 

X_principal = pd.DataFrame(X_principal) 

X_principal.columns = ['P1', 'P2'] 

  

X_principal.head(2)
plt.figure(figsize =(6, 6)) 

plt.title('Visualising the data') 

Dendrogram = shc.dendrogram((shc.linkage(X_principal, method ='ward'))) 
silhouette_scores = [] 



for n_cluster in range(2, 8):

    silhouette_scores.append( 

        silhouette_score(X_principal, AgglomerativeClustering(n_clusters = n_cluster).fit_predict(X_principal))) 

    

# Plotting a bar graph to compare the results 

k = [2, 3, 4, 5, 6,7] 

plt.bar(k, silhouette_scores) 

plt.xlabel('Number of clusters', fontsize = 10) 

plt.ylabel('Silhouette Score', fontsize = 10) 

plt.show() 
agg = AgglomerativeClustering(n_clusters=3)

agg.fit(X_principal)
# Visualizing the clustering 

plt.scatter(X_principal['P1'], X_principal['P2'],  

           c = AgglomerativeClustering(n_clusters = 3).fit_predict(X_principal), cmap =plt.cm.winter) 

plt.show() 