import pandas as pd 

import matplotlib.pyplot as plt 

from sklearn.cluster import SpectralClustering 

from sklearn.preprocessing import StandardScaler, normalize 

from sklearn.decomposition import PCA 

from sklearn.metrics import silhouette_score 
raw_df = pd.read_csv('../input/ccdata/CC GENERAL.csv')

raw_df = raw_df.drop('CUST_ID', axis = 1) 

raw_df.fillna(method ='ffill', inplace = True) 

raw_df.head(2)
# Preprocessing the data to make it visualizable 

  

# Scaling the Data 

scaler = StandardScaler() 

X_scaled = scaler.fit_transform(raw_df) 

  

# Normalizing the Data 

X_normalized = normalize(X_scaled) 

  

# Converting the numpy array into a pandas DataFrame 

X_normalized = pd.DataFrame(X_normalized) 

  

# Reducing the dimensions of the data 

pca = PCA(n_components = 2) 

X_principal = pca.fit_transform(X_normalized) 

X_principal = pd.DataFrame(X_principal) 

X_principal.columns = ['P1', 'P2'] 

  

X_principal.head(2) 
# Building the clustering model 

spectral_model_rbf = SpectralClustering(n_clusters = 2, affinity ='rbf') 

  

# Training the model and Storing the predicted cluster labels 

labels_rbf = spectral_model_rbf.fit_predict(X_principal)
# Visualizing the clustering 

plt.scatter(X_principal['P1'], X_principal['P2'],  

           c = SpectralClustering(n_clusters = 2, affinity ='rbf') .fit_predict(X_principal), cmap =plt.cm.winter) 

plt.show() 
# Building the clustering model 

spectral_model_nn = SpectralClustering(n_clusters = 2, affinity ='nearest_neighbors') 

  

# Training the model and Storing the predicted cluster labels 

labels_nn = spectral_model_nn.fit_predict(X_principal)
# Visualizing the clustering 

plt.scatter(X_principal['P1'], X_principal['P2'],  

           c = SpectralClustering(n_clusters = 2, affinity ='nearest_neighbors') .fit_predict(X_principal), cmap =plt.cm.winter) 

plt.show() 
# List of different values of affinity 

affinity = ['rbf', 'nearest-neighbours'] 

  

# List of Silhouette Scores 

s_scores = [] 

  

# Evaluating the performance 

s_scores.append(silhouette_score(raw_df, labels_rbf)) 

s_scores.append(silhouette_score(raw_df, labels_nn)) 

  

# Plotting a Bar Graph to compare the models 

plt.bar(affinity, s_scores) 

plt.xlabel('Affinity') 

plt.ylabel('Silhouette Score') 

plt.title('Comparison of different Clustering Models') 

plt.show() 



print(s_scores)