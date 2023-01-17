import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline



# Import the 3 dimensionality reduction methods

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA
train = pd.read_csv('../input/train.csv')

print(train.shape)
# save the labels to a Pandas series target

target = train['label']

# Drop the label feature

train = train.drop("label",axis=1)
# Standardize the data

from sklearn.preprocessing import StandardScaler

X = train.values

X_std = StandardScaler().fit_transform(X)



# Calculating Eigenvectors and eigenvalues of Cov matirx

mean_vec = np.mean(X_std, axis=0)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# Create a list of (eigenvalue, eigenvector) tuples

eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]



# Sort the eigenvalue, eigenvector pair from high to low

eig_pairs.sort(key = lambda x: x[0], reverse= True)



# Calculation of Explained Variance from the eigenvalues

tot = sum(eig_vals)

var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance

cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance
# Find the eigenvector beyond which 90% of the data is explained

[ n for n,i in enumerate(cum_var_exp) if i>90 ][0]

# Call the PCA method with 228 components. 

pca = PCA(n_components=228)

pca.fit(X_std)

X_228d = pca.transform(X_std)

print(X_228d.shape)
from sklearn.ensemble import RandomForestClassifier as RF

# Use 25 decision trees in our random forest and initialize

clf = RF(n_estimators = 500)



# Train the classifier

clf = clf.fit(X_228d,target)
# read test data from CSV file 

test_images = pd.read_csv('../input/test.csv')



test_values = test_images.values

test_std = StandardScaler().fit_transform(test_values)

test_228d = pca.transform(test_std)



output_predictions = clf.predict(test_228d)
np.savetxt('submission_rf_500.csv', 

           np.c_[range(1,len(test_images)+1),output_predictions], 

           delimiter=',', 

           header = 'ImageId,Label', 

           comments = '', 

           fmt='%d')