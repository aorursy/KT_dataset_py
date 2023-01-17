# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
AB_NYC_df = pd.read_csv( '/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
#show Data
AB_NYC_df.head()
#show data shape
AB_NYC_df.shape
collumn = ['host_id','neighbourhood_group','latitude','longitude','room_type','price','number_of_reviews','reviews_per_month']
datadf = AB_NYC_df[collumn]

# cleaning NaN data 
datadf = datadf.fillna(0)
datadf.head()
# Convert categorical data to indexing number
citylist = pd.unique(AB_NYC_df.neighbourhood_group).tolist()
roomlist = pd.unique(AB_NYC_df.room_type).tolist()

datadf['neighbourhood_group'] = pd.Categorical(datadf['neighbourhood_group'],categories=citylist)
datadf['neighbourhood_group'] = datadf['neighbourhood_group'].cat.codes

datadf['room_type'] = pd.Categorical(datadf['room_type'],categories=roomlist)
datadf['room_type'] = datadf['room_type'].cat.codes
datadf.head()
# Memory concerned 
# Randomly sampling n record 
samples = datadf.sample(n=100)
# Min-Max normalization
samples=(samples-samples.min())/(samples.max()-samples.min())

samples = samples.loc[:, collumn].values
# Calculate Hierarchical Clustering Complete-Link 
from scipy.cluster.hierarchy import complete, fcluster, dendrogram, linkage
z = complete(samples)
z
%matplotlib inline
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
import math
plt.figure(figsize=(20,10));
plt.title("Hierarchical Clustering")
plt.xlabel('Rooms/Houses')
plt.ylabel('Distances(Euclidean)')
dendrogram(z);
# Sorted index from z
n = len(z) + 1
C = dict()
for k in range(len(z)):
  c1, c2 = int(z[k][0]), int(z[k][1])
  c1 = [c1] if c1 < n else C.pop(c1)
  c2 = [c2] if c2 < n else C.pop(c2)
  C[n+k] = c1 + c2
    
ID = C[2*len(z)]
# Sort samples by Z order
samples = samples[ID]
samples
# Plot features heatmap from sorted samples

fig, axes = plt.subplots(2, 1, figsize=(20,20),gridspec_kw={'height_ratios': [1, 2]})

p = dendrogram(z,ax=axes[0],no_labels=True)
axes[0].set_ylabel('Euclidean distances')

datamap = np.array(samples).T
sns.heatmap(datamap,ax=axes[1],cmap="YlGnBu",cbar=False,xticklabels=ID,yticklabels=collumn)
plt.title("Hierarchical Clustering")
plt.xlabel("Samples")
plt.ylabel("Features")

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
