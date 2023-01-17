import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA # Principal Component Analysis module

from sklearn.cluster import KMeans # KMeans clustering 

import matplotlib.pyplot as plt # Python defacto plotting library

import seaborn as sns # More snazzy plotting library

%matplotlib inline 
df=pd.read_csv('../input/movie_metadata.csv')
df.head()
df.shape
df=df.dropna(axis=0)
df.shape
str_list = [] # empty list to contain columns with strings (words)

for colname, colvalue in df.iteritems():

    if type(colvalue[1]) == str:

         str_list.append(colname)

# Get to the numeric columns by inversion            

num_list = df.columns.difference(str_list)
num_list
df_num=df[num_list]
X = df_num.values

from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)
df.plot(y= 'imdb_score', x ='duration',kind='hexbin',gridsize=35, sharex=False, colormap='cubehelix', title='Hexbin of Imdb_Score and Duration')

df.plot(y= 'imdb_score', x ='gross',kind='hexbin',gridsize=45, sharex=False, colormap='afmhot', title='Hexbin of Imdb_Score and Gross')
# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(7, 7))

plt.title('Pearson Correlation of Movie Features')

# Draw the heatmap using seaborn

sns.heatmap(df_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="YlGnBu")

df_num.astype(float).corr()
mean_vec = np.mean(X_std, axis=0)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]



# Sort from high to low

eig_pairs.sort(key = lambda x: x[0], reverse= True)



# Calculation of Explained Variance from the eigenvalues

tot = sum(eig_vals)

var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance

cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance
plt.figure(figsize=(8, 5))

plt.bar(range(16), var_exp, alpha=0.3333, align='center', label='individual explained variance', color = 'g')

plt.step(range(16), cum_var_exp, where='mid',label='cumulative explained variance')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.legend(loc='best')

plt.show()
pca = PCA(n_components=9)

x_9d = pca.fit_transform(X_std)
plt.figure(figsize = (7,7))

plt.scatter(x_9d[:,0],x_9d[:,1], c='goldenrod',alpha=0.5)

plt.ylim(-10,30)

plt.show()
kmeans = KMeans(n_clusters=3)

# Compute cluster centers and predict cluster indices

X_clustered = kmeans.fit_predict(x_9d)



# Define our own color map

LABEL_COLOR_MAP = {0 : 'r',1 : 'g',2 : 'b'}

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]



# Plot the scatter digram

plt.figure(figsize = (7,7))

plt.scatter(x_9d[:,0],x_9d[:,2], c= label_color, alpha=0.5) 

plt.show()