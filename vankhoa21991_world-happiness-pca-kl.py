# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA # Principal Component Analysis module

from sklearn.cluster import KMeans # KMeans clustering 

import matplotlib.pyplot as plt # Python defacto plotting library

import seaborn as sns # More snazzy plotting library

%matplotlib inline 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
country16 = pd.read_csv('../input/2016.csv')

country15 = pd.read_csv('../input/2015.csv')



del country16['Country']

country16.head()
country16.describe()

np.sum(country16.isnull())
myset = set(country16['Region'])

print(myset)
i=0

for region in myset:

    country16['Region'][country16['Region'] == region] = i

    i=i+1

    

a= country16['Region'] == 'Western Europe'
country16.plot(y= 'Happiness Rank', x ='Happiness Score',kind='hexbin',gridsize=35, sharex=False, colormap='cubehelix',figsize=(12,8))
X = country16.values

from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)
sns.factorplot('Economy (GDP per Capita)', 'Happiness Score',data=country16)
# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 10))

plt.title('Pearson Correlation of Movie Features')

# Draw the heatmap using seaborn

sns.heatmap(country16.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="YlGnBu", linecolor='black', annot=True)
mean_vec = np.mean(X_std,axis=0)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print(eig_vals)
# Create a list of (eigenvalue, eigenvector) tuples

eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

#print(eig_pairs)

# Sort from high to low

eig_pairs.sort(key = lambda x: x[0], reverse= True)



# Calculation of Explained Variance from the eigenvalues

tot = sum(eig_vals)



var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance

cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance

print(len(cum_var_exp))
# PLOT OUT THE EXPLAINED VARIANCES SUPERIMPOSED 

#plt.figure(figsize=(10, 5))

#plt.bar(range(9), var_exp, label='individual explained variance')



plt.step(range(12), cum_var_exp, where='mid',label='cumulative explained variance')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.legend(loc='best')
pca = PCA(n_components=4)

x_4d = pca.fit_transform(X_std)

print(x_4d)
plt.figure(figsize=(9,7))

plt.scatter(x_4d[:,0],x_4d[:,1], c='goldenrod',alpha=0.5)

plt.ylim(-10,30)

plt.show()
kmeans = KMeans(n_clusters = 3)



X_clustered = kmeans.fit_predict(x_4d)

LABEL_COLOR_MAP = {0 : 'r',1 : 'g',2 : 'b'}

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]



# Plot the scatter digram

plt.figure(figsize = (7,7))

plt.scatter(x_4d[:,0],x_4d[:,3], c= label_color, alpha=0.5) 

plt.show()
# Create a temp dataframe from our PCA projection data "x_9d"

df = pd.DataFrame(x_4d)

df['X_cluster'] = X_clustered

print(df['X_cluster'])
sns.pairplot(df, hue='X_cluster')