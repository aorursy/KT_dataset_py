# Basic Libraries

import numpy as np

import pandas as pd



import warnings

warnings.simplefilter("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Visualization

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

import seaborn as sns



# Data preprocessing

from sklearn.preprocessing import StandardScaler
# data loading

df = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
# dataframe

df.head()
# Null values

df.isnull().sum().sum()
# Data info

df.info()
# Sample 200

sns.pairplot(df.sample(400, random_state=10), hue="quality")
# Correlation

matrix = df.corr()

plt.figure(figsize=(10,10))

sns.heatmap(matrix, vmax=1, vmin=-1, cmap="bwr", square=True)
# Variables

columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

X = df[columns]

label = df["quality"]



# Scaling

sc = StandardScaler()

sc.fit(X)



X_std = sc.fit_transform(X)
# Library

!pip install factor_analyzer

from factor_analyzer import FactorAnalyzer
# calculate eigen values

eigen_vals = sorted(np.linalg.eigvals(pd.DataFrame(X_std).corr()), reverse=True)



# plot

plt.figure(figsize=(10,6))

plt.plot(eigen_vals, 's-')

plt.xlabel("factor")

plt.ylabel("eigenvalue")
# Create instance

fa = FactorAnalyzer(n_factors=5, rotation="varimax", impute="drop")



# Fitting

fa.fit(X)

result = fa.loadings_

colnames = columns



# Visualization by heatmap

plt.figure(figsize=(10,10))

hm = sns.heatmap(result, cbar=True, annot=True, cmap='bwr', fmt=".2f", 

                 annot_kws={"size":10}, yticklabels=colnames, xticklabels=["factor1", "factor2", "factor3", "factor4", "factor5"], vmax=1, vmin=-1, center=0)

plt.xlabel("factors")

plt.ylabel("variables")
# factor dataframe

factor_df = pd.DataFrame(fa.transform(X), columns=["factor1", "factor2", "factor3", "factor4", "factor5"])

factor_df.head()
# Check correlation fot quality

matrix = pd.concat([factor_df, label], axis=1).corr()

plt.figure(figsize=(10,10))

sns.heatmap(matrix, vmax=1, vmin=-1, cbar=True, annot=True, cmap='bwr', fmt=".2f",  square=True)
# visualization by plot

x = factor_df["factor2"]

y = factor_df["factor4"]

c = label



plt.figure(figsize=(10,8))

plt.scatter(x, y, c=c, alpha=0.7)

plt.xlabel("factor2 (Chlirides)")

plt.ylabel("factor4 (A little sweet and sour and rich)")

plt.colorbar()
# Library

from sklearn.decomposition import PCA



# Create instance

pca = PCA(n_components=5)
# Fitting

pca_result = pca.fit_transform(X)

pca_result = pd.DataFrame(pca_result, columns=("pca1", "pca2", "pca3", "pca4", "pca5"))

pca_result.head()
# Visualization by heatmap

plt.figure(figsize=(10,10))

hm = sns.heatmap(pca.components_.T, cbar=True, annot=True, cmap='bwr', fmt=".2f", 

                 annot_kws={"size":10}, yticklabels=colnames, xticklabels=pca_result.columns, vmax=1, vmin=-1, center=0)

plt.xlabel("pca")

plt.ylabel("variables")
# visualization by plot

x = pca_result["pca3"]

y = pca_result["pca5"]

c = label



plt.figure(figsize=(10,8))

plt.scatter(x, y, c=c, alpha=0.7)

plt.xlabel("PCA3")

plt.ylabel("PCA5")

plt.colorbar()
# Library

from sklearn.decomposition import FastICA



# Create instance

ica = FastICA(n_components=5)
# Fitting

ica_result = ica.fit_transform(X)

ica_result = pd.DataFrame(ica_result, columns=("ica1", "ica2", "ica3", "ica4", "ica5"))

ica_result.head()
# Visualization by heatmap

plt.figure(figsize=(10,10))

hm = sns.heatmap(ica.mixing_, cbar=True, annot=True, cmap='bwr', fmt=".2f", 

                 annot_kws={"size":10}, yticklabels=colnames, center=0)

plt.xlabel("ica")

plt.ylabel("variables")
# visualization by plot

x = ica_result["ica3"]

y = ica_result["ica5"]

c = label



plt.figure(figsize=(10,8))

plt.scatter(x, y, c=c, alpha=0.7)

plt.xlabel("ICA3")

plt.ylabel("ICA5")

plt.colorbar()
# Library

from sklearn.cluster import KMeans



# algorism

distortions = []

for i in range(1,11):

    km = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=100, random_state=10)

    km.fit(X_std)

    distortions.append(km.inertia_)

    

# Plotting distortions

plt.figure(figsize=(10,6))

plt.plot(range(1,11), distortions, marker='o')

plt.xlabel("Number of clusters")

plt.xticks(range(1,11))

plt.ylabel("Distortion")
# Create instance

kmeans = KMeans(n_clusters=6, max_iter=30, init="k-means++", random_state=10)
# Fitting

kmeans.fit(X_std)
# output

cluster = kmeans.labels_



# Combine with PCA results

pca_result["cluster"] = cluster
# visualization, cluster

x = pca_result["pca3"]

y = pca_result["pca5"]

cluster = pca_result["cluster"]



plt.figure(figsize=(10,6))

plt.scatter(x,y,c=cluster)

plt.xlabel("PCA3")

plt.ylabel("PCA5")

plt.title("Cluster analysis result")

plt.colorbar()
# Feature comparison of each cluster

cluster_df = pd.DataFrame(X_std, columns=X.columns)

cluster_df["cluster"] = cluster



cluster_mean = cluster_df.groupby("cluster").mean().T



cluster_std = cluster_df.groupby("cluster").std().T
# Visualization

cluster_name = ["cluster1","cluster2","cluster3","cluster4","cluster5","cluster6"]

fig, ax = plt.subplots(2,3,figsize=(20,12))

plt.subplots_adjust(hspace=0.8)



for i in range(len(cluster_mean.columns)):

    if i <=2:

        ax[0,i].plot(cluster_mean.index, cluster_mean[i])

        ax[0,i].fill_between(cluster_mean.index, cluster_mean[i]+cluster_std[i], cluster_mean[i]-cluster_std[i], color="blue", alpha=0.3)

        ax[0,i].set_title(cluster_name[i])

        ax[0,i].set_xlabel("variables")

        ax[0,i].set_ylabel("Standarlized values")

        ax[0,i].set_ylim([-2,8])

        ax[0,i].tick_params(axis='x', labelrotation=90)

    else:

        ax[1,i-3].plot(cluster_mean.index, cluster_mean[i])

        ax[1,i-3].fill_between(cluster_mean.index, cluster_mean[i]+cluster_std[i], cluster_mean[i]-cluster_std[i], color="blue", alpha=0.3)

        ax[1,i-3].set_title(cluster_name[i])

        ax[1,i-3].set_xlabel("variables")

        ax[1,i-3].set_ylabel("Standarlized values")

        ax[1,i-3].set_ylim([-2,8])

        ax[1,i-3].tick_params(axis='x', labelrotation=90)
# create data

quality_cluster = pd.DataFrame({"cluster":cluster, "quality":df["quality"]})

mapping = {0:"cluster1", 1:"cluster2", 2:"cluster3", 3:"cluster4", 4:"cluster5", 5:"cluster6"}

quality_cluster["cluster"] = quality_cluster["cluster"].map(mapping)

quality_cluster = quality_cluster.sort_values(by="cluster")



# sample count

sample_count = pd.DataFrame({"cluster":quality_cluster["cluster"].value_counts().index,"count":quality_cluster["cluster"].value_counts()})



# merge

quality_cluster = pd.merge(quality_cluster, sample_count, on="cluster", how="left")

quality_cluster["cluster+count"] = [str(str(quality_cluster["cluster"][i]+"\n N="+str(quality_cluster["count"][i]))) for i in range(len(quality_cluster))]



# Visualization by boxplot

plt.figure(figsize=(10,6))

sns.violinplot("cluster+count", "quality", data=quality_cluster, xticklabels=cluster_name)
# Library

from scipy.cluster.hierarchy import dendrogram, linkage



# Create instance

h_cluster = linkage(X, method = 'average', metric = 'euclidean')



# Since the number of N is large, omitted by 100 samples

plt.figure(figsize=(25,6))

dendr = dendrogram(h_cluster, p=100, truncate_mode="lastp")

plt.xticks(fontsize=8)

print("dendrogram")
# Since the number of N is large, level = 3 adjusted

plt.figure(figsize=(25,6))

dendr = dendrogram(h_cluster, p=5, truncate_mode="level")

plt.xticks(fontsize=8)

print("dendrogram")
# Library

from sklearn.mixture import GaussianMixture



# Create instance

gaussian = GaussianMixture(n_components=6, max_iter=30, random_state=10)
# Fitting

gaussian.fit(X_std)
# output

cluster = gaussian.predict(X_std)



# Combine with PCA results

pca_result["cluster"] = cluster
# Probability

probs = gaussian.predict_proba(X_std)

cluster_name = ["cluster1","cluster2","cluster3","cluster4","cluster5","cluster6"]



pd.DataFrame(probs, columns=cluster_name).round(3).head()
# visualization, cluster, with probability

size = 100*probs.max(1)**2 # Square to emphasize



x = pca_result["pca3"]

y = pca_result["pca5"]

cluster = pca_result["cluster"]



plt.figure(figsize=(10,6))

plt.scatter(x,y,c=cluster, s=size, alpha=0.7)

plt.xlabel("PCA3")

plt.ylabel("PCA5")

plt.title("Cluster analysis result")

plt.colorbar()
# Calculate each n_components

n_components = np.arange(1,21)

models = [GaussianMixture(n, covariance_type="full", random_state=10, reg_covar=0.001).fit(X_std) for n in n_components]



plt.figure(figsize=(10,6))

plt.plot(n_components, [m.bic(X_std) for m in models], label='BIC')

plt.plot(n_components, [m.aic(X_std) for m in models], label='AIC')

plt.legend()

plt.xlabel("n_components")

plt.xticks(range(1,21))

print()
# Library

from sklearn.manifold import Isomap



# Create instance

iso = Isomap(n_components=2)
# Fitting

iso.fit(X_std)
# output

iso_projected = iso.transform(X_std)
# visualization, quality

x = iso_projected[:,0]

y = iso_projected[:,1]

qual = label



plt.figure(figsize=(10,6))

plt.scatter(x,y,c=label, alpha=0.7)

plt.xlabel("ISO 1")

plt.ylabel("ISO 2")

plt.title("ISO map visualization")

plt.colorbar()
# Create instance

kmeans = KMeans(n_clusters=6, max_iter=30, init="k-means++", random_state=10)



# Fitting

kmeans.fit(X_std)



# output

cluster = kmeans.labels_



# visualization, quality

x = iso_projected[:,0]

y = iso_projected[:,1]



plt.figure(figsize=(10,6))

plt.scatter(x,y,c=cluster, alpha=0.7)

plt.xlabel("ISO 1")

plt.ylabel("ISO 2")

plt.title("ISO map visualization")

plt.colorbar()
# Library

from sklearn.manifold import TSNE



# Create instance

tsne = TSNE(n_components=2, random_state=10)
# Fitting

tsne.fit(X_std)
# Fit transform

tsne_X = tsne.fit_transform(X_std)
# visualization, quality

x = tsne_X[:,0]

y = tsne_X[:,1]

qual = label



plt.figure(figsize=(10,6))

plt.scatter(x,y,c=label, alpha=0.7)

plt.xlabel("tsne 1")

plt.ylabel("tsne 2")

plt.title("t-SNE map visualization")

plt.colorbar()
# Create instance

kmeans = KMeans(n_clusters=6, max_iter=30, init="k-means++", random_state=10)



# Fitting

kmeans.fit(X_std)



# output

cluster = kmeans.labels_



# visualization, quality

x = tsne_X[:,0]

y = tsne_X[:,1]



plt.figure(figsize=(10,6))

plt.scatter(x,y,c=cluster, alpha=0.7)

plt.xlabel("tsne 1")

plt.ylabel("tsne 2")

plt.title("t-SNE map visualization")

plt.colorbar()