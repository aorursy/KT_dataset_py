import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

%matplotlib inline
df = pd.read_csv('../input/Pokemon.csv')
def basicinfo(df):

    print(df.head())

    print(df.describe())

    print(df.info())

basicinfo(df)
df = df.drop('#', axis = 1)

df.head()
features = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 

            'Sp. Def', 'Speed', 'Generation', 'Legendary']

def df_view(df, features):

    sns.set_style("whitegrid")

    fig, ax = plt.subplots(3,3, figsize = (16,20))

    k = 0

    for i in range(3):

        for j in range(3):

            sns.swarmplot(x = 'Type 1', y = features[k], data = df, 

                          ax = ax[i,j], palette = "hls", split = True)

            plt.setp(ax[i,j].get_xticklabels(), rotation = 90)

            k += 1 

    return(fig)
p = df_view(df, features)
features_spec = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

corr = df[features_spec].corr()

plt.figure(figsize = (8,8))

sns.heatmap(corr, annot = True)
def scalePCA(df, features):

    scale = StandardScaler()

    df_pca = scale.fit_transform(df[features])

    pca = PCA(n_components=2)

    df_pca = pd.DataFrame(pca.fit_transform(df_pca))

    df_pca.columns = ['PC1', 'PC2']

    df_final = pd.concat([df, df_pca], axis = 1)

    return(df_final)
df_pca = scalePCA(df, features_spec)

df_pca.head()
plt.figure(figsize = (8,8))

sns.set_style("whitegrid")

sns.lmplot(x = 'PC1', y = 'PC2', data = df_pca, hue = 'Type 1', fit_reg = False)
plt.figure(figsize = (8,8))

sns.set_style("whitegrid")

sns.lmplot(x = 'PC1', y = 'PC2', data = df_pca, hue = 'Legendary', fit_reg = False)
km_score = {}

for n in range(2,10):

    km = KMeans(n_clusters = n)

    km_pca = km.fit(df_pca[['PC1', 'PC2']])

    cluster_labels = km_pca.predict(df_pca[['PC1', 'PC2']])

    silhouette_avg = silhouette_score(df_pca[['PC1', 'PC2']], cluster_labels)

    km_score[n] = silhouette_avg

km_score
km2 = KMeans(n_clusters = 2)

km2_pca = km2.fit(df_pca[['PC1', 'PC2']])

df_pca_2 = pd.concat([df_pca, pd.DataFrame(km2_pca.labels_)], axis = 1)

df_pca_2.rename(columns = {0:'kmeans'}, inplace = True)

plt.figure(figsize = (8,8))

sns.set_style("whitegrid")

sns.lmplot(x = 'PC1', y = 'PC2', data = df_pca_2, hue = 'kmeans', fit_reg = False)
km3 = KMeans(n_clusters = 3)

km3_pca = km3.fit(df_pca[['PC1', 'PC2']])

df_pca_3 = pd.concat([df_pca, pd.DataFrame(km3_pca.labels_)], axis = 1)

df_pca_3.rename(columns = {0:'kmeans'}, inplace = True)

plt.figure(figsize = (8,8))

sns.set_style("whitegrid")

sns.lmplot(x = 'PC1', y = 'PC2', data = df_pca_3, hue = 'kmeans', fit_reg = False)
df_pca_3_mean = df_pca_3.groupby('kmeans').mean()

df_pca_3_mean[features_spec] = StandardScaler().fit_transform(df_pca_3_mean[features_spec])

sns.heatmap(df_pca_3_mean[features_spec], center = 0, annot = True)
km4 = KMeans(n_clusters = 4)

km4_pca = km4.fit(df_pca[['PC1', 'PC2']])

df_pca_4 = pd.concat([df_pca, pd.DataFrame(km4_pca.labels_)], axis = 1)

df_pca_4.rename(columns = {0:'kmeans'}, inplace = True)

plt.figure(figsize = (8,8))

sns.set_style("whitegrid")

sns.lmplot(x = 'PC1', y = 'PC2', data = df_pca_4, hue = 'kmeans', fit_reg = False)
df_pca_4_mean = df_pca_4.groupby('kmeans').mean()

df_pca_4_mean.head()

df_pca_4_mean[features_spec] = StandardScaler().fit_transform(df_pca_4_mean[features_spec])

sns.heatmap(df_pca_4_mean[features_spec], center = 0, annot = True)
plt.figure(figsize = (12,6))

sns.countplot(x = 'Type 1', hue = 'kmeans', data = df_pca_4)

plt.xticks(rotation=90)
df_pca_count = df_pca_4.groupby(['Type 1', 'kmeans']).count()

df_pca_sum = df_pca_4.groupby('Type 1').count()

df_pca_final = df_pca_count.div(df_pca_sum, level = 'Type 1') * 100

df_unstack = pd.DataFrame(df_pca_final['Attack'])

df_unstack = pd.DataFrame(df_unstack.unstack(['kmeans', 'Type 1']))

df_unstack = df_unstack.reset_index()

df_unstack.drop('level_0', axis = 1, inplace = True)

df_unstack.columns = ['kmeans', 'Type 1', 'Ratio']
plt.figure(figsize = (12,6))

sns.barplot(x = 'Type 1', y = 'Ratio', hue = 'kmeans', data = df_unstack)

plt.xticks(rotation=90)