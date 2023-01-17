import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv(r"../input/CC GENERAL.csv")
df.head(2)
df.info()
df.shape
df.describe()
missing = df.isna().sum()
print(missing);
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis');
df.dropna(inplace=True)
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis');
df.drop('CUST_ID', axis=1, inplace=True)
from scipy.cluster.hierarchy import ward, dendrogram, linkage
np.set_printoptions(precision=4, suppress=True)
distance = linkage(df, 'ward')
plt.figure(figsize=(25,10))
plt.title("Hierachical Clustering Dendrogram")
plt.xlabel("Index")
plt.ylabel("Ward's distance")
dendrogram(distance,
           leaf_rotation=90.,
           leaf_font_size=9.,);
plt.axhline(150000, c='k');
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('Index')
plt.ylabel("Ward's distance")
dendrogram(distance, truncate_mode='lastp',
           p=6, leaf_rotation=0., leaf_font_size=12.,
           show_contracted=True);
plt.axhline(150000, c='k');
from sklearn.cluster import KMeans
sse_ = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, random_state=5).fit(df)
    sse_.append([k, kmeans.inertia_])
plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1])
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances (sse)')
plt.title('Elbow Method For Optimal k');
from sklearn.metrics import silhouette_score
sse_ = []
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=5).fit(df)
    sse_.append([k, silhouette_score(df, kmeans.labels_)])
plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1]);

k = n_clusters=8
kmeans = KMeans(k)
clusters = kmeans.fit_predict(df)
plt.hist(clusters, bins=range(k+1))
plt.title('# Customers per Cluster')
plt.xlabel('Cluster')
plt.ylabel('# Customers')
plt.show()
df.columns
data = df.copy()
data["cluster"] = clusters
cols = list(data.columns)
cols
sns.countplot(x='cluster', data=data)
sns.pairplot(data, hue="cluster")
