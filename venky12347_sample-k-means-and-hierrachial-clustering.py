import pandas                  as     pd

import numpy                   as     np

from   sklearn.cluster         import AgglomerativeClustering, KMeans

import sklearn.datasets

from   scipy.cluster.hierarchy import dendrogram, linkage

from   matplotlib              import pyplot as plt

%matplotlib inline

np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
data     =  sklearn.datasets.load_iris()

x        =  pd.DataFrame(data.data, columns = list(data.feature_names))
z   = linkage(x, method = 'median')



plt.figure(figsize=(20,7))



den = dendrogram(z)



plt.title('Dendrogram for the clustering of the dataset iris)')

plt.xlabel('Type')

plt.ylabel('Euclidean distance in the space with other variables')
cluster_H = AgglomerativeClustering(n_clusters = 3, affinity = 'cosine', linkage = 'complete')
model = cluster_H.fit(x)

print(model)
ms          = np.column_stack((data.target,model.labels_))

df          = pd.DataFrame(ms, columns = ['Actual', 'Clusters'])

pd.crosstab(df['Actual'], df['Clusters'], margins=True)
n_clusters = 3

plt.figure()

plt.axes([0, 0, 1, 1])

for l, c in zip(np.arange(n_clusters), 'rgbk'):

    plt.plot(x[model.labels_ == l].T, c=c, alpha=.5)

    plt.axis('tight')

    plt.axis('off')

    plt.suptitle("AgglomerativeClustering(affinity=%s)" % 'cosine', size=20)
df['Clusters']  = model.labels_.astype('int32')

df['Target']    = data.target.astype('int32')
pd.crosstab(df['Target'], df['Clusters'])
iris_X  = x.values

iris_X  = np.array(iris_X)

iris_Y1 = df['Clusters']

iris_Y1 = np.array(iris_Y1)
plt.scatter(iris_X[iris_Y1 == 0, 0], iris_X[iris_Y1 == 0, 1], s = 80, c = 'orange', label = 'Iris-setosa')

plt.scatter(iris_X[iris_Y1 == 1, 0], iris_X[iris_Y1 == 1, 1], s = 80, c = 'yellow', label = 'Iris-versicolour')

plt.scatter(iris_X[iris_Y1 == 2, 0], iris_X[iris_Y1 == 2, 1], s = 80, c = 'green', label = 'Iris-virginica')

plt.legend()
wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)

plt.title('The elbow method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')
cluster_Kmeans = KMeans(n_clusters = 3, random_state = 25)

model_kmeans   = cluster_Kmeans.fit(x)

est_kmeans    = model_kmeans.labels_

est_kmeans
df.drop(columns       = ['Clusters', 'Target'], inplace = True)

df['Clusters']        = est_kmeans

df['Target']          = data.target

pd.crosstab(df['Target'], df['Clusters'])
iris_X = x.values

iris_X = np.array(iris_X)

iris_Y1 = est_kmeans

iris_Y1 = np.array(iris_Y1)
plt.scatter(iris_X[iris_Y1 == 0, 0], iris_X[iris_Y1 == 0, 1], s = 80, c = 'orange', label = 'Iris-setosa')

plt.scatter(iris_X[iris_Y1 == 1, 0], iris_X[iris_Y1 == 1, 1], s = 80, c = 'yellow', label = 'Iris-versicolour')

plt.scatter(iris_X[iris_Y1 == 2, 0], iris_X[iris_Y1 == 2, 1], s = 80, c = 'green', label = 'Iris-virginica')

plt.legend()
n_clusters = 3

plt.figure()

plt.axes([0, 0, 1, 1])

for l, c in zip(np.arange(n_clusters), 'rgbk'):

    plt.plot(x[model_kmeans.labels_ == l].T, c=c, alpha=.5)

    plt.axis('tight')

    plt.axis('off')

    plt.suptitle("K Means( clusters = %d)" % n_clusters, size=20)