%matplotlib inline
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
iris = pd.read_csv("../input/Iris.csv")
iris.head()
le = LabelEncoder()
le.fit(iris['Species'])
iris['Species'] = le.transform(iris['Species'])
iris_matrix = iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values
iris['Species'].unique()
model_evaluation = {}
for n_clusters in range(2,11):
    cluster_model = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = cluster_model.fit_predict(iris_matrix)
    model_evaluation[n_clusters] = adjusted_rand_score(iris['Species'],cluster_labels)
    
model_evaluation

