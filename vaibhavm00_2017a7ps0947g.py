import pandas as pd

import numpy as np

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score
df=pd.read_csv("/kaggle/input/dmassign1/data.csv",sep=',')

df.replace({'?':np.nan},inplace=True)
df['Col189']
df.replace({'yes':1,'no':0},inplace=True)
df['Col197'].unique()
df.replace({'la':'LA','me':'ME','M.E.':'ME','sm':'SM'},inplace=True)
df.head()
final_df=df.copy()

final_df.drop(['ID','Class'],axis=1,inplace=True)

for col in final_df.columns[final_df.isnull().any()]:

    if col in ['Col190','Col191','Col192','Col193','Col194','Col195','Col196','Col197']:

        final_df[col].fillna(final_df[col].mode()[0],inplace=True)

    elif col!='Class':

        final_df[col].fillna(final_df[col].astype('float').dropna().mean(),inplace=True)

labels=df['Class'][:1300]


final_df=pd.get_dummies(final_df,columns=['Col190','Col191','Col192','Col193','Col194','Col195','Col196','Col197'])

final_df.head()
from sklearn.preprocessing import MinMaxScaler,StandardScaler

n2=StandardScaler()

final_df=pd.DataFrame(n2.fit_transform(final_df),columns=final_df.columns)

final_df.head()

# FOR VISUALISATION



from sklearn.decomposition import PCA

pca1 = PCA(n_components=2)

d1=pca1.fit_transform(final_df)



from sklearn.cluster import KMeans



wcss = []

for i in range(2, 20):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(final_df)

    wcss.append(kmean.inertia_)

    

plt.plot(range(2,20),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
plt.figure(figsize=(32, 16))

preds1 = []

for i in range(2, 21):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(final_df)

    pred = kmean.predict(final_df)

    preds1.append(pred)

    

    plt.subplot(4, 5, i - 1)

    plt.title(str(i)+" clusters")

    plt.scatter(d1[:, 0], d1[:, 1], c=pred)

    

    centroids = kmean.cluster_centers_

    centroids = pca1.transform(centroids)

    plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)
from sklearn.cluster import AgglomerativeClustering as AC

aggclus = AC(n_clusters = 16,affinity='euclidean',linkage='ward',compute_full_tree='auto')

y_aggclus= aggclus.fit_predict(final_df)

plt.scatter(d1[:, 0], d1[:, 1], c=y_aggclus)
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from scipy.cluster.hierarchy import fcluster

linkage_matrix1 = linkage(final_df, "ward",metric="euclidean")

y_ac=cut_tree(linkage_matrix1, n_clusters = 15).T
plt.scatter(d1[:,0], d1[:,1], c=y_ac[0,:], s=100, label='')

plt.show()
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=100)

pred = dbscan.fit_predict(final_df)

plt.scatter(d1[:, 0], d1[:, 1], c=pred)


kmeans=KMeans(n_clusters=16,random_state=42)

y_pred=kmeans.fit_predict(final_df)



plt.scatter(d1[:,0], d1[:,1], c=y_pred, cmap='rainbow', edgecolors='b')

#mapping function

confusion=np.zeros((16,5),dtype='int64')

for i in range(0,1300):

    confusion[y_pred[i]][df.loc[i]['Class'].astype('int64')-1]+=1



confusion

mapping={}

for i,x in enumerate(confusion):

    mapping[i]=np.argmax(x)

mapping

for i,x in enumerate(y_pred):

    y_pred[i]=mapping[x]

    

np.unique(y_pred)

print(mapping)

print(confusion)
ans_df=pd.DataFrame({'ID':np.array(df.loc[1300:]['ID']),'Class':y_pred[1300:]+1})

ans_df['Class'].unique()
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(ans_df)