# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.decomposition import PCA
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

df = pd.read_csv("../input/ytl_aggregated.csv", index_col="opiskelijaKoodi")
df.head()
df = df.fillna(-1)
from scipy.spatial.distance import norm
all_sse = []
for k in range(4, 20):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df)
    sse = 0
    for i in range(len(df)):
        sse += norm(kmeans.cluster_centers_[kmeans.labels_[i]] - df.iloc[i])
    all_sse.append(sse)
    print("At cluster number {}".format(k))
        
plt.figure(figsize=(20, 8))
plt.plot(np.arange(4, 20), all_sse)
plt.show()

umap = UMAP(n_components=2, n_neighbors=33, min_dist=0.4, random_state=1)
Y = umap.fit_transform(df)
import datashader as ds
import datashader.transfer_functions as tf

subject = "Matematiikka, pitkä oppimäärä"
umap_df = pd.DataFrame(Y, columns=["x_col", "y_col"], index=df.index)
umap_df[subject] = df[subject].astype("category")
cvs = ds.Canvas(plot_width=1400, plot_height=400)
agg = cvs.points(umap_df, 'x_col', 'y_col', ds.count_cat(subject))
color_key = {-1: "orange", 0: 'aqua', 2: 'aqua', 3: 'aqua', 4: 'aqua', 5: 'aqua', 6: 'aqua', 7: 'aqua'}
tf.set_background(tf.shade(agg, color_key=color_key, how='eq_hist'),"black")

df_sample = df.iloc[np.random.randint(0, len(df), 10000)]
umap = UMAP(n_components=2, n_neighbors=10, min_dist=0.3, random_state=4)
Y_sample = umap.fit_transform(df_sample)
data1 = [go.Scatter(x=Y_sample[:, 0], y=Y_sample[:, 1], marker=dict(size=4), mode="markers", name="Kaikki")]
idxs = df_sample["Matematiikka, pitkä oppimäärä"] != -1
data2 = [go.Scatter(x=Y_sample[idxs][:, 0], y=Y_sample[idxs][:, 1], marker=dict(size=4), mode="markers", name="Matematiikka, pitkä oppimäärä")]
idxs = df_sample["Matematiikka, lyhyt oppimäärä"] != -1
data3 = [go.Scatter(x=Y_sample[idxs][:, 0], y=Y_sample[idxs][:, 1], marker=dict(size=4), mode="markers", name="Matematiikka, lyhyt oppimäärä")]
idxs = df_sample["Fysiikka"] != -1
data4 = [go.Scatter(x=Y_sample[idxs][:, 0], y=Y_sample[idxs][:, 1], marker=dict(size=4), mode="markers", name="Fysiikka")]
idxs = df_sample["Äidinkieli, ruotsi"] != -1
data5 = [go.Scatter(x=Y_sample[idxs][:, 0], y=Y_sample[idxs][:, 1], marker=dict(size=4), mode="markers", name="Äidinkieli, ruotsi")]
idxs = df_sample["Terveystieto"] != -1
data6 = [go.Scatter(x=Y_sample[idxs][:, 0], y=Y_sample[idxs][:, 1], marker=dict(size=4), mode="markers", name="Terveystieto")]

iplot(go.Figure(data1 + data2 + data3 + data4 + data5 + data6, layout=go.Layout(title="Ylioppilaat 2015S-2018K otos (n=10000)")))
#plt.figure(figsize=(20, 8))
#colors = np.where(df["Filosofia"] != -1, "green", "red")
#plt.scatter(Y[:, 0], Y[:, 1], s=1, c=colors)
#plt.tight_layout()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(df)
df["cluster"] = kmeans.labels_
plt.figure(figsize=(20, 8))
plt.scatter(Y[:, 0], Y[:, 1], s=1, c=df["cluster"], cmap="Paired")
plt.tight_layout()
red_cluster = df2[Y[:, 0] < -20]
iplot([go.Bar(x=red_cluster.columns, y=red_cluster.sum())])
print("Keskiarvo: \n{}".format(red_cluster.mean()))
df["cluster"].value_counts()
print("Koko datasetille:")
print("Keskiarvo: {}".format(round(df.replace(-1, np.nan).drop(["lukioKoko", "sukupuoli", "cluster"], axis=1).mean().mean(), 3)))
print("Opiskelijoiden lkm: {}".format(len(df)))
print("Sukupuoli: {}% M, {}% N".format(round((df["sukupuoli"] == 0).sum() / len(df), 2),
                                       round((df["sukupuoli"] == 1).sum() / len(df), 2)))
df2 = df.replace(-1, np.nan)
group = df2.groupby("cluster").get_group(0)
plottable = group.sum().drop(["lukioKoko", "sukupuoli", "cluster"])
fig = go.Figure([go.Bar(x=plottable.index, y=plottable)])
iplot(fig)
print("Keskiarvo: {}".format(round(group.drop(["lukioKoko", "sukupuoli", "cluster"], axis=1).mean().mean(), 3)))
print("Opiskelijoiden lkm: {}".format(len(group)))
print("Sukupuoli: {}% M, {}% N".format(round((group["sukupuoli"] == 0).sum() / len(group), 2) * 100,
                                       round((group["sukupuoli"] == 1).sum() / len(group), 2) * 100))
df = df.replace(-1, np.nan)
group = df.groupby("cluster").get_group(1)
plottable = group.sum().drop(["lukioKoko", "sukupuoli", "cluster"])
iplot([go.Bar(x=plottable.index, y=plottable)])
print("Keskiarvo: {}".format(round(group.drop(["lukioKoko", "sukupuoli", "cluster"], axis=1).mean().mean(), 3)))
print("Opiskelijoiden lkm: {}".format(len(group)))
print("Sukupuoli: {}% M, {}% N".format(round((group["sukupuoli"] == 0).sum() / len(group), 2) * 100,
                                       round((group["sukupuoli"] == 1).sum() / len(group), 2) * 100))
df = df.replace(-1, np.nan)
group = df.groupby("cluster").get_group(2)
plottable = group.sum().drop(["lukioKoko", "sukupuoli", "cluster"])
iplot([go.Bar(x=plottable.index, y=plottable)])
print("Keskiarvo: {}".format(round(group.drop(["lukioKoko", "sukupuoli", "cluster"], axis=1).mean().mean(), 3)))
print("Opiskelijoiden lkm: {}".format(len(group)))
print("Sukupuoli: {}% M, {}% N".format(round((group["sukupuoli"] == 0).sum() / len(group), 2) * 100,
                                       round((group["sukupuoli"] == 1).sum() / len(group), 2) * 100))
df = df.replace(-1, np.nan)
group = df.groupby("cluster").get_group(3)
plottable = group.sum().drop(["lukioKoko", "sukupuoli", "cluster"])
iplot([go.Bar(x=plottable.index, y=plottable)])
print("Keskiarvo: {}".format(round(group.drop(["lukioKoko", "sukupuoli", "cluster"], axis=1).mean().mean(), 3)))
print("Opiskelijoiden lkm: {}".format(len(group)))
print("Sukupuoli: {}% M, {}% N".format(round((group["sukupuoli"] == 0).sum() / len(group), 2) * 100 ,
                                       round((group["sukupuoli"] == 1).sum() / len(group), 2) * 100))
df = df.replace(-1, np.nan)
group = df.groupby("cluster").get_group(4)
plottable = group.sum().drop(["lukioKoko", "sukupuoli", "cluster"])
iplot([go.Bar(x=plottable.index, y=plottable)])
print("Keskiarvo: {}".format(round(group.drop(["lukioKoko", "sukupuoli", "cluster"], axis=1).mean().mean(), 3)))
print("Opiskelijoiden lkm: {}".format(len(group)))
print("Sukupuoli: {}% M, {}% N".format(round((group["sukupuoli"] == 0).sum() / len(group), 2) * 100,
                                       round((group["sukupuoli"] == 1).sum() / len(group), 2) * 100))
df = df.replace(-1, np.nan)
group = df.groupby("cluster").get_group(5)
plottable = group.sum().drop(["lukioKoko", "sukupuoli", "cluster"])
iplot([go.Bar(x=plottable.index, y=plottable)])
print("Keskiarvo: {}".format(round(group.drop(["lukioKoko", "sukupuoli", "cluster"], axis=1).mean().mean(), 3)))
print("Opiskelijoiden lkm: {}".format(len(group)))
print("Sukupuoli: {}% M, {}% N".format(round((group["sukupuoli"] == 0).sum() / len(group), 2) * 100,
                                       round((group["sukupuoli"] == 1).sum() / len(group), 2) * 100))
df = df.replace(-1, np.nan)
group = df.groupby("cluster").get_group(6)
plottable = group.sum().drop(["lukioKoko", "sukupuoli", "cluster"])
iplot([go.Bar(x=plottable.index, y=plottable)])
print("Keskiarvo: {}".format(round(group.drop(["lukioKoko", "sukupuoli", "cluster"], axis=1).mean().mean(), 3)))
print("Opiskelijoiden lkm: {}".format(len(group)))
print("Sukupuoli: {}% M, {}% N".format(round((group["sukupuoli"] == 0).sum() / len(group), 2) * 100,
                                       round((group["sukupuoli"] == 1).sum() / len(group), 2) * 100))
df = df.replace(-1, np.nan)
group = df.groupby("cluster").get_group(7)
plottable = group.sum().drop(["lukioKoko", "sukupuoli", "cluster"])
iplot([go.Bar(x=plottable.index, y=plottable)])
print("Keskiarvo: {}".format(round(group.drop(["lukioKoko", "sukupuoli", "cluster"], axis=1).mean().mean(), 3)))
print("Opiskelijoiden lkm: {}".format(len(group)))
print("Sukupuoli: {}% M, {}% N".format(round((group["sukupuoli"] == 0).sum() / len(group), 2) * 100,
                                       round((group["sukupuoli"] == 1).sum() / len(group), 2) * 100))
df = df.replace(-1, np.nan)
group = df.groupby("cluster").get_group(8)
plottable = group.sum().drop(["lukioKoko", "sukupuoli", "cluster"])
iplot([go.Bar(x=plottable.index, y=plottable)])
print("Keskiarvo: {}".format(round(group.drop(["lukioKoko", "sukupuoli", "cluster"], axis=1).mean().mean(), 3)))
print("Opiskelijoiden lkm: {}".format(len(group)))
print("Sukupuoli: {}% M, {}% N".format(round((group["sukupuoli"] == 0).sum() / len(group), 2) * 100,
                                       round((group["sukupuoli"] == 1).sum() / len(group), 2) * 100))
df = df.replace(-1, np.nan)
group = df.groupby("cluster").get_group(7)
plottable = group.count().drop(["lukioKoko", "sukupuoli", "cluster"]) / df.count().drop(["lukioKoko", "sukupuoli", "cluster"])
iplot([go.Bar(x=plottable.index, y=plottable)])
print("Keskiarvo: {}".format(round(group.drop(["lukioKoko", "sukupuoli", "cluster"], axis=1).mean().mean(), 3)))
print("Opiskelijoiden lkm: {}".format(len(group)))
print("Sukupuoli: {}% M, {}% N".format(round((group["sukupuoli"] == 0).sum() / len(group), 2) * 100,
                                       round((group["sukupuoli"] == 1).sum() / len(group), 2) * 100))
df2 = df.replace(-1, 0)
df2.head()
from copy import deepcopy
import numba

#@numba.njit(fastmath=True)
def dist(x, y):
    result = 0.0
    for i in range(x.shape[0]):
        if x[i] == 0 or y[i] == 0:
            result += 2 * (x[i] - y[i]) ** 2
        else:
            result += (x[i] - y[i]) ** 2
    return np.sqrt(result)

#@numba.njit()
def dists(x, y):
    if len(x.shape) == 2 and len(y.shape) == 2:
        results = []
        for i in range(x.shape[0]):
            results.append(dist(x[i], y[i]))
        results = np.array(results)
        return np.sqrt((results ** 2).sum())
    elif len(x.shape) == 2:
        results = []
        for i in range(x.shape[0]):
            results.append(dist(x[i], y))
        return np.array(results)
    else:
        results = []
        for i in range(y.shape[0]):
            results.append(dist(x, y[i]))
        return np.array(results)
from scipy.spatial.distance import mahalanobis
import scipy
def dists(x, y, invcovm):
    
    if len(x.shape) == 2 and len(y.shape) == 2:
        results = []
        for i in range(x.shape[0]):
            results.append(mahalanobis(x[i], y[i], invcovm))
        results = np.array(results)
        return np.sqrt((results ** 2).sum())
    elif len(y.shape) == 2:
        results = []
        for i in range(y.shape[0]):
            results.append(mahalanobis(x, y[i], invcovm))
        return np.array(results)
    
# Number of clusters
k = 9
X = np.array(df2.drop(["lukioKoko", "sukupuoli"], axis=1))
np.random.seed(0)
C = np.random.choice(7, (k, len(df2.columns) - 2))
from copy import deepcopy
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
covm = df2.drop(["lukioKoko", "sukupuoli"], axis=1).cov()
invcovm = scipy.linalg.inv(covm)
error = dists(C, C_old, invcovm)
# Loop will run till the error becomes zero
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dists(X[i], C, invcovm)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dists(C, C_old, invcovm)
    print(error)
pd.Series(clusters).value_counts()
df2 = df2.replace(0, np.nan)
df2["cluster"] = clusters
group = df2.groupby("cluster").get_group(2)
plottable = group.sum().drop(["lukioKoko", "sukupuoli", "cluster"])
iplot([go.Bar(x=plottable.index, y=plottable)])
print("Keskiarvo: {}".format(round(group.drop(["lukioKoko", "sukupuoli", "cluster"], axis=1).mean().mean(), 3)))
print("Opiskelijoiden lkm: {}".format(len(group)))
print("Sukupuoli: {}% M, {}% N".format(round((group["sukupuoli"] == 0).sum() / len(group), 2),
                                       round((group["sukupuoli"] == 1).sum() / len(group), 2)))
import sompy
df = pd.read_csv("../input/ytl_aggregated.csv", index_col="opiskelijaKoodi").fillna(-1)
som = sompy.SOMFactory().build(np.array(df), normalization = 'var', initialization='random', component_names=df.columns)
som.train(n_job=1, verbose=False, train_rough_len=2, train_finetune_len=5)
view2D  = sompy.visualization.mapview.View2D(10,10,"rand data",text_size=10)
view2D.show(som, col_sz=4, which_dim="all", desnormalize=True)
from sompy.visualization.bmuhits import BmuHitsView

vhts  = BmuHitsView(4,4,"Hits Map",text_size=8)
vhts.show(som, anotate=True, onlyzeros=False, labelsize=8, cmap="Greys", logaritmic=False)
from sompy.visualization.hitmap import HitMapView
som.cluster(9)
hits  = HitMapView(20, 20,"Clustering",text_size=12)
a=hits.show(som)
som.cluster_labels.shape
dft = df2.fillna(-1)
dft = dft[(dft["Äidinkieli, ruotsi"] == -1) & (dft["Äidinkieli, suomi"] == -1)].replace(-1, np.nan)
iplot([go.Bar(x=dft.columns, y=dft.sum() / dft.sum().max())] + [go.Bar(x=df2.columns, y=df2.sum() / df2.sum().max())])
dft.mean()
