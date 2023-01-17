import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

import warnings
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
py.offline.init_notebook_mode(connected = True)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
dataset = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
dataset.head()
dataset.info()
dataset.describe()
lab = dataset["Gender"].value_counts().keys().tolist()
val = dataset["Gender"].value_counts().values.tolist()

trace = go.Pie(labels = lab ,
               values = val ,
               marker = dict(colors =  [ 'royalblue' ,'lime'],
                             line = dict(color = "white",
                                         width =  1.3)
                            ),
               rotation = 20,
               hoverinfo = "label+value+text",
               hole = .5
              )
layout = go.Layout(dict(title = "Customer attrition in data",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                       )
                  )

data = [trace]
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
sns.set(style="darkgrid",font_scale=1.5)
f, axes = plt.subplots(1,3,figsize=(20,8))
sns.distplot(dataset["Age"], ax = axes[0], color = 'y')     
sns.distplot(dataset["Annual Income (k$)"], ax = axes[1], color = 'g')
sns.distplot(dataset["Spending Score (1-100)"],ax = axes[2], color = 'r')
plt.tight_layout()
dz=ff.create_table(dataset.groupby('Gender').mean())
py.iplot(dz)
plt.figure(figsize=(8,4))
sns.heatmap(dataset.corr(),annot=True,cmap=sns.cubehelix_palette(light=1, as_cmap=True),fmt='.2f',linewidths=2)
plt.show()
x = dataset.iloc[:,2:]
print(x.head())
x = x.values
kMeans = KMeans(n_clusters = 3, init = 'k-means++')
y_pred = kMeans.fit_predict(x)
print('Pred:\n', y_pred)
print('\n\ninertia: ', kMeans.inertia_, '\n\nclusters centers:\n', kMeans.cluster_centers_)
result = []
for i in range(1, 12):
    kMeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 123)
    kMeans.fit(x)        
    result.append(kMeans.inertia_)


plt.plot(range(1,12), result)
plt.title('WCSS')
plt.show()
kMeans = KMeans(n_clusters = 6, init = 'k-means++') 
y_pred_kMeans = kMeans.fit_predict(x)
print('Pred:\n', y_pred_kMeans)
print('\n\ninertia: ', kMeans.inertia_, '\n\nclusters centers:\n', kMeans.cluster_centers_)

agglomerative = AgglomerativeClustering(n_clusters = 6, affinity = 'euclidean', linkage = 'ward')
y_pred_agg = agglomerative.fit_predict(x)
print('Pred:\n', y_pred_agg)
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, sharey='col', num = 10, figsize = (15,5))

ax1.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = dataset , c = y_pred_kMeans,s = 100)
ax1.title.set_text('KMeans')

ax2.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = dataset , c = y_pred_agg,s = 100)
ax2.title.set_text('Agglomerative')
f.show()
x = dataset.iloc[:,3:].values
kMeans = KMeans(n_clusters = 6, init = 'k-means++') 
y_pred_kMeans = kMeans.fit_predict(x)
print('Pred:\n', y_pred_kMeans)
print('\n\ninertia: ', kMeans.inertia_, '\n\nclusters centers:\n', kMeans.cluster_centers_)

result = []
for i in range(1, 14):
    kMeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 123)
    kMeans.fit(x)        
    result.append(kMeans.inertia_)


plt.plot(range(1,14), result)
plt.title('WCSS')
plt.show()
print('K-Means')
kMeans = KMeans(n_clusters = 5, init = 'k-means++') 
y_pred_kMeans = kMeans.fit_predict(x)
print('Pred:\n', y_pred_kMeans)
print('\n\ninertia: ', kMeans.inertia_, '\n\nclusters centers:\n', kMeans.cluster_centers_)

print('\n\nAgglomerative')
agglomerative = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_pred_agg = agglomerative.fit_predict(x)
print('Pred:\n', y_pred_agg)
f, (ax1, ax2) = plt.subplots(1, 2, sharey='col', num = 10, figsize = (15,5))

ax1.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = dataset , c = y_pred_kMeans,s = 100)
ax1.title.set_text('K-Means')
ax2.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = dataset , c = y_pred_agg,s = 100)
ax2.title.set_text('Agglomerative')
f.show()