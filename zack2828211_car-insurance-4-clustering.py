import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import os

print(os.listdir("../input"))

pd.set_option('max_columns', 1000)

pd.set_option('max_rows', 10)

x= pd.read_csv('../input/clustering_x.csv',encoding='latin1')
from sklearn.cluster import KMeans

Sum_of_squared_distances = []

K = range(1,15)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(x)

    Sum_of_squared_distances.append(km.inertia_)

import matplotlib.pyplot as plt

plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')

plt.show()
iteration = 500 

model = KMeans(n_clusters =5, n_jobs = 4, max_iter = iteration)

model.fit(x) 



print(pd.Series(model.labels_).value_counts())



pd.DataFrame(model.cluster_centers_) 



new_data=pd.concat([x, pd.Series(model.labels_, index = x.index)], axis = 1) 

new_data.rename(columns={0:'clusters'},inplace=True) 
new_data2=new_data.loc[:5000,]

from sklearn.manifold import TSNE

tsne = TSNE()

tsne.fit_transform(new_data2) 

tsne = pd.DataFrame(tsne.embedding_, index = new_data2.index) 

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] 

plt.rcParams['axes.unicode_minus'] = False 

#不同类别用不同颜色和样式绘图

d = tsne[new_data2['clusters'] == 0]

plt.plot(d[0], d[1], 'r.')

d = tsne[new_data2['clusters'] == 1]

plt.plot(d[0], d[1], 'go')

d = tsne[new_data2['clusters'] == 2]

plt.plot(d[0], d[1], 'b*')

d = tsne[new_data2['clusters'] == 3]

plt.plot(d[0], d[1], 'k*')

d = tsne[new_data2['clusters'] == 4]

plt.plot(d[0], d[1], 'y.')

plt.show()
from sklearn.cluster import DBSCAN

model = DBSCAN(eps=1000,min_samples=1500)

model.fit(x) 



pd.Series(model.labels_).value_counts()



new_data=pd.concat([x, pd.Series(model.labels_, index = x.index)], axis = 1) 

new_data.rename(columns={0:'clusters'},inplace=True) 
new_data2=new_data.loc[:5000,]

from sklearn.manifold import TSNE

tsne = TSNE()

tsne.fit_transform(new_data2) 

tsne = pd.DataFrame(tsne.embedding_, index = new_data2.index) 

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] 

plt.rcParams['axes.unicode_minus'] = False 

#不同类别用不同颜色和样式绘图

d = tsne[new_data2['clusters'] == 0]

plt.plot(d[0], d[1], 'r.')

d = tsne[new_data2['clusters'] == 1]

plt.plot(d[0], d[1], 'go')

d = tsne[new_data2['clusters'] == -1]

plt.plot(d[0], d[1], 'b*')

d = tsne[new_data2['clusters'] == 2]

plt.plot(d[0], d[1], 'k*')

plt.show()

from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=7)



x=x[:20000]

model.fit(x)



print(pd.Series(model.labels_).value_counts())



new_data=pd.concat([x, pd.Series(model.labels_, index = x.index)], axis = 1) 

new_data.rename(columns={0:'clusters'},inplace=True) 
new_data2=new_data.loc[:5000,]

from sklearn.manifold import TSNE

tsne = TSNE()

tsne.fit_transform(new_data2) 

tsne = pd.DataFrame(tsne.embedding_, index = new_data2.index) 

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] 

plt.rcParams['axes.unicode_minus'] = False 

#不同类别用不同颜色和样式绘图

d = tsne[new_data2['clusters'] == 3]

plt.plot(d[0], d[1], 'r.')

d = tsne[new_data2['clusters'] == 2]

plt.plot(d[0], d[1], 'go')

d = tsne[new_data2['clusters'] == 4]

plt.plot(d[0], d[1], 'b*')

d = tsne[new_data2['clusters'] == 6]

plt.plot(d[0], d[1], 'k*')

d = tsne[new_data2['clusters'] == 0]

plt.plot(d[0], d[1], 'y.')

plt.show()
