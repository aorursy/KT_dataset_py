import numpy as np

from sklearn.cluster import KMeans

import random

import matplotlib as plt
x=[]

for i in range(60):

    a = [10+random.gauss(-2,2),10+random.gauss(-2,2)]

    b = [30+random.gauss(-2,2),30+random.gauss(-2,2)]

    c = [30+random.gauss(-2,2),10+random.gauss(-2,2)]

    x.append(a)

    x.append(b)

    x.append(c)

x=np.array(x)

X=[]

Y=[]

for a in range(60):

    X.append(x[a][0])

    Y.append(x[a][1])

plt.pyplot.scatter(X,Y)
model = KMeans(n_clusters=3,init='random',max_iter=300)

model.fit(x)

a=[]

a.append([2,2])

b=model.predict(a)
x_center=[]

y_center=[]

for a in range(3):

    x_center.append(model.cluster_centers_[a][0])

    y_center.append(model.cluster_centers_[a][1])

plt.pyplot.scatter(x_center,y_center,c='red')

plt.pyplot.scatter(X,Y)