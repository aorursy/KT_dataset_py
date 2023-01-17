import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv')

X = np.array(data.drop('label', axis = 1,inplace = False))

y = np.array(data.label)
def show_image(im):

    f, ax = plt.subplots(1)

    ax.imshow(im.reshape(28,28),cmap = 'binary')

    ax.set_axis_off()
from sklearn.manifold import Isomap
mini_X = X[0:10000]

mini_y = y[0:10000]
iso = Isomap(n_neighbors=5,n_components=2,n_jobs=-1)

iso.fit(mini_X)
transformed_X = iso.transform(mini_X)
f = plt.figure(num = None, figsize = (15,15))

cma = plt.get_cmap('tab10').colors

cols = np.zeros((len(mini_y),4))



for i in range(0,len(cols)):

    cols[i,:3] = cma[mini_y[i]]

cols[:,3] = 0.3





plt.scatter(transformed_X[:,0],

            transformed_X[:,1],

            c = cols)

plt.axis('off')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda_solver = LinearDiscriminantAnalysis()
lda_solver.fit(mini_X, mini_y)
mini_X_lda = lda_solver.fit_transform(mini_X, mini_y)
f = plt.figure(num = None, figsize = (15,15))

cma = plt.get_cmap('tab10').colors

cols = np.zeros((len(mini_y),4))



for i in range(0,len(cols)):

    cols[i,:3] = cma[mini_y[i]]

cols[:,3] = 0.3



plt.scatter(mini_X_lda[:,0],

            mini_X_lda[:,1],

            c = cols)

plt.axis('off')
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

lda_solver = QuadraticDiscriminantAnalysis()

lda_solver.fit(mini_X, mini_y)

f = plt.figure(num = None, figsize = (15,15))

cma = plt.get_cmap('tab10').colors

cols = np.zeros((len(mini_y),4))



for i in range(0,len(cols)):

    cols[i,:3] = cma[mini_y[i]]

cols[:,3] = 0.3



plt.scatter(mini_X_lda[:,0],

            mini_X_lda[:,1],

            c = cols)

plt.axis('off')