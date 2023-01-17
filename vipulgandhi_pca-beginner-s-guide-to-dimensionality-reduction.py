import pandas as pd

Iris = pd.read_csv("../input/iris/Iris.csv")

X = Iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

y=Iris.Species
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)
import numpy as np



X_mean = np.mean(X, axis=0)

# cov_mat = np.cov(X)

cov_mat = (X - X_mean).T.dot((X - X_mean)) / (X.shape[0]-1)

print('Covariance matrix \n%s' %cov_mat)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)



print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)
u,s,v = np.linalg.svd(X.T)

u
# Make a list of (eigenvalue, eigenvector) tuples

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]



# Sort the (eigenvalue, eigenvector) tuples from high to low

eig_pairs.sort(key=lambda x: x[0], reverse=True)



# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in descending order:')

for i in eig_pairs:

    print(i[0])
tot = sum(eig_vals)

var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]

cum_var_exp = np.cumsum(var_exp)

cum_var_exp
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),

                      eig_pairs[1][1].reshape(4,1)))



print('Matrix W:\n', matrix_w)
Y = X.dot(matrix_w)
import matplotlib.pyplot as plt



with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(6, 4))

    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), ('blue', 'red', 'green')):

        plt.scatter(Y[y==lab, 0], Y[y==lab, 1], label=lab, c=col)

    plt.xlabel('Principal Component 1')

    plt.ylabel('Principal Component 2')

    plt.legend(loc='lower center')

    plt.tight_layout()

    plt.show()
from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=2)

Y_sklearn = sklearn_pca.fit_transform(X)



sklearn_pca.explained_variance_ratio_
with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(6, 4))

    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),

                        ('blue', 'red', 'green')):

        plt.scatter(Y_sklearn[y==lab, 0],

                    Y_sklearn[y==lab, 1],

                    label=lab,

                    c=col)

    plt.xlabel('Principal Component 1')

    plt.ylabel('Principal Component 2')

    plt.legend(loc='lower center')

    plt.tight_layout()

    plt.show()
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
from sklearn.model_selection import train_test_split

train_img, test_img, train_lbl, test_lbl = train_test_split( X, y, test_size=0.15, random_state=0)
from sklearn.decomposition import PCA



# scikit-learn choose the minimum number of principal components such that 95% of the variance is retained.

pca = PCA(0.95)

pca.fit(train_img)

print(pca.n_components_)

train_img = pca.transform(train_img)

test_img = pca.transform(test_img)
from sklearn.decomposition import IncrementalPCA

n_batches = 100



inc_pca = IncrementalPCA(n_components=154)



for X_batch in np.array_split(train_img, n_batches):

        inc_pca.partial_fit(X_batch)

        

X_mnist_reduced = inc_pca.transform(train_img)
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import make_swiss_roll



X, y = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)



axes = [-11.5, 14, -2, 23, -12, 15]



fig = plt.figure(figsize=(6, 5))

ax = fig.add_subplot(111, projection='3d')



ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.hot)

ax.view_init(10, -70)

ax.set_xlabel("$x_1$", fontsize=18)

ax.set_ylabel("$x_2$", fontsize=18)

ax.set_zlabel("$x_3$", fontsize=18)

ax.set_xlim(axes[0:2])

ax.set_ylim(axes[2:4])

ax.set_zlim(axes[4:6])



plt.show()
from sklearn.manifold import TSNE



tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

X_reduced = tsne.fit_transform(X)
import seaborn as sns; sns.set()

import matplotlib.pyplot as plt



#ax = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue = y)



plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.hot)
from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)

X_reduced = rbf_pca.fit_transform(X)



import seaborn as sns; sns.set()

import matplotlib.pyplot as plt



#ax = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue = y)



plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.hot)
from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components = 2, kernel="poly", gamma=0.04)

X_reduced = rbf_pca.fit_transform(X)



import seaborn as sns; sns.set()

import matplotlib.pyplot as plt



#ax = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue = y)



plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.hot)
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)

X_reduced = lle.fit_transform(X)

X_reduced.shape



import seaborn as sns; sns.set()

import matplotlib.pyplot as plt



#ax = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue = y)



plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.hot)
from sklearn.manifold import Isomap

isomap = Isomap(n_components=2, n_neighbors=10)

X_reduced = isomap.fit_transform(X)

X_reduced.shape



import seaborn as sns; sns.set()

import matplotlib.pyplot as plt



#ax = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue = y)



plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.hot)