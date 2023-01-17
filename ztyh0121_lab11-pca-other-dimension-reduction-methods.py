import numpy as np 
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib.pyplot as plt
iris = datasets.load_iris()
pca = PCA(n_components=2)
X = iris.data
y = iris.target
target_names = iris.target_names
X_r = pca.fit(X).transform(X)
print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))
plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=2,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.show()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2,store_covariance=True)
X_r2 = lda.fit(X, y).transform(X)
print('explained variance ratio (first two components): %s'% str(lda.explained_variance_ratio_))
plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,lw=2,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')
plt.show()
lda.covariance_
from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis(n_components=2)
X_r3 = fa.fit(X, y).transform(X)
X_r3.shape
print(fa.components_)
print(fa.components_[0,:].shape)
fa.components_[0,:].reshape(1,4)@np.cov(X.T)@fa.components_[1,:].reshape(4,1)
plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r3[y == i, 0], X_r3[y == i, 1], alpha=.8, color=color,lw=2,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('FA of IRIS dataset')
plt.show()