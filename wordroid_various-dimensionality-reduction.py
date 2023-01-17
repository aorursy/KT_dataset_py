# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import os.path
import sys
import re
import itertools
import csv
import datetime, time
import pickle
import random
from collections import defaultdict, Counter
import gc

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import seaborn as sns
import scipy
import gensim
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, cmap=cmap, **kwargs)
def scatter(x, y, color, **kwargs):
    plt.scatter(x, y, marker='.')
NN_word = 2000
NN_sentence = 10000
NN_SEG = 7
product_list = [ee+1 for ee in range(NN_word)]
product_set = set(product_list)
nn_each_group, amari = divmod(len(product_list), NN_SEG)
if amari != 0:
    nn_each_group += 1
del_none = lambda l: filter(lambda x: x is not None, l)
product_group = [[e1 for e1 in del_none(ee)] for ee in itertools.zip_longest(*[iter(product_list)]*nn_each_group)]

user_list = [ee+1 for ee in range(NN_sentence)]
random.seed(0)

a, _ = divmod(len(user_list), NN_SEG)
X_list = [np.zeros((1,NN_word+1))]

for iuser in user_list:
    nword = random.randint(5, 20)
    ii = int(iuser / (a+1))
    prods = random.sample(product_group[ii], nword)
    irow = np.zeros((1,NN_word+1))
    irow[0,prods] = 1
    X_list.append(irow)

X = np.concatenate(X_list)
X = X[1:,1:]
print(X.shape)
X
a, _ = divmod(NN_sentence, NN_SEG)
cl = ['c'+str(int(user_id / (a+1))) for user_id in range(1, 1+NN_sentence)]
X_df = pd.DataFrame(X, dtype=int)
X_df.index = ['r'+ee.astype('str') for ee in (np.arange(X_df.shape[0])+1)]
X_df.columns = ['c'+ee.astype('str') for ee in np.arange(X_df.shape[1])+1]
X_df['cls'] = cl
print(X_df.shape)
X_df.head()
plt.figure(figsize=(10, 10))
plt.imshow(X.T)
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
sns.set_context('paper')
#----------------------------------------------------------------------
# Projection on to the first 5 principal components

print("Computing PCA projection")
t0 = time.time()
X_pca = decomposition.TruncatedSVD(n_components=5).fit_transform(X)
print(time.time() - t0)
df = pd.DataFrame(X_pca[:,:5])
df['cl'] = cl
sns.pairplot(df, markers='.', hue='cl', size=2.5)
#----------------------------------------------------------------------
# Isomap projection 
n_neighbors = 30

print("Computing Isomap embedding")
t0 = time.time()
X_iso = manifold.Isomap(n_neighbors, n_components=5, n_jobs=-1).fit_transform(X)
print("Done.")
print(time.time() - t0)
df = pd.DataFrame(X_iso[:,:5])
df['cl'] = cl
sns.pairplot(df, markers='.', hue='cl', size=2.5)
sns.set_context('paper')

g = sns.PairGrid(df, size=2.5)
g.map_diag(plt.hist, edgecolor="w")
g.map_lower(scatter)
g.map_upper(hexbin)
n_neighbors = 30
#----------------------------------------------------------------------
# Locally linear embedding
print("Computing LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=5,
                                      method='standard')
t0 = time.time()
X_lle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
print(time.time() - t0)
df = pd.DataFrame(X_lle[:,:5])
df['cl'] = cl
sns.pairplot(df, markers='.', hue='cl', size=2.5)
df = pd.DataFrame(X_lle[:,:5])
sns.set_context('paper')
g = sns.PairGrid(df, size=2.5)
g.map_diag(plt.hist, edgecolor="w")
g.map_lower(scatter)
g.map_upper(hexbin)
# n_neighbors = 561
# #----------------------------------------------------------------------
# # HLLE embedding
# print("Computing Hessian LLE embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=32,
#                                       method='hessian')
# t0 = time.time()
# X_hlle = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# print(time.time() - t0)
# df = pd.DataFrame(X_hlle[:,:5])
# df['cl'] = cl
# sns.pairplot(df, markers='.', hue='cl', size=2.5)
# df = pd.DataFrame(X_hlle[:,:5])
# sns.set_context('paper')
# g = sns.PairGrid(df, size=2.5)
# g.map_diag(plt.hist, edgecolor="w")
# g.map_lower(scatter)
# g.map_upper(hexbin)
# n_neighbors = 561
# #----------------------------------------------------------------------
# # LTSA embedding
# print("Computing LTSA embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=32,
#                                       method='ltsa')
# t0 = time.time()
# X_ltsa = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# print(time.time() - t0)
# df = pd.DataFrame(X_ltsa[:,:5])
# df['cl'] = cl
# sns.pairplot(df, markers='.', hue='cl', size=2.5)
# df = pd.DataFrame(X_ltsa[:,:5])
# sns.set_context('paper')

# g = sns.PairGrid(df, size=2.5)
# g.map_diag(plt.hist, edgecolor="w")
# g.map_lower(scatter)
# g.map_upper(hexbin)
#----------------------------------------------------------------------
# MDS embedding
print("Computing MDS embedding")
clf = manifold.MDS(n_components=5, n_init=1, max_iter=100)
t0 = time.time()
X_mds = clf.fit_transform(X)
print("Done. Stress: %f" % clf.stress_)
print(time.time() - t0)
df = pd.DataFrame(X_mds[:,:5])
df['cl'] = cl
sns.pairplot(df, markers='.', hue='cl', size=2.5)
df = pd.DataFrame(X_mds[:,:5])
sns.set_context('paper')

g = sns.PairGrid(df, size=2.5)
g.map_diag(plt.hist, edgecolor="w")
g.map_lower(scatter)
g.map_upper(hexbin)
#----------------------------------------------------------------------
# Random Trees embedding
print("Computing Totally Random Trees embedding")
hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
                                       max_depth=5)
t0 = time.time()
X_transformed = hasher.fit_transform(X)
pca = decomposition.TruncatedSVD(n_components=5)
X_reduced = pca.fit_transform(X_transformed)
print(time.time() - t0)
df = pd.DataFrame(X_reduced[:,:5])
df['cl'] = cl
sns.pairplot(df, markers='.', hue='cl', size=2.5)
df = pd.DataFrame(X_reduced[:,:5])
sns.set_context('paper')

g = sns.PairGrid(df, size=2.5)
g.map_diag(plt.hist, edgecolor="w")
g.map_lower(scatter)
g.map_upper(hexbin)
#----------------------------------------------------------------------
# Spectral embedding
print("Computing Spectral embedding")
embedder = manifold.SpectralEmbedding(n_components=5, random_state=0,
                                      eigen_solver="arpack")
t0 = time.time()
X_se = embedder.fit_transform(X)
print(time.time() - t0)
df = pd.DataFrame(X_se[:,:5])
df['cl'] = cl
sns.pairplot(df, markers='.', hue='cl', size=2.5)
df = pd.DataFrame(X_se[:,:5])
sns.set_context('paper')

g = sns.PairGrid(df, size=2.5)
g.map_diag(plt.hist, edgecolor="w")
g.map_lower(scatter)
g.map_upper(hexbin)
#----------------------------------------------------------------------
# t-SNE embedding
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
t0 = time.time()
X_tsne = tsne.fit_transform(X)
print(time.time() - t0)
df = pd.DataFrame(X_tsne[:,:5])
df['cl'] = cl
sns.pairplot(df, markers='.', hue='cl', size=2.5)
df = pd.DataFrame(X_tsne)
sns.set_context('paper')

g = sns.PairGrid(df, size=2.5)
g.map_diag(plt.hist, edgecolor="w")
g.map_lower(scatter)
g.map_upper(hexbin)
