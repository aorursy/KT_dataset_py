# system

import os, time, datetime

# data structure

import pandas as pd

import numpy as np



# model

import tensorflow as tf

from tensorflow import keras

from tensorflow.python.keras.utils.data_utils import Sequence

from sklearn.utils import class_weight

from sklearn.preprocessing import LabelEncoder

from sklearn import manifold, datasets



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from matplotlib.ticker import NullFormatter



# utilities

from collections import OrderedDict

from functools import partial

from time import time

root_dir = '../input/lish-moa/'

os.listdir(root_dir)
train_features_dir = root_dir + 'train_features.csv'

train_targets_dir = root_dir + 'train_targets_scored.csv'

test_features_dir = root_dir + 'test_features.csv'

train_features = pd.read_csv(train_features_dir)

train_targets = pd.read_csv(train_targets_dir).drop(columns = 'sig_id')

test_features = pd.read_csv(test_features_dir)

test_id = test_features['sig_id']
def preprocess(df):

    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

    df.loc[:, 'cp_time'] = df.loc[:, 'cp_time'].map({24: 0, 48: 1, 72:2})

    del df['sig_id']

    return df

train_features = preprocess(train_features)

test_features = preprocess(test_features)
def onlyPositive(X,y):

    positive_index = y.eq(1).any(1)

    X = X[y.eq(1).any(1)]

    y = y[y.eq(1).any(1)]

    return X,y
feature_names = list(train_features.columns)

target_names = list(train_targets.columns)
MoA_sum = train_targets.sum().to_frame().reset_index(drop=False).rename(columns={"index": "MoA", 0: "sum"}).sort_values(ascending = False, by= 'sum')



fig, ax = plt.subplots()

plt.barh(MoA_sum.head(20)['MoA'], MoA_sum.head(20)['sum'])

plt.gca().invert_yaxis()

plt.title('The count of MoAs')

plt.show()

MoA_sum.head(20)
n_neighbors = 10

n_components = 3
# Set-up manifold methods

LLE = partial(manifold.LocallyLinearEmbedding,

              n_neighbors, n_components, eigen_solver='auto')



methods = OrderedDict()

# methods['LLE'] = LLE(method='standard')

# methods['LTSA'] = LLE(method='ltsa')

# methods['Hessian LLE'] = LLE(method='hessian')

# methods['Modified LLE'] = LLE(method='modified')

methods['Isomap'] = manifold.Isomap(n_neighbors, n_components)

methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)

methods['SE'] = manifold.SpectralEmbedding(n_components=n_components,

                                           n_neighbors=n_neighbors)

methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca',

                                 random_state=0)
frac = 0.2

print(train_features.shape, train_targets.shape)

CP = train_features.iloc[:,0:3].sample(frac = frac, random_state = 0)

CP_name = list(CP.columns)

CP = np.array(CP)

X = train_features.drop(columns = CP_name).sample(frac = frac, random_state = 0)

X = np.array(X)

targets = train_targets.sample(frac = frac, random_state = 0)

targets = np.array(targets)

print(X.shape, targets.shape)
# calculate results

Y = np.empty((len(methods), X.shape[0], 3), dtype=float)

t = np.empty((len(methods)), dtype=float)

for i, (label, method) in enumerate(methods.items()):

    t0 = time()

    Y[i,] = method.fit_transform(X)

    t[i] = time() - t0

    print("%s: %.2g sec" % (label, t[i]))

# show the control feature

ctl_train_features_index = train_features.iloc[:,0].sample(frac = frac, random_state = 0) ==1

fig = plt.figure(figsize=(18,6))

for i, (label, method) in enumerate(methods.items()):

    Y_ctl = Y[i,][ctl_train_features_index]

    ax = fig.add_subplot(1,4, 1 + i, projection='3d')

    ax.scatter(Y_ctl[:, 0], Y_ctl[:, 1], Y_ctl[:, 2], alpha = 0.1)

    ax.set_title("%s" % (label))

    ax.xaxis.set_major_formatter(NullFormatter())

    ax.yaxis.set_major_formatter(NullFormatter())

    ax.axis('tight')

#     if i >= 0:

#         break

plt.show()
# Create figure

for j in np.arange(3):

    fig = plt.figure(figsize=(18,6))

    fig.suptitle("Manifold Learning with %i neighbors: %s [index: %s]" % (n_neighbors, CP_name[j], j), fontsize=14)

    for i, (label, method) in enumerate(methods.items()):

        ax = fig.add_subplot(1,4, 1 + i, projection='3d')

        ax.scatter(Y[i,][:, 0], Y[i,][:, 1], Y[i,][:, 2], c=CP[:,j], cmap=plt.cm.brg, alpha = 0.1)

        ax.set_title("%s" % (label))

        ax.xaxis.set_major_formatter(NullFormatter())

        ax.yaxis.set_major_formatter(NullFormatter())

        ax.axis('tight')

#     if i >= 0:

#         break

    plt.show()
# droping CP col

train_features, train_targets = onlyPositive(train_features, train_targets)

print(train_features.shape, train_targets.shape)
frac = 0.3



X = train_features.sample(frac = frac, random_state = 0)

X = np.array(X)

targets = train_targets.sample(frac = frac, random_state = 0)

targets = np.array(targets)

print(X.shape, targets.shape)
# calculate results

Y = np.empty((len(methods), X.shape[0], 3), dtype=float)

t = np.empty((len(methods)), dtype=float)

for i, (label, method) in enumerate(methods.items()):

    t0 = time()

    Y[i,] = method.fit_transform(X)

    t[i] = time() - t0

    print("%s: %.2g sec" % (label, t[i]))

# Create figure

# n_show = 10

# n = 0

for j in np.array(MoA_sum.reset_index(drop=False)['index']):

    fig = plt.figure(figsize=(18,6))

    fig.suptitle("Manifold Learning with %i neighbors: %s [index: %s]" % (n_neighbors, target_names[j], j), fontsize=14)

    for i, (label, method) in enumerate(methods.items()):



        # 2d plot

#         ax = fig.add_subplot(1,4, 1 + i)

#         ax.scatter(Y[i,][:, 0], Y[i,][:, 1], c=targets[:,j], cmap=plt.cm.Spectral)

        # 3d plot

        ax = fig.add_subplot(1,4, 1 + i, projection='3d')

        ax.scatter(Y[i,][:, 0], Y[i,][:, 1], Y[i,][:, 2], c=targets[:,j], cmap=plt.cm.binary, alpha = 0.1)

        ax.set_title("%s" % (label))

        ax.xaxis.set_major_formatter(NullFormatter())

        ax.yaxis.set_major_formatter(NullFormatter())

        ax.axis('tight')

#     if i >= 0:

#         break

#     n += 1

#     if n >= n_show:

#         break

    plt.show()