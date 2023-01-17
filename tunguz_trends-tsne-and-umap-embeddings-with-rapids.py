%%time

# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)

import sys

!cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz

sys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import cudf, cuml

import cupy as cp

import numpy as np

import pandas as pd

import os

from cuml.manifold import TSNE, UMAP

import matplotlib.pyplot as plt

from matplotlib.pyplot import ylim, xlim

%matplotlib inline

fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")

loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")



fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])

df = fnc_df.merge(loading_df, on="Id")





labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")

labels_df["is_train"] = True



df = df.merge(labels_df, on="Id", how="left")



test_df = df[df["is_train"] != True].copy()

df = df[df["is_train"] == True].copy()



df.shape, test_df.shape
# Giving less importance to FNC features since they are easier to overfit due to high dimensionality.

FNC_SCALE = 1/600



df[fnc_features] *= FNC_SCALE

test_df[fnc_features] *= FNC_SCALE



features = loading_features + fnc_features
train_test = np.vstack([df[features], test_df[features]])

train_test.shape
%%time

tsne = TSNE(n_components=2)

train_test_2D = tsne.fit_transform(train_test)
plt.scatter(train_test_2D[:,0], train_test_2D[:,1], s = 0.5)



%%time

tsne = TSNE(n_components=2)



train_2D = tsne.fit_transform(df[features].values)
plt.scatter(train_2D[:,0], train_2D[:,1], s = 0.5)
df['age'].values
plt.scatter(train_2D[:,0], train_2D[:,1], c=df['age'].values, s = 0.5)
plt.scatter(train_2D[:,0], train_2D[:,1], c=df['domain1_var1'].values, s = 0.5)
plt.scatter(train_2D[:,0], train_2D[:,1], c=df['domain1_var2'].values, s = 0.5)
%%time

umap = UMAP(n_components=2)

train_test_2D = umap.fit_transform(train_test)
plt.scatter(train_test_2D[:,0], train_test_2D[:,1], s = 0.5)



%%time

umap = UMAP(n_components=2)



train_2D = umap.fit_transform(df[features].values)
plt.scatter(train_2D[:,0], train_2D[:,1], s = 0.5)
plt.scatter(train_2D[:,0], train_2D[:,1], c=df['age'].values, s = 0.5)