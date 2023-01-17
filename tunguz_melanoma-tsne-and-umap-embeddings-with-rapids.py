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
train_df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")

train = np.load('../input/siimisic-melanoma-resized-images/x_train_32.npy')
train_df.head()
train = train.reshape((train.shape[0], 32*32*3))

train.shape
%%time

tsne = TSNE(n_components=2)

train_2D = tsne.fit_transform(train)
plt.scatter(train_2D[:,0], train_2D[:,1], c=train_df['target'].values, s = 0.8)

%%time

umap = UMAP(n_components=2)

train_2D = umap.fit_transform(train)
plt.scatter(train_2D[:,0], train_2D[:,1], c=train_df['target'].values, s = 0.8)
