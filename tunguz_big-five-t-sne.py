%%time

# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)

import sys

!cp ../input/rapids/rapids.0.12.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz

sys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import cudf, cuml

import cupy as cp

import numpy as np

import pandas as pd

from cuml.manifold import TSNE

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv', sep='\t')
X_train = train[train.columns[:50]]

X_train.head()
X_train.shape
X_train = X_train.dropna()

X_train.shape
X_train = X_train.values/5
%%time

tsne = TSNE(n_components=2)

X_train_2D = tsne.fit_transform(X_train)
# Plot the embedding

plt.scatter(X_train_2D[:,0], X_train_2D[:,1], s = 0.5)