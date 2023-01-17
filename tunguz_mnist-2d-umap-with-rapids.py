%%time

# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)

import sys

!cp ../input/rapids/rapids.0.11.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz

sys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import cudf, cuml

import pandas as pd

import numpy as np

from cuml.manifold import UMAP

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
y = train['label'].values

train = train[test.columns].values

test = test[test.columns].values
train_test = np.vstack([train, test])

train_test.shape
%%time

umap = UMAP()

train_test_2D = umap.fit_transform(train_test)
%%time

umap = UMAP()

train_2D = umap.fit_transform(train)
plt.scatter(train_2D[:,0], train_2D[:,1], c = y, s = 0.5)
train_2D = train_test_2D[:train.shape[0]]

test_2D = train_test_2D[train.shape[0]:]



np.save('train_2D', train_2D)

np.save('test_2D', test_2D)