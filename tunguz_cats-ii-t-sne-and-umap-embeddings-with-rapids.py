%%time

# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)

import sys

!cp ../input/rapids/rapids.0.12.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz

sys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/




# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cudf, cuml

from cuml.manifold import UMAP, TSNE

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/multi-cat-encodings/X_train_te.csv')

test = pd.read_csv('../input/multi-cat-encodings/X_test_te.csv')

sample_submission = pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv')

y = np.load('/kaggle/input/multi-cat-encodings/target.npy')

features = test.columns
train = train[features].values

test = test[features].values

train_test = np.vstack([train, test])

train_test.shape
%%time

tsne = TSNE(n_components=2)

train_test_2D = tsne.fit_transform(train_test)
train_2D = train_test_2D[:train.shape[0]]

test_2D = train_test_2D[train.shape[0]:]



np.save('train_tsne_2D', train_2D)

np.save('test_tsne_2D', test_2D)
plt.scatter(train_2D[:,0], train_2D[:,1], c = y, s = 0.5)

%%time

umap = UMAP(n_components=2)

train_test_2D = umap.fit_transform(train_test)
train_2D = train_test_2D[:train.shape[0]]

test_2D = train_test_2D[train.shape[0]:]



np.save('train_umap_2D', train_2D)

np.save('test_umap_2D', test_2D)
plt.scatter(train_2D[:,0], train_2D[:,1], c = y, s = 0.5)
