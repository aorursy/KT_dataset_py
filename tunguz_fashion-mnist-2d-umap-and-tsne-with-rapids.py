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

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cudf, cuml

from cuml.manifold import UMAP, TSNE

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
train_test = np.vstack([train, test])

train_test.shape
y = train_test[:,0]

train_test = train_test[:,1:]
%%time

tsne = TSNE(n_components=2)

train_test_2D = tsne.fit_transform(train_test)
plt.scatter(train_test_2D[:,0], train_test_2D[:,1], c = y, s = 0.5)

%%time

umap = UMAP(n_components=2)

train_test_2D = umap.fit_transform(train_test)
plt.scatter(train_test_2D[:,0], train_test_2D[:,1], c = y, s = 0.5)
