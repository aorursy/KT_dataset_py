%%time

# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)

import sys

!cp ../input/rapids/rapids.0.11.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz

sys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import cudf, cuml

import cupy as cp

import numpy as np

import pandas as pd

from cuml.manifold import TSNE

import matplotlib.pyplot as plt

from matplotlib.pyplot import ylim, xlim

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/cattells-16-personality-factors/16PF/data.csv", sep="\t")
data.head()
gendered_data = data[(data['gender'] == 1) | (data['gender'] == 2)]
gendered_data['gender'] = gendered_data['gender'].values -1
features = gendered_data.columns[:-6]

gendered_data[features] = gendered_data[features].values/5.

gendered_data['std'] = gendered_data[features].std(axis=1)

gendered_data = gendered_data[gendered_data['std'] > 0.0]

X = gendered_data[features].values

Y = gendered_data['gender'].values
%%time

tsne = TSNE(n_components=2)

train_2D = tsne.fit_transform(X)
ylim(-150, 150)

xlim(-150, 150)
plt.scatter(train_2D[:,0], train_2D[:,1], c = Y, s = 0.5)