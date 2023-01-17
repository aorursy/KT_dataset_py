%%time

# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)

import sys

!cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz

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
%%time

embeddings = pd.read_csv('../input/rxrx19a/embeddings.csv')
metadata = pd.read_csv('../input/rxrx19a/metadata.csv')

metadata.head()
metadata['disease_condition'].unique()
embeddings = embeddings[embeddings.columns[1:]].values
%%time

tsne = TSNE(n_components=2)

embeddings_2D = tsne.fit_transform(embeddings)
plt.scatter(embeddings_2D[:,0], embeddings_2D[:,1])
