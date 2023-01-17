import numpy as np

import pandas as pd

from umap import UMAP

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
train_2D = train_test_2D[:train.shape[0]]

test_2D = train_test_2D[train.shape[0]:]
np.save('train_2D', train_2D)

np.save('test_2D', test_2D)
plt.scatter(train_2D[:,0], train_2D[:,1], c = y, s = 0.5)