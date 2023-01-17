import numpy as np

import pandas as pd

from sklearn.manifold import TSNE
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
train = train[test.columns].values

test = test[test.columns].values
train_test = np.vstack([train, test])

train_test.shape
%%time

tsne = TSNE(n_components=2)

train_test_2D = tsne.fit_transform(train_test)
train_2D = train_test_2D[:train.shape[0]]

test_2D = train_test_2D[train.shape[0]:]
np.save('train_2D', train_2D)

np.save('test_2D', test_2D)