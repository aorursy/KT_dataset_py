# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%%time

# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)

import sys

!cp ../input/rapids/rapids.0.12.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz

sys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_auc_score

import cudf, cuml

import cupy as cp

from cuml.linear_model import LogisticRegression
train = pd.read_csv('../input/digit-recognizer/train.csv')

submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
train.head()
submission.head()
y = train['label'].values
y
train_tsne = np.load('../input/mnist-2d-t-sne-with-rapids/train_2D.npy')

test_tsne = np.load('../input/mnist-2d-t-sne-with-rapids/test_2D.npy')
train_x, val_x, train_y, val_y = train_test_split(train_tsne, y, test_size=0.10)
clf = LogisticRegression(C = 0.1)

clf.fit(train_x, train_y.astype('float32'))
preds = clf.predict(val_x)
np.mean(cp.array(val_y) == preds.values.astype('int64'))
train_umap = np.load('../input/mnist-2d-umap-with-rapids/train_2D.npy')

test_umap = np.load('../input/mnist-2d-umap-with-rapids/test_2D.npy')
train_x, val_x, train_y, val_y = train_test_split(train_umap, y, test_size=0.10)
clf = LogisticRegression(C = 12)

clf.fit(train_x, train_y.astype('float64'))

preds = clf.predict(val_x)

np.mean(cp.array(val_y) == preds.values.astype('int64'))
test_preds = clf.predict(test_umap)
train_y.astype('float32')
train_both = np.hstack([train_umap, train_tsne])

test_both = np.hstack([test_umap, test_tsne])
train_x, val_x, train_y, val_y = train_test_split(train_both, y, test_size=0.10)
clf = LogisticRegression(C = 1)

clf.fit(train_x, train_y.astype('float64'))

preds = clf.predict(val_x)

np.mean(cp.array(val_y) == preds.values.astype('int64'))
#test_preds = clf.predict(test_both)
cp.asnumpy(test_preds.values.astype('int64'))
submission['Label'] = cp.asnumpy(test_preds.values.astype('int64'))
submission.to_csv('submission.csv', index=False)
submission.head()