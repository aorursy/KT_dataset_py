import numpy as np

import pandas as pd

from sklearn.manifold import TSNE

%matplotlib inline

import matplotlib.pyplot as plt

import gc# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
X_train = np.load('/kaggle/input/distilbert-use-features-just-the-features/X_train.npy')

X_test = np.load('/kaggle/input/distilbert-use-features-just-the-features/X_test.npy')
train_test = np.vstack([X_train, X_test])
train_test.shape
del X_train, X_test

gc.collect()
%%time

train_test = TSNE(n_components=2).fit_transform(train_test)
train_test.shape
plt.scatter(x= train_test[:,0], y= train_test[:,1])