import numpy as np

import pandas as pd

from sklearn import datasets

from sklearn import svm

from sklearn.model_selection import train_test_split
X, y = datasets.load_iris(return_X_y=True)

X.shape, y.shape
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)

# splitting the data, fitting the model and computing 

# the score 5 consecutive times(with diff splits each time)

scores = cross_val_score(clf, X, y, cv=5)
scores
# change the score parameters in cross_val_score

scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
scores
from sklearn.model_selection import ShuffleSplit
n_samples = X.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

cross_val_score(clf, X, y, cv=cv)