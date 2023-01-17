import pandas

import numpy as np

from sklearn import preprocessing

df = pandas.read_csv('../input/001-customer.csv',

                     index_col=False, header=0);

Group    = df["Group"].values

Features = df[["F1","F2","F3","F4","F5"]].values

Features = preprocessing.scale(Features)
from sklearn.svm import SVC,LinearSVC,NuSVC



clf = SVC(C=0.01, kernel='linear')

clf.fit(Features, Group)

clf.decision_function(Features)

clf.support_vectors_.shape

clf.coef_



predictions = clf.predict(Features)

np.sum(predictions==Group)
clx = NuSVC(nu=0.80, kernel='linear')

clx.fit(Features, Group)

clx.decision_function(Features)

clx.support_vectors_.shape

clx.coef_



predictions = clx.predict(Features)

np.sum(predictions==Group)
cls = LinearSVC(penalty="l1",dual=False)

cls.fit(Features,Group)

predictions = cls.predict(Features)

np.sum(predictions==Group)