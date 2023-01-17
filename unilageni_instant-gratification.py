# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn import svm, neighbors, linear_model

train, test = pd.read_csv('../input/train.csv'), pd.read_csv('../input/test.csv')

col = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

knn = neighbors.KNeighborsClassifier(n_neighbors=17, p=2.9)

svnu = svm.NuSVC(probability=True, kernel='poly', degree=4, gamma='scale', random_state=42, nu=0.6, coef0=0.08)

lr = linear_model.LogisticRegression(solver='liblinear',penalty='l1',C=0.05,n_jobs=-1)

svc = svm.SVC(probability=True, kernel='poly', degree=4, gamma='scale', random_state=42)

lr = linear_model.LogisticRegression(solver='liblinear',penalty='l1',C=0.05,n_jobs=-1)

for m in sorted(train['wheezy-copper-turtle-magic'].unique()):

    idx_tr, idx_te  = (train['wheezy-copper-turtle-magic']==m), (test['wheezy-copper-turtle-magic']==m)

    svc.fit(train[idx_tr][col], train[idx_tr]['target']); knn.fit(train[idx_tr][col], train[idx_tr]['target']); svnu.fit(train[idx_tr][col], train[idx_tr]['target']); lr.fit(train[idx_tr][col], train[idx_tr]['target'])

    test.loc[idx_te,'target'] = 0.75*svnu.predict_proba(test[idx_te][col])[:,1]+ 0.06*svc.predict_proba(test[idx_te][col])[:,1] + 0.14*knn.predict_proba(test[idx_te][col])[:,1]+ 0.05*lr.predict_proba(test[idx_te][col])[:,1]

test[['id','target']].to_csv("submission.csv", index=False)
