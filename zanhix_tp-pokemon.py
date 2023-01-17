# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os





X=np.zeros((800,10))

Y=np.zeros(800)



max_f = 0

h2float={}

for i,l in enumerate(open('../input/Pokemon.csv')):

    if(i==0):

        continue

    t_l = l.rstrip('\r\n').split(',')

    for j,c in enumerate(t_l[2:-1]):

        if (j==0 or j==1):

            if c not in h2float:

                max_f += 1

                h2float[c]=max_f

            X[i-1,j]=h2float[c]

        else: X[i-1,j]=float(c)

    Y[i-1]= 0. if t_l[-1]=='False' else 1.

X_train, X_test = X[:200], X[200:]

Y_train, Y_test = Y[:200], Y[200:]

clf = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(clf, X, Y, cv=10)

print(scores.mean())





#clf = clf.fit(X_train, Y_train)

#print(clf.score(X_test,Y_test))   



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
#solver newton-cg score = 0.915

#solver liblinear  score = 0.903



clf1 = LogisticRegression(solver='newton-cg').fit(X_train,Y_train)





scores = clf1.score(X_test,Y_test)

print(scores)



clf2 = LogisticRegression(solver='newton-cg')

scores = cross_val_score(clf2, X, Y, cv=10)

print(scores.mean())
clf3 = GaussianNB().fit(X_train,Y_train)

print(clf3.score(X_test,Y_test))