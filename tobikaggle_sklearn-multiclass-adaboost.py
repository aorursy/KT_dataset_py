# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
print(__doc__)



# Author: Noel Dawe <noel.dawe@gmail.com>

#

# License: BSD 3 clause



from sklearn.externals.six.moves import zip



import matplotlib.pyplot as plt



from sklearn.datasets import make_gaussian_quantiles

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier





X, y = make_gaussian_quantiles(n_samples=13000, n_features=10,

                               n_classes=3, random_state=1)



n_split = 3000



X_train, X_test = X[:n_split], X[n_split:]

y_train, y_test = y[:n_split], y[n_split:]



bdt_real = AdaBoostClassifier(

    DecisionTreeClassifier(max_depth=2),

    n_estimators=600,

    learning_rate=1)



bdt_discrete = AdaBoostClassifier(

    DecisionTreeClassifier(max_depth=2),

    n_estimators=600,

    learning_rate=1.5,

    algorithm="SAMME")



bdt_real.fit(X_train, y_train)

bdt_discrete.fit(X_train, y_train)



real_test_errors = []

discrete_test_errors = []



for real_test_predict, discrete_train_predict in zip(

        bdt_real.staged_predict(X_test), bdt_discrete.staged_predict(X_test)):

    real_test_errors.append(

        1. - accuracy_score(real_test_predict, y_test))

    discrete_test_errors.append(

        1. - accuracy_score(discrete_train_predict, y_test))



n_trees_discrete = len(bdt_discrete)

n_trees_real = len(bdt_real)



# Boosting might terminate early, but the following arrays are always

# n_estimators long. We crop them to the actual number of trees here:

discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]

real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]

discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]



plt.figure(figsize=(15, 5))



plt.subplot(131)

plt.plot(range(1, n_trees_discrete + 1),

         discrete_test_errors, c='black', label='SAMME')

plt.plot(range(1, n_trees_real + 1),

         real_test_errors, c='black',

         linestyle='dashed', label='SAMME.R')

plt.legend()

plt.ylim(0.18, 0.62)

plt.ylabel('Test Error')

plt.xlabel('Number of Trees')



plt.subplot(132)

plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,

         "b", label='SAMME', alpha=.5)

plt.plot(range(1, n_trees_real + 1), real_estimator_errors,

         "r", label='SAMME.R', alpha=.5)

plt.legend()

plt.ylabel('Error')

plt.xlabel('Number of Trees')

plt.ylim((.2,

         max(real_estimator_errors.max(),

             discrete_estimator_errors.max()) * 1.2))

plt.xlim((-20, len(bdt_discrete) + 20))



plt.subplot(133)

plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights,

         "b", label='SAMME')

plt.legend()

plt.ylabel('Weight')

plt.xlabel('Number of Trees')

plt.ylim((0, discrete_estimator_weights.max() * 1.2))

plt.xlim((-20, n_trees_discrete + 20))



# prevent overlapping y-axis labels

plt.subplots_adjust(wspace=0.25)

plt.show()