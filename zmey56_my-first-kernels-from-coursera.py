# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/from-coursera"))

# Any results you write to the current directory are saved as output.
import math
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/from-coursera/gbm-data.csv')
df.tail()
df_arr = df.values
y = df_arr[:,0]
X = df_arr[:,1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)
learning_rates = [1, 0.5, 0.3, 0.2, 0.1]
def sigmoid(y_pred):
    return 1.0 / (1.0 + np.exp(-y_pred))
# import machine learning algorithms
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


global_learning_rate = 0
global_iter_num = 0
global_min_log = 1000

for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=250, learning_rate = learning_rate, verbose=True, random_state = 241)
    gb.fit(X_train, y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_test, y_test)))
    # compute test set deviance
    result = []
    iter_num = 0
    min_log = 0
    
    for i, y_pred in enumerate(gb.staged_decision_function(X_test)):
        # clf.loss_ assumes that y_test[i] in {0, 1}
        tmp = log_loss(y_test, sigmoid(y_pred))
        if i == 1:
            min_log = tmp
            iter_num = 1
        if tmp < min_log:
            min_log = tmp
            iter_num = i
        result.append(tmp)
    if global_min_log > min_log:
        global_min_log = min_log
        global_learning_rate = learning_rate
        global_iter_num = iter_num
    plt.plot(result)
    plt.show()
    
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=250, learning_rate = learning_rate, verbose=True, random_state = 241)
    gb.fit(X_train, y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_test, y_test)))
    # compute test set deviance
    train_loss = []
    test_loss = []
    
    for i, y_pred in enumerate(gb.staged_decision_function(X_test)):
        # clf.loss_ assumes that y_test[i] in {0, 1}
        tmp = log_loss(y_test, sigmoid(y_pred))
        test_loss.append(tmp)
    
    for i, y_pred in enumerate(gb.staged_decision_function(X_train)):
        # clf.loss_ assumes that y_test[i] in {0, 1}
        tmp = log_loss(y_train, sigmoid(y_pred))
        train_loss.append(tmp)
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()
print(global_learning_rate, global_iter_num, global_min_log)
print(1, 'overfitting')
gb = GradientBoostingClassifier(n_estimators=250, learning_rate = 0.2, verbose=True, random_state = 241)
gb.fit(X_train, y_train)
print("Learning rate: ", learning_rate)
print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(gb.score(X_test, y_test)))
# compute test set deviance
iter_num = 0
min_log = 0
    
for i, y_pred in enumerate(gb.staged_decision_function(X_test)):
    # clf.loss_ assumes that y_test[i] in {0, 1}
    tmp = log_loss(y_test, sigmoid(y_pred))
    if i == 1:
        min_log = tmp
        iter_num = 1
    if tmp < min_log:
        min_log = tmp
        iter_num = i
print(iter_num, min_log)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
clf = RandomForestClassifier(n_estimators=51, random_state=241)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
tmp = log_loss(y_test, y_pred)
print(tmp)
clf = RandomForestClassifier(n_estimators=36, random_state=241)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
tmp = log_loss(y_test, y_pred)
print(tmp)
