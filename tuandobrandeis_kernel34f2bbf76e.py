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
import numpy as np 

import pandas as pd 

import xgboost as xgb

from sklearn import preprocessing

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

import seaborn as sns

sns.set(style="white") #white background style for seaborn plots

sns.set(style="whitegrid", color_codes=True)



data = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
from sklearn.model_selection import train_test_split
X = data[list(data.columns)[2:]]

Y = data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=7)
X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, test_size = 0.25, random_state=7)
print (X_train.shape)

print (X_validate.shape)

print (X_test.shape)

print (Y_train.shape)

print (Y_validate.shape)

print (Y_test.shape)
X_train, X_train_2, Y_train, Y_train_2 = train_test_split(X_train, Y_train, test_size = 0.5, random_state=7)
print (X_train.shape)

print (X_validate.shape)

print (X_test.shape)

print (Y_train.shape)

print (Y_validate.shape)

print (Y_test.shape)
from sklearn.metrics import f1_score
# from sklearn.tree import DecisionTreeClassifier

# decision_tree_gini = DecisionTreeClassifier(criterion = 'gini')

# decision_tree_gini.fit(X_train, Y_train)

# Y_validate_predicted = decision_tree_gini.predict(X_validate)

# f1_score(Y_validate, Y_validate_predicted)

# Y_train_predicted = decision_tree_gini.predict(X_train)

# f1_score(Y_train, Y_train_predicted)



# import time

# for k in range(5, 50, 5):

#     start_time = time.time()

#     print ('Max depth = %d' % k)

#     decision_tree_gini_depth = DecisionTreeClassifier(criterion = 'gini', max_depth = k)

#     decision_tree_gini_depth.fit(X_train, Y_train)

    

#     Y_train_predicted = decision_tree_gini_depth.predict(X_train)

#     Y_validate_predicted = decision_tree_gini_depth.predict(X_validate)

    

#     print ('F1 score on training data %.3f' % f1_score(Y_train, Y_train_predicted))

#     print ('F1 score on validate data %.3f' % f1_score(Y_validate, Y_validate_predicted))

    

#     print ('Time to train model is %d seconds' % (time.time() - start_time))
from sklearn.ensemble import BaggingClassifier
bagging_classifier = BaggingClassifier(n_estimators=5, max_samples=0.3)
import time

start_time = time.time()

bagging_classifier.fit(X_train, Y_train)



print ('Time to train model is %d seconds' % (time.time() - start_time))



Y_train_predicted = bagging_classifier.predict(X_train)

Y_validate_predicted = bagging_classifier.predict(X_validate)



print ('F1 score on training data %.3f' % f1_score(Y_train, Y_train_predicted))

print ('F1 score on validate data %.3f' % f1_score(Y_validate, Y_validate_predicted))
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=30, max_features=15, class_weight = 'balanced')
start_time = time.time()

rf_classifier.fit(X_train, Y_train)



print ('Time to train model is %d seconds' % (time.time() - start_time))



Y_train_predicted = rf_classifier.predict(X_train)

Y_validate_predicted = rf_classifier.predict(X_validate)



print ('F1 score on training data %.3f' % f1_score(Y_train, Y_train_predicted))

print ('F1 score on validate data %.3f' % f1_score(Y_validate, Y_validate_predicted))
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier



adaboost_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth = 2))
start_time = time.time()

adaboost_classifier.fit(X_train, Y_train)



print ('Time to train model is %d seconds' % (time.time() - start_time))



Y_train_predicted = adaboost_classifier.predict(X_train)

Y_validate_predicted = adaboost_classifier.predict(X_validate)



print ('F1 score on training data %.3f' % f1_score(Y_train, Y_train_predicted))

print ('F1 score on validate data %.3f' % f1_score(Y_validate, Y_validate_predicted))
adaboost_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth = 5))



start_time = time.time()

adaboost_classifier.fit(X_train, Y_train)



print ('Time to train model is %d seconds' % (time.time() - start_time))



Y_train_predicted = adaboost_classifier.predict(X_train)

Y_validate_predicted = adaboost_classifier.predict(X_validate)



print ('F1 score on training data %.3f' % f1_score(Y_train, Y_train_predicted))

print ('F1 score on validate data %.3f' % f1_score(Y_validate, Y_validate_predicted))