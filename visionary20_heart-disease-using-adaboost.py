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
import sklearn

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.datasets import load_breast_cancer

from sklearn.datasets import load_digits

from sklearn import metrics

%matplotlib inline



import os

import warnings

warnings.filterwarnings('ignore')

##Importing Data

df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.head()
df.info()
df.describe()
df['target'].value_counts()
# adaboost experiments

# create x and y train



X = df.drop('target', axis=1)

y = df[['target']]



# split data into train and test/validation sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
# check the average cancer occurence rates in train and test data, should be comparable

print(y_train.mean())

print(y_test.mean())
# base estimator: a weak learner with max_depth=2

shallow_tree = DecisionTreeClassifier(max_depth=1, random_state = 100)
# fit the shallow decision tree 

shallow_tree.fit(X_train, y_train)



# test error

y_pred = shallow_tree.predict(X_test)

score = metrics.accuracy_score(y_test, y_pred)

score
# adaboost with the tree as base estimator



estimators = list(range(20,25))



abc_scores = []

for n_est in estimators:

    ABC = AdaBoostClassifier(

    base_estimator=shallow_tree, 

    n_estimators = n_est)

    

    ABC.fit(X_train, y_train)

    y_pred = ABC.predict(X_test)

    score = metrics.accuracy_score(y_test, y_pred)

    abc_scores.append(score)

    
abc_scores
# plot test scores and n_estimators

# plot

plt.plot(estimators, abc_scores)

plt.xlabel('n_estimators')

plt.ylabel('accuracy')

plt.ylim([0.70, 1])

plt.show()
ABC = AdaBoostClassifier(

    base_estimator=shallow_tree, 

    n_estimators = 22)

    

ABC.fit(X_train, y_train)

y_pred = ABC.predict(X_test)
y_pred
np.array(y_test).T
# Importing classification report and confusion matrix from sklearn metrics

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
# Let's check the report of our default model

print(classification_report(y_test,y_pred))
# Printing confusion matrix

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))