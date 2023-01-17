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
test = pd.read_csv('../input/test_V2.csv')

train = pd.read_csv('../input/train_V2.csv')
# np.array(train)[:,12]

x_train = train.iloc[:,3:-1].values

y_train = train.iloc[:,-1].values

x_test = test.iloc[:,3:].values
x_test[12]
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X1 = LabelEncoder()

x_train[:,12] = labelencoder_X1.fit_transform(x_train[:,12])

x_test[:,12] = labelencoder_X1.fit_transform(x_test[:,12])



# onehotencoder = OneHotEncoder(categorical_features = [12])

# x_train = onehotencoder.fit_transform(x_train).toarray()

# x_test = onehotencoder.fit_transform(x_test).toarray()



y_train = np.nan_to_num(y_train,copy=True)

for i in range(len(y_train)):

    if y_train[i] > 0.5: 

        y_train[i] = 1

    else:

        y_train[i] = 0
import pandas as pd

import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.datasets import make_hastie_10_2

import matplotlib.pyplot as plt



""" HELPER FUNCTION: GET ERROR RATE ========================================="""

def get_error_rate(pred, Y):

    return sum(pred != Y) / float(len(Y))



""" HELPER FUNCTION: PRINT ERROR RATE ======================================="""

def print_error_rate(err):

    print('Error rate: Training: %.4f - Test: %.4f' % err)



""" HELPER FUNCTION: GENERIC CLASSIFIER ====================================="""

def generic_clf(Y_train, X_train, X_test, clf):

    clf.fit(X_train,Y_train)

    pred_train = clf.predict(X_train)

    pred_test = clf.predict(X_test)

    return get_error_rate(pred_train, Y_train) ,pred_test

    

""" ADABOOST IMPLEMENTATION ================================================="""

def adaboost_clf(Y_train, X_train, X_test, M, clf):

    n_train, n_test = len(X_train), len(X_test)

    # Initialize weights

    w = np.ones(n_train) / n_train

    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    

    for i in range(M):

        # Fit a classifier with the specific weights

        clf.fit(X_train, Y_train, sample_weight = w)

        pred_train_i = clf.predict(X_train)

        pred_test_i = clf.predict(X_test)

        # Indicator function

        miss = [int(x) for x in (pred_train_i != Y_train)]

        # Equivalent with 1/-1 to update weights

        miss2 = [x if x==1 else -1 for x in miss]

        # Error

        err_m = np.dot(w,miss) / sum(w)

        # Alpha

        alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))

        # New weights

        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))

        # Add to prediction

        pred_train = [sum(x) for x in zip(pred_train, 

                                          [x * alpha_m for x in pred_train_i])]

        pred_test = [sum(x) for x in zip(pred_test, 

                                         [x * alpha_m for x in pred_test_i])]

#     pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)



#     print('Predictions : ',pred_test[:9])

    # Return error rate in train and test set

    return get_error_rate(pred_train, Y_train) , pred_test



""" PLOT FUNCTION ==========================================================="""

def plot_error_rate(er_train):

    df_error = pd.DataFrame([er_train, er_test]).T

    df_error.columns = ['Training']

    plot1 = df_error.plot(linewidth = 3, figsize = (8,6),

            color = ['lightblue'], grid = True)

    plot1.set_xlabel('Number of iterations', fontsize = 12)

    plot1.set_xticklabels(range(0,450,50))

    plot1.set_ylabel('Error rate', fontsize = 12)

    plot1.set_title('Error rate vs number of iterations', fontsize = 16)

    plt.axhline(y=er_test[0], linewidth=1, color = 'red', ls = 'dashed')



""" MAIN SCRIPT ============================================================="""



    



X_train, Y_train =x_train, y_train

X_test = x_test



# Fit a simple decision tree first

clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)

er_tree,pred_test_main = generic_clf(Y_train, X_train, X_test, clf_tree)



# Fit Adaboost classifier using a decision tree as base estimator

# Test with different number of iterations

er_train = np.array([er_tree])



# er_train.append(er_tree)

print(er_train)

print()

x_range = range(5, 20, 5)

for i in x_range:

    er_i , pred_test = adaboost_clf(Y_train, X_train, X_test, i, clf_tree)

    if(er_i < er_tree):

        pred_test_main = pred_test

    print(er_i)

#     er_train.append(np.array([er_i]))



# Compare error rate vs number of iterations

# np.isnan(y_train).any()

# y_train = np.nan_to_num(y_train,copy=True)

for i in range(len(pred_test_main)):

    if pred_test_main[i] > 0.5: 

        pred_test_main[i] = 1

    else:

        pred_test_main[i] = 0

d = {'Id': list(test['Id']), 'winPlacePerc': list(pred_test_main)}

df_opt = pd.DataFrame(data=d)
# y_train = np.nan_to_num(y_train,copy=True)

df_opt