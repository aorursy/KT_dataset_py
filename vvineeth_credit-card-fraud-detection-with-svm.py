import numpy as np

import pandas as pd

import sys

import matplotlib.pyplot as plt

import seaborn as sns

import scipy 

import sklearn
data = pd.read_csv("../input/creditcardfraud/creditcard.csv")

data.columns
data.shape
# random_state helps assure that you always get the same output when you split the data

# this helps create reproducible results and it does not actually matter what the number is

# frac is percentage of the data that will be returned

data = data.sample(frac = 0.2, random_state = 1)

print(data.shape)
# plot the histogram of each parameter

data.hist(figsize=(20,20))

plt.show()
# determine the number of fraud cases

fraud = data[data['Class'] == 1]

valid = data[data['Class'] == 0]

 

outlier_frac = len(fraud)/float(len(valid))

print("outliar frac: ", outlier_frac)



print("Number of fraud transactions: ", len(fraud))

print("Number of valid transactions: ", len(valid))
# Correlaton matrix

corr = data.corr()

fig = plt.figure(figsize=(10,8))



sns.heatmap(corr, cmap='viridis')

plt.show()
# get the columns from the dataframe

columns = data.columns.tolist()



# filter the columns to remove the data we do not want

columns = [c for c in columns if c not in ["Class"]]



# store the variable we will be predicting on which is class

target = "Class"



# X includes everything except our class column

X = data[columns]



# y includes all the class labels for each sample

# this is also one-dimensional

Y = data[target]



# print the shapes of X and Y

print(X.shape)

print(Y.shape)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor
# define a random state

state = 1



# define the outlier detection methods

classifiers = {

    # contamination is the number of outliers we think there are

    'Isolation Forest': IsolationForest(max_samples = len(X),

                                       contamination = outlier_frac,

                                       random_state = state),

    # number of neighbors to consider, the higher the percentage of outliers the higher you want to make this number

    'Local Outlier Factor': LocalOutlierFactor(

    n_neighbors = 20,

    contamination = outlier_frac)

}
n_outliers = len(fraud)



for i, (clf_name, clf) in enumerate(classifiers.items()):

    

    # fit the data and tag outliers

    if clf_name == 'Local Outlier Factor':

        y_pred = clf.fit_predict(X)

        scores_pred = clf.negative_outlier_factor_

    else:

        clf.fit(X)

        scores_pred = clf.decision_function(X)

        y_pred = clf.predict(X)

        

    # reshape the prediction values to 0 for valid and 1 for fraud

    y_pred[y_pred == 1] = 0

    y_pred[y_pred == -1] = 1



    # calculate the number of errors

    n_errors = (y_pred != Y).sum()

    

    # classification matrix

    print('{}: {}'.format(clf_name, n_errors))

    print(accuracy_score(Y, y_pred))

    print(classification_report(Y, y_pred))
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split



X = data[columns]

y = data[target]



train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.4, random_state=100)

model = SVC()



model.fit(train_X, train_y)



prediction = model.predict(test_X)
print(confusion_matrix(test_y, prediction))

print(classification_report(test_y, prediction))

print('\n')

print(accuracy_score(test_y, prediction))