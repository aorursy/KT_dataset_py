# Credit Card Fraud Detection 

# import the necessary packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# load the dataset

data = pd.read_csv("../input/creditcardfraud/creditcard.csv")
# explore the dataset

print(data.columns)
print(data.shape)

print(data.describe())

data = data.sample(frac = 0.1 , random_state = 1)

print(data.shape)



# plot histogram of each parameter

data.hist(figsize= (20,20))

plt.show()


# Determine number of fraud cases in dataset

Fraud = data[data['Class'] == 1]

Valid = data[data['Class'] == 0] 

outlier_fraction = len(Fraud) / float(len(Valid))

print(outlier_fraction)

print('Fraud Cases: {}'.format(len(Fraud)))

print('Valid Cases: {}'.format(len(Valid)))
# Correlation matrix

corrmat = data.corr()

fig = plt.figure(figsize=(12,9))

sns.heatmap(corrmat,vmax = .8,square = True)

plt.show()


# Get all the columns from the DataFrame

columns = data.columns.tolist()



# Filter the columns to remove data we do not want 

columns = [c for c in columns if c not in ["Class"]]



# Store the variable we'll be predicting on

target = 'Class'

X = data[columns]



Y = data[target]



print(X.shape)

print(Y.shape)
from sklearn.metrics import classification_report,accuracy_score

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor



# Define a random state

state = 1



# define the outlier detection methods

classifiers = {

    "Isolation Forest": IsolationForest(max_samples=len(X),contamination = outlier_fraction,random_state = state),

    "Local Outlier Factor": LocalOutlierFactor(

    n_neighbors = 20 ,

    contamination = outlier_fraction)

}
# Fit the model

n_outliers = len(Fraud)



for i, (clf_name,clf) in enumerate(classifiers.items()):

    # fit the data and tag outliers

    if clf_name == "Local Outlier Factor":

        y_pred = clf.fit_predict(X)

        scores_pred = clf.negative_outlier_factor_

    else:

        clf.fit(X)

        scores_pred = clf.decision_function(X)

        y_pred = clf.predict(X)

    

    # Reshape the prediction values to 0 for valid , 1 for fraud

    y_pred[y_pred == 1] = 0

    y_pred[y_pred == -1] = 1

    

    n_errors = (y_pred != Y).sum()

    

    # Run classification metrics

    print('{}: {}'.format(clf_name,n_errors))

    print(accuracy_score(Y,y_pred))

    print(classification_report(Y,y_pred))