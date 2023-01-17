%matplotlib inline

import pandas as pd

import numpy as np



df = pd.read_csv('../input/HR_comma_sep.csv')

df.head()
# convert categorical variables to dummy variables

from pandas import get_dummies

df = get_dummies(df, prefix='is_')

df.head()
# Seperate label and features

y = df['left']

X = df.drop('left',1)

X.head()
# print Left vs stayed

left = df[df['left']==1].count()[0]

stayed = df[df['left']==0].count()[0]

prc_left = (left/(df.count()[0]))*100.0

print('# Left: {:}'.format(left))

print('# Stayed: {:}'.format(stayed))

print('% Left: {:.2f}%'.format(prc_left))



# Other important metrics

n_samples, n_features = df.shape

n_classes = len(y.unique())

print('n_samples: {}'.format(n_samples))

print('n_features: {}'.format(n_features))

print('n_classes: {}'.format(n_classes))
# Review skew

import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(14,6))

fig, ax = plt.subplots(5)



for i, column in enumerate(['satisfaction_level',

                            'last_evaluation',

                            'number_project',

                            'average_montly_hours',

                            'time_spend_company']):

    sns.distplot(X[column], ax=ax[i])

    

plt.tight_layout()
# scale features that don't have gaussian distribution

from sklearn import preprocessing



X[['satisfaction_level',

  'last_evaluation',

  'number_project',

  'average_montly_hours',

  'time_spend_company']] = preprocessing.scale(X[['satisfaction_level',

                                                  'last_evaluation',

                                                  'number_project',

                                                  'average_montly_hours',

                                                  'time_spend_company']])

X.head()
# Review scaled features

plt.figure(figsize=(14,6))

fig, ax = plt.subplots(5)



for i, column in enumerate(['satisfaction_level',

                            'last_evaluation',

                            'number_project',

                            'average_montly_hours',

                            'time_spend_company']):

    sns.distplot(X[column], ax=ax[i])

    

plt.tight_layout()
# Split data and train model

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)



clf = SVC() # Add cross validation to find best hyper paramaters later

clf.fit(X_train, y_train)
print('MSE on train set: {:.2f}%'.format(np.mean((y_train - clf.predict(X_train))**2)*100))

print('MSE on test set: {:.2f}%'.format(np.mean((y_test - clf.predict(X_test))**2)*100))
# Accuracy Score

clf.score(X_test, y_test)
# F1 Score

from sklearn.metrics import classification_report



print(classification_report(y_test, clf.predict(X_test)))