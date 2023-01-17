# Imports

import numpy as np

import pandas as pd



# Load data

df = pd.read_csv('../input/HR_comma_sep.csv')



df.head()
# Get Label

y = df['left']

del df['left']

y.value_counts()
# Rename sales column to department

df = df.rename(columns={'sales':'department'})

df.columns
# Convert categorical features to dummy variables



df = pd.get_dummies(df, prefix=['department', 'salary'])

df.head()
# Scale features

from sklearn.preprocessing import StandardScaler



# Create scaler

columns = df.columns

scaler = StandardScaler().fit(df)



# Scale Data

df = pd.DataFrame(scaler.transform(df), columns=columns)

df.head()
# Check standard deviation of scaled DataFrame

df.std()
# Convert dataframe into X numpy array

X = df.values

print(X)

print('#'*20)



# Convert dataframe into y numpy array

y = np.array(y.values)

print(y)
# Grid search to find best paramaters

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold



param_grid = [

  #{'C': [1, 10, 100, 1000], 'kernel': ['linear']},

  #{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}, # Ran too slowly with these

  {'kernel':['linear']},

  {'kernel':['rbf'], 'gamma':[0.001, 0.0001]}

]



estimator = SVC(C=1)

clf = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1)

cross_val_score(clf, X, y)
# Split data

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)



#cv_split = round(len(X_train)/2)



#CV_X_train = X_train[cv_split:]

#CV_y_train = y_train[cv_split:]

#X_train = X_train[:cv_split]

#y_train = y_train[:cv_split]
# Set paramaters by cross-validation

clf.fit(X_train, y_train)



# QUESTION: Does this mean the best params have already been chosen against the 

#           cross validation set? Is this a nested or non-nested cross validation?

clf.best_estimator_
# Get Classification report

from sklearn.metrics import classification_report, confusion_matrix



pred = clf.predict(X_test)



print(classification_report(y_test,pred))

print(confusion_matrix(y_test,pred))
# 

# Ideas on how to improve this?

#
# Create Data Frame of confidence vs correct prediction/actual/prediction

confidence = pd.DataFrame({'confidence': clf.decision_function(X_test),

                           'actual': y_test,

                           'prediction': clf.predict(X_test),

                           'correct_prediction': y_test == clf.predict(X_test)})

confidence.head()
confidence.sort_values('confidence', ascending=False)[confidence.prediction == 1][confidence.actual == 0]
# Return Employee most likely to leave, but did not

mostlikely_idx = confidence.sort_values('confidence', ascending=False)[confidence.prediction == 1][confidence.actual == 0][:1].index



mostlikely_e = pd.concat([confidence.ix[mostlikely_idx, :],

                          df.ix[mostlikely_idx, :]], axis=1)

mostlikely_e.unstack()
# Most confident employee to stay

mostlikely_idx = confidence.sort_values('confidence', ascending=False)[confidence.prediction == 0][:1].index

mostlikely_idx_e = pd.concat([confidence.ix[mostlikely_idx,:],

                              df.ix[mostlikely_idx,:]], axis=1)

mostlikely_idx_e.unstack()