from __future__ import division

import operator



import pandas as pd

from pandas import Series, DataFrame

import numpy as np



from sklearn.cross_validation import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
X_train = pd.read_csv('../input/train.csv', dtype={'Age': np.float64})

y_train = X_train['Survived']
X_train[X_train.Fare < 1.0][['Name','Pclass','Sex','Ticket','Survived']]
X_train[X_train['Name'].apply(lambda x: x.find('Rev.') != -1)][['Name','Pclass','Sex','Ticket','Survived']]
def check_classifiers(X_train, Y_train):

    

    _cv = 5

    classifier_score = {}

    

    scores = cross_val_score(LogisticRegression(), X, y, cv=_cv)

    classifier_score['LogisticRegression'] = scores.mean()

    

    scores = cross_val_score(KNeighborsClassifier(), X, y, cv=_cv)

    classifier_score['KNeighborsClassifier'] = scores.mean()

    

    scores = cross_val_score(RandomForestClassifier(), X, y, cv=_cv)

    classifier_score['RandomForestClassifier'] = scores.mean()

    

    scores = cross_val_score(SVC(), X, y, cv=_cv)

    classifier_score['SVC'] = scores.mean()

    

    scores = cross_val_score(GaussianNB(), X, y, cv=_cv)

    classifier_score['GaussianNB'] = scores.mean()



    return sorted(classifier_score.items(), key=operator.itemgetter(1), reverse=True)
def check_employee(passenger):

    name, fare = passenger

    if fare < 1 or name.find('Rev.') != -1:

        return 1.0

    else:

        return 0.0



X_train['employee'] = X_train[['Name', 'Fare']].apply(check_employee, axis=1)
features = ['employee']
X = DataFrame(X_train[features])

y = y_train

scores = check_classifiers(X, y)

scores
# get Correlation Coefficient for each feature using Logistic Regression

coeff_df = DataFrame(X.columns)

coeff_df.columns = ['Features']

classifier = LogisticRegression()

coeff_df["Coefficient Estimate"] = pd.Series(classifier.fit(X, y).coef_[0])



# preview

coeff_df
null_hypothesis = 1 - X_train.Survived.mean()

null_hypothesis