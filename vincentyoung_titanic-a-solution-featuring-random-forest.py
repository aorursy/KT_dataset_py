# Importing libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
# Configuring visualizations

%matplotlib inline

plt.rcParams['figure.figsize'] = 12, 8
# Setting random state

RANDOM_STATE = 123
# Loading datasets

train_set = pd.read_csv('../input/titanic/train.csv')

test_set = pd.read_csv('../input/titanic/test.csv')

X_train = train_set.iloc[:, [2, 4, 5, 6, 7, 9, 11]].values

X_test = test_set.iloc[:, [1, 3, 4, 5, 6, 8, 10]].values

y_train = train_set.iloc[:, 1].values
# Exploring train set

train_set.info()

train_set.describe(include='all')
# Exploring test set

test_set.info()

test_set.describe(include='all')
# Taking care of missing data (Age, Embarked, Fare)

from sklearn.impute import SimpleImputer

imputer_age = SimpleImputer(missing_values=np.nan, strategy='mean')

X_train[:, 2:3] = imputer_age.fit_transform(X_train[:, 2:3])

X_test[:, 2:3] = imputer_age.fit_transform(X_test[:, 2:3])

imputer_embarked = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

X_train[:, 6:7] = imputer_embarked.fit_transform(X_train[:, 6:7])

imputer_fare = SimpleImputer(missing_values=np.nan, strategy='median')

X_test[:, 5:6] = imputer_fare.fit_transform(X_test[:, 5:6])
# Encoding categorical data (PClass, Sex, Embarked)

from sklearn.preprocessing import LabelEncoder

labelencoder_pclass = LabelEncoder()

X_train[:, 0] = labelencoder_pclass.fit_transform(X_train[:, 0])

X_test[:, 0] = labelencoder_pclass.transform(X_test[:, 0])

labelencoder_sex = LabelEncoder()

X_train[:, 1] = labelencoder_sex.fit_transform(X_train[:, 1])

X_test[:, 1] = labelencoder_sex.transform(X_test[:, 1])

labelencoder_embarked = LabelEncoder()

X_train[:, 6] = labelencoder_embarked.fit_transform(X_train[:, 6])

X_test[:, 6] = labelencoder_embarked.transform(X_test[: ,6])
# Plotting OOB scores against number of trees and number of features

from collections import OrderedDict

from sklearn.ensemble import RandomForestClassifier

ensemble_clfs = [     

                 ('max_features=2', RandomForestClassifier(max_features=2)), 

#                 ('max_features=3', RandomForestClassifier(max_features=3)), 

#                 ('max_features=4', RandomForestClassifier(max_features=4)), 

#                 ('max_features=5', RandomForestClassifier(max_features=5)), 

#                 ('max_features=6', RandomForestClassifier(max_features=6)), 

                 ('max_features=7', RandomForestClassifier(max_features=7))

                 ]

error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

[min_estimators, max_estimators] = [20, 750]

for label, clf in ensemble_clfs:

    for i in range(min_estimators, max_estimators + 1, 10):

        clf.set_params(n_estimators=i, warm_start=True, oob_score=True, 

                       min_impurity_decrease=1e-4, random_state=RANDOM_STATE)

        clf.fit(X_train, y_train)

        error_rate[label].append((i, clf.oob_score_))

for label, clf_err in error_rate.items():

    xs, ys = zip(*clf_err)

    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)

plt.xlabel('n_estimators')

plt.ylabel('OOB error rate')

plt.legend(loc='best')

plt.grid()

plt.show()
# Building a Random Forest model

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=400, bootstrap=True, 

                                    criterion='gini', max_depth=None, 

                                    min_samples_split=2, min_samples_leaf=1, 

                                    max_features='auto', max_leaf_nodes=None, 

                                    min_impurity_decrease=1e-3, oob_score=True, 

                                    n_jobs=-1, random_state=RANDOM_STATE)

classifier.fit(X_train, y_train)

print('OOB score:', classifier.oob_score_)

print('feature importances:', classifier.feature_importances_)
# Cross validation

from sklearn.model_selection import cross_validate

results = cross_validate(classifier, X_train, y_train, cv=10, scoring='accuracy',

                         return_train_score=True, return_estimator=False, n_jobs=-1)

print('train score:', results['train_score'].mean())

print('test score:', results['test_score'].mean())
# Plotting learning curves

from sklearn.model_selection import learning_curve

m_range = np.linspace(.05, 1.0, 20)

train_sizes, train_scores, test_scores = learning_curve(classifier, X_train, y_train, cv=10,  

                                                        train_sizes=m_range, shuffle=False,

                                                        scoring='accuracy', n_jobs=-1)

plt.figure()

plt.title('Learning Curves')

plt.ylim(.6, 1.05)

plt.xlabel('Training examples')

plt.ylabel('Score')

train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)

test_scores_std = np.std(test_scores, axis=1)

plt.grid()

plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')

plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 

                 train_scores_mean + train_scores_std, alpha=0.1, color='r')

plt.fill_between(train_sizes, test_scores_mean - test_scores_std, 

                 test_scores_mean + test_scores_std, alpha=0.1, color='g')

plt.legend(loc='best')

plt.show()
# Plotting validation curves

from sklearn.model_selection import validation_curve

param_range = np.geomspace(1e-5, 1e-1, 21)

train_scores, test_scores = validation_curve(classifier, X_train, y_train, cv=10,

                                             param_name='min_impurity_decrease', 

                                             param_range=param_range, 

                                             scoring='accuracy', n_jobs=-1)

plt.figure()

plt.title('Validation Curves')

plt.ylim(.7, 1.05)

plt.xlabel('Size of Trees')

plt.ylabel('Score')

train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)

test_scores_std = np.std(test_scores, axis=1)

plt.grid()

plt.semilogx(param_range, train_scores_mean, label='Training score', color='darkorange', lw=2)

plt.semilogx(param_range, test_scores_mean, label='Cross-validation score', color='navy', lw=2)

plt.fill_between(param_range, train_scores_mean - train_scores_std, 

                 train_scores_mean + train_scores_std, alpha=0.2, color='darkorange', lw=2)

plt.fill_between(param_range, test_scores_mean - test_scores_std,

                 test_scores_mean + test_scores_std, alpha=0.2, color='navy', lw=2)

plt.legend(loc='best')

plt.show()
# Grid search

from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [400], 'max_features': ['auto', None], 

              'min_impurity_decrease': list(np.geomspace(1e-4, 1e-3, 6))}

grid_search = GridSearchCV(classifier, param_grid, iid=False, refit=True, cv=10,

                           return_train_score=False, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

print('best parameters:', grid_search.best_params_)

print('best score:', grid_search.best_score_)
# Createing a new feature representing family sizes and conducting preprocessing

family_train = X_train[:, 3] + X_train[:, 4] + 1

family_train = family_train.reshape(len(family_train), 1)

family_test = X_test[:, 3] + X_test[:, 4] + 1

family_test = family_test.reshape(len(family_test), 1)

X_train = np.concatenate((X_train[:, :3], family_train, X_train[:, 5:]), axis=1)

X_test = np.concatenate((X_test[:, :3], family_test, X_test[:, 5:]), axis=1)

del family_train, family_test
# Dropping feature Embarked

X_train = X_train[:, :-1]

X_test = X_test[:, :-1]
# Building a Random Forest model

from sklearn.ensemble import RandomForestClassifier

best_mid = grid_search.best_params_['min_impurity_decrease']

classifier = RandomForestClassifier(n_estimators=400, bootstrap=True, 

                                    criterion='gini', max_depth=None, 

                                    min_samples_split=2, min_samples_leaf=1, 

                                    max_features='auto', max_leaf_nodes=None, 

                                    min_impurity_decrease=best_mid, oob_score=True, 

                                    n_jobs=-1, random_state=RANDOM_STATE)

classifier.fit(X_train, y_train)

print('OOB score:', classifier.oob_score_)

print('feature importances:', classifier.feature_importances_)
# Cross validation

from sklearn.model_selection import cross_validate

results = cross_validate(classifier, X_train, y_train, cv=10, scoring='accuracy',

                         return_train_score=True, return_estimator=False, n_jobs=-1)

print('train score:', results['train_score'].mean())

print('test score:', results['test_score'].mean())
# Plotting learning curves

from sklearn.model_selection import learning_curve

m_range = np.linspace(.05, 1.0, 20)

train_sizes, train_scores, test_scores = learning_curve(classifier, X_train, y_train, cv=10,  

                                                        train_sizes=m_range, shuffle=False,

                                                        scoring='accuracy', n_jobs=-1)

plt.figure()

plt.title('Learning Curves')

plt.ylim(.6, 1.05)

plt.xlabel('Training examples')

plt.ylabel('Score')

train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)

test_scores_std = np.std(test_scores, axis=1)

plt.grid()

plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')

plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 

                 train_scores_mean + train_scores_std, alpha=0.1, color='r')

plt.fill_between(train_sizes, test_scores_mean - test_scores_std, 

                 test_scores_mean + test_scores_std, alpha=0.1, color='g')

plt.legend(loc='best')

plt.show()
# Plotting validation curves

from sklearn.model_selection import validation_curve

param_range = np.geomspace(1e-5, 1e-1, 21)

train_scores, test_scores = validation_curve(classifier, X_train, y_train, cv=10,

                                             param_name='min_impurity_decrease', 

                                             param_range=param_range, 

                                             scoring='accuracy', n_jobs=-1)

plt.figure()

plt.title('Validation Curves')

plt.ylim(.7, 1.05)

plt.xlabel('Size of Trees')

plt.ylabel('Score')

train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)

test_scores_std = np.std(test_scores, axis=1)

plt.grid()

plt.semilogx(param_range, train_scores_mean, label='Training score', color='darkorange', lw=2)

plt.semilogx(param_range, test_scores_mean, label='Cross-validation score', color='navy', lw=2)

plt.fill_between(param_range, train_scores_mean - train_scores_std, 

                 train_scores_mean + train_scores_std, alpha=0.2, color='darkorange', lw=2)

plt.fill_between(param_range, test_scores_mean - test_scores_std,

                 test_scores_mean + test_scores_std, alpha=0.2, color='navy', lw=2)

plt.legend(loc='best')

plt.show()
# Grid search

from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [400], 'max_features': [2, 3, 4, 5], 

              'min_impurity_decrease': list(np.geomspace(1e-4, 1e-2, 11))}

grid_search = GridSearchCV(classifier, param_grid, iid=False, refit=True, cv=10,

                           return_train_score=False, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

print('best parameters:', grid_search.best_params_)

print('best score:', grid_search.best_score_)
# Making predictions and submitting

y_pred = grid_search.predict(X_test)

submission = pd.DataFrame({'PassengerId': test_set.iloc[:, 0].values,

                           'Survived': y_pred})

submission.to_csv('submission_RandomForest.csv', index=False)