# Importing libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
# Configuring visualizations

%matplotlib inline

plt.rcParams['figure.figsize'] = 12, 8
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
# Encoding categorical data (Sex)

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

X_train[:, 1] = labelencoder.fit_transform(X_train[:, 1])

X_test[:, 1] = labelencoder.transform(X_test[:, 1])
# Encoding categorical data (PClass, Embarked)

from sklearn.preprocessing import OneHotEncoder

onehotencoder_pclass = OneHotEncoder(categories='auto', drop='first', sparse=False)

pclass_train = onehotencoder_pclass.fit_transform(X_train[:, 0:1])

pclass_test = onehotencoder_pclass.transform(X_test[:, 0:1])

onehotencoder_embarked = OneHotEncoder(categories='auto', drop='first', sparse=False)

embarked_train = onehotencoder_embarked.fit_transform(X_train[:, 6:7])

embarked_test = onehotencoder_embarked.transform(X_test[:, 6:7])

X_train = np.concatenate((pclass_train, X_train[:, 1:6], embarked_train), axis=1)

X_test = np.concatenate((pclass_test, X_test[:, 1:6], embarked_test), axis=1)

del pclass_train, pclass_test, embarked_train, embarked_test
# Feature scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_mean=True, with_std=True)

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
# Building a SVM model

from sklearn.svm import SVC

classifier = SVC(kernel='rbf', C=1.0, gamma='auto', probability=False, 

                 decision_function_shape='ovr')

classifier.fit(X_train, y_train)
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

plt.ylim(.69, .92)

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
# Plotting validation curves against C

from sklearn.model_selection import validation_curve

C_range = np.geomspace(1e-2, 1e+2, 41)

train_scores, test_scores = validation_curve(classifier, X_train, y_train, cv=10,

                                             param_name='C', param_range=C_range,

                                             scoring='accuracy', n_jobs=-1)

plt.figure()

plt.title('Validation Curves')

plt.ylim(.5, .9)

plt.xlabel('C')

plt.ylabel('Score')

train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)

test_scores_std = np.std(test_scores, axis=1)

plt.grid()

plt.semilogx(C_range, train_scores_mean, label='Training score', color='darkorange', lw=2)

plt.semilogx(C_range, test_scores_mean, label='Cross-validation score', color='navy', lw=2)

plt.fill_between(C_range, train_scores_mean - train_scores_std, 

                 train_scores_mean + train_scores_std, alpha=0.2, color='darkorange', lw=2)

plt.fill_between(C_range, test_scores_mean - test_scores_std,

                 test_scores_mean + test_scores_std, alpha=0.2, color='navy', lw=2)

plt.legend(loc='best')

plt.show()
# Plotting validation curves against gamma

from sklearn.model_selection import validation_curve

gamma_range = np.geomspace(1e-4, 1e+1, 51)

train_scores, test_scores = validation_curve(classifier, X_train, y_train, cv=10,

                                             param_name='gamma', param_range=gamma_range,

                                             scoring='accuracy', n_jobs=-1)

plt.figure()

plt.title('Validation Curves')

plt.ylim(.5, 1)

plt.xlabel(r'$\gamma$')

plt.ylabel('Score')

train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)

test_scores_std = np.std(test_scores, axis=1)

plt.grid()

plt.semilogx(gamma_range, train_scores_mean, label='Training score', color='darkorange', lw=2)

plt.semilogx(gamma_range, test_scores_mean, label='Cross-validation score', color='navy', lw=2)

plt.fill_between(gamma_range, train_scores_mean - train_scores_std, 

                 train_scores_mean + train_scores_std, alpha=0.2, color='darkorange', lw=2)

plt.fill_between(gamma_range, test_scores_mean - test_scores_std,

                 test_scores_mean + test_scores_std, alpha=0.2, color='navy', lw=2)

plt.legend(loc='best')

plt.show()
# Grid search

from sklearn.model_selection import GridSearchCV

param_grid = {'kernel': ['rbf'], 'C': list(np.geomspace(1e-1, 1e+1, 21)), 

              'gamma': list(np.geomspace(1e-2, 1e+0, 21))}

grid_search = GridSearchCV(classifier, param_grid, iid=False, refit=True, cv=10, 

                           return_train_score=True, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

print('best parameters:', grid_search.best_params_)

print('best score:', grid_search.best_score_)
# Making predictions and submitting outputs

y_pred = grid_search.predict(X_test)

submission = pd.DataFrame({'PassengerId': test_set.iloc[:, 0].values,

                           'Survived': y_pred})

submission.to_csv('submission_logistic.csv', index=False)