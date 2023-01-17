#Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('whitegrid')

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression

from sklearn import svm
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.info()
train.describe(include='all')
train.head()
plt.subplot(1,4,1)

train.groupby('type').mean()['rotting_flesh'].plot(kind='bar',figsize=(7,4), color='r')

plt.subplot(1,4,2)

train.groupby('type').mean()['bone_length'].plot(kind='bar',figsize=(7,4), color='g')

plt.subplot(1,4,3)

train.groupby('type').mean()['hair_length'].plot(kind='bar',figsize=(7,4), color='y')

plt.subplot(1,4,4)

train.groupby('type').mean()['has_soul'].plot(kind='bar',figsize=(7,4), color='teal')
sns.factorplot("type", col="color", col_wrap=4, data=train, kind="count", size=2.4, aspect=.8)
#test_id will be used later, so save it

test_id = test['id']

train.drop(['id'], axis=1, inplace=True)

test.drop(['id'], axis=1, inplace=True)
#Deal with 'color' column

col = 'color'

dummies = pd.get_dummies(train[col], drop_first=False)

dummies = dummies.add_prefix("{}#".format(col))

train.drop(col, axis=1, inplace=True)

train = train.join(dummies)

dummies = pd.get_dummies(test[col], drop_first=False)

dummies = dummies.add_prefix("{}#".format(col))

test.drop(col, axis=1, inplace=True)

test = test.join(dummies)
X_train = train.drop('type', axis=1)

le = LabelEncoder()

Y_train = le.fit_transform(train.type.values)

X_test = test
clf = RandomForestClassifier(n_estimators=200)

clf = clf.fit(X_train, Y_train)

indices = np.argsort(clf.feature_importances_)[::-1]



# Print the feature ranking

print('Feature ranking:')



for f in range(X_train.shape[1]):

    print('%d. feature %d %s (%f)' % (f + 1, indices[f], X_train.columns[indices[f]],

                                      clf.feature_importances_[indices[f]]))
best_features=X_train.columns[indices[0:4]]

X = X_train[best_features]

Xt = X_test[best_features]
#Splitting data for validation

Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y_train, test_size=0.20, random_state=36)
#At first I try Random Forest.

#Normally you input all parameters and their potential values and run GridSearchCV.

#My PC isn't good enough so I divide parameters in two groups and repeatedly run two GridSearchCV until I'm satisfied with the result.

forest = RandomForestClassifier(max_depth = None,                                

                                min_samples_split =5,

                                min_weight_fraction_leaf = 0.0,

                                max_leaf_nodes = 60)



parameter_grid = {'n_estimators' : [10, 20, 100, 150],

                  'criterion' : ['gini', 'entropy'],

                  'max_features' : ['auto', 'sqrt', 'log2', None]

                 }



grid_search = GridSearchCV(forest, param_grid=parameter_grid, cv=StratifiedKFold(5))

grid_search.fit(X, Y_train)

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
forest = RandomForestClassifier(n_estimators = 20,

                                criterion = 'entropy',

                                max_features = 'auto')

parameter_grid = {

                  'max_depth' : [None, 5, 20, 100],

                  'min_samples_split' : [2, 5, 7],

                  'min_weight_fraction_leaf' : [0.0, 0.1],

                  'max_leaf_nodes' : [40, 60, 80],

                 }



grid_search = GridSearchCV(forest, param_grid=parameter_grid, cv=StratifiedKFold(5))

grid_search.fit(X, Y_train)

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
#Optimal parameters

clf = RandomForestClassifier(n_estimators=20, n_jobs=-1, criterion = 'entropy', max_features = 'auto',

                             min_samples_split=5, min_weight_fraction_leaf=0.0,

                             max_leaf_nodes=60, max_depth=100)

#Calibration improves probability predictions

calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)

calibrated_clf.fit(Xtrain, ytrain)

y_val = calibrated_clf.predict_proba(Xtest)

#Prediction 'y_val' shows probabilities of classes, so at first the most probable class is chosen,

#then it is converted to classes.

print("Validation accuracy: ", sum(pd.DataFrame(y_val, columns=le.classes_).idxmax(axis=1).values == le.inverse_transform(ytest))/len(ytest))
svc = svm.SVC(kernel='linear')

svc.fit(Xtrain, ytrain)

y_val_s = svc.predict(Xtest)

print("Validation accuracy: ", sum(le.inverse_transform(y_val_s)

                                   == le.inverse_transform(ytest))/len(ytest))
#The last model is logistic regression

logreg = LogisticRegression()



parameter_grid = {'solver' : ['newton-cg', 'lbfgs'],

                  'multi_class' : ['ovr', 'multinomial'],

                  'C' : [0.005, 0.01, 1, 10, 100, 1000],

                  'tol': [0.0001, 0.001, 0.005]

                 }



grid_search = GridSearchCV(logreg, param_grid=parameter_grid, cv=StratifiedKFold(5))

grid_search.fit(Xtrain, ytrain)

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
log_reg = LogisticRegression(C = 1, tol = 0.0001, solver='newton-cg', multi_class='multinomial')

log_reg.fit(Xtrain, ytrain)

y_val_l = log_reg.predict_proba(Xtest)

print("Validation accuracy: ", sum(pd.DataFrame(y_val_l, columns=le.classes_).idxmax(axis=1).values

                                   == le.inverse_transform(ytest))/len(ytest))
#So this is it. Now fit and model on full dataset

log_reg.fit(X, Y_train)

y_pred = log_reg.predict_proba(Xt)
submission = pd.DataFrame({'id':test_id,

                           'type':pd.DataFrame(y_pred, columns=le.classes_).idxmax(axis=1).values})
submission.to_csv('GGG_submission.csv', index=False)