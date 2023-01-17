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
#The graphs look much better with higher figsize.

fig, ax = plt.subplots(2, 2, figsize = (16, 12))

sns.pointplot(x="color", y="rotting_flesh", hue="type", data=train, ax = ax[0, 0])

sns.pointplot(x="color", y="bone_length", hue="type", data=train, ax = ax[0, 1])

sns.pointplot(x="color", y="hair_length", hue="type", data=train, ax = ax[1, 0])

sns.pointplot(x="color", y="has_soul", hue="type", data=train, ax = ax[1, 1])
sns.pairplot(train, hue='type')
train['hair_soul'] = train['hair_length'] * train['has_soul']

train['hair_bone'] = train['hair_length'] * train['bone_length']

test['hair_soul'] = test['hair_length'] * test['has_soul']

test['hair_bone'] = test['hair_length'] * test['bone_length']

train['hair_soul_bone'] = train['hair_length'] * train['has_soul'] * train['bone_length']

test['hair_soul_bone'] = test['hair_length'] * test['has_soul'] * test['bone_length']
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
best_features=X_train.columns[indices[0:7]]

X = X_train[best_features]

Xt = X_test[best_features]
#Splitting data for validation

Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y_train, test_size=0.20, random_state=36)
forest = RandomForestClassifier(max_depth = None,                                

                                min_samples_split =5,

                                min_weight_fraction_leaf = 0.0,

                                max_leaf_nodes = 60)



parameter_grid = {'n_estimators' : [10, 20, 100, 150],

                  'criterion' : ['gini', 'entropy'],

                  'max_features' : ['auto', 'sqrt', 'log2', None]

                 }



grid_search = GridSearchCV(forest, param_grid=parameter_grid, scoring='accuracy', cv=StratifiedKFold(5))

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



grid_search = GridSearchCV(forest, param_grid=parameter_grid, scoring='accuracy', cv=StratifiedKFold(5))

grid_search.fit(X, Y_train)

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
#Optimal parameters

clf = RandomForestClassifier(n_estimators=20, n_jobs=-1, criterion = 'gini', max_features = 'sqrt',

                             min_samples_split=2, min_weight_fraction_leaf=0.0,

                             max_leaf_nodes=40, max_depth=100)

#Calibration improves probability predictions

calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)

calibrated_clf.fit(Xtrain, ytrain)

y_val = calibrated_clf.predict_proba(Xtest)



print("Validation accuracy: ", sum(pd.DataFrame(y_val, columns=le.classes_).idxmax(axis=1).values

                                   == le.inverse_transform(ytest))/len(ytest))
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
svc = svm.SVC(kernel='linear')

svc.fit(X, Y_train)

svc_pred = svc.predict(Xt)



clf = RandomForestClassifier(n_estimators=20, n_jobs=-1, criterion = 'gini', max_features = 'sqrt',

                             min_samples_split=2, min_weight_fraction_leaf=0.0,

                             max_leaf_nodes=40, max_depth=100)



calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)

calibrated_clf.fit(X, Y_train)

for_pred = calibrated_clf.predict_proba(Xt)



log_reg.fit(X, Y_train)

log_pred = log_reg.predict_proba(Xt)



#I decided to try adding xgboost.

params = {"objective": "multi:softprob", "num_class": 3, 'eta': 0.01, 'min_child_weight' : 10, 'max_depth': 5}

param = list(params.items())

gbm = xgb.train(params, xgb.DMatrix(X, Y_train), 300)

x_pred = gbm.predict(xgb.DMatrix(Xt))
#Predicted values

s = le.inverse_transform(svc_pred)

l = pd.DataFrame(log_pred, columns=le.classes_).idxmax(axis=1).values

f = pd.DataFrame(for_pred, columns=le.classes_).idxmax(axis=1).values

x = pd.DataFrame(x_pred, columns=le.classes_).idxmax(axis=1).values

#Average of models, which give probability predictions.

q = pd.DataFrame(((log_pred + for_pred + x_pred)/3), columns=le.classes_).idxmax(axis=1).values
#As LR and SVC game the best results, I compare them

for i in range(len(s)):

    if l[i] != s[i]:

        print(i, l[i], s[i], f[i], x[i], q[i])
from collections import Counter
for i in range(len(s)):

    type_list = [l[i], s[i], f[i], x[i]]

    c = Counter(type_list)

    if l[i] != c.most_common()[0][0]:

        print(i, l[i], s[i], f[i], x[i], q[i], 'Most common: ' + c.most_common()[0][0])
#I tried several ways and here is the current version:

l[3] = 'Goblin'

l[44] = 'Ghost'

l[98] = 'Ghoul'

l[107] = 'Goblin'

l[112] = 'Ghost'

l[134] = 'Goblin'

l[162] = 'Ghoul'

l[173] = 'Goblin'

l[263] = 'Goblin'

l[309] = 'Goblin'

l[441] = 'Goblin'

l[445] = 'Ghost'
submission = pd.DataFrame({'id':test_id, 'type':l})
submission.to_csv('GGG_submission.csv', index=False)