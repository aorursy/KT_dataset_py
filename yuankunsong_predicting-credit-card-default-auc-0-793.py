import numpy as np 
import pandas as pd 

# graphics
import seaborn as sns
import matplotlib.pyplot as plt

# modeling
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

import xgboost as xgb
from xgboost import XGBClassifier


import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
credit_card = pd.read_csv('../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')

credit_card.drop(columns='ID', inplace=True) # drop ID, as this is irrelevant

credit_card.rename(columns={'default.payment.next.month':'default'}, inplace=True)
credit_card.head()
credit_card.info()
credit_card.describe()
credit_card.isnull().sum()
# count 

plt.figure(figsize=(6,6))
sns.countplot(data=credit_card, x='default')
plt.title('default', size=19)
plt.show()
print('non-default:', len(credit_card.default) - sum(credit_card.default))
print('default:', sum(credit_card.default))
plt.figure(figsize=(6,6))
sns.countplot(data=credit_card, x='default', hue='SEX')
plt.title('default', size=19)
plt.show()
plt.figure(figsize=(6,6))
sns.countplot(data=credit_card, x='default', hue='EDUCATION')
plt.title('default', size=19)
plt.show()
plt.figure(figsize=(6,6))
sns.countplot(data=credit_card, x='default', hue='MARRIAGE')
plt.title('default', size=19)
plt.show()
plt.figure(figsize=(6,6))
sns.distplot(credit_card.AGE)
plt.title('Age', size=19)
plt.show()
non_default = credit_card.loc[credit_card.default == 0]
default = credit_card.loc[credit_card.default == 1]

plt.figure(figsize=(6,6))
sns.distplot(non_default.AGE)
plt.title('non-default', size=19)
plt.show()

plt.figure(figsize=(6,6))
sns.distplot(non_default.AGE)
plt.title('default', size=19)
plt.show()
cc = credit_card.copy()

# base X and y
y = cc.iloc[:,-1]
X = cc.iloc[:,0:-1]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=10)

print('x_train:', x_train.shape)
print('x_test:', x_test.shape)
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)
# proportion of 0 and 1 in train/test, just to make sure train and test had same distribution

print(y_train.value_counts()/len(y_train))
print('\n')
print(y_test.value_counts()/len(y_test))
model = XGBClassifier(scale_pos_weight=3.52, eval_metric='auc')
model.fit(x_train, y_train)
pred = model.predict_proba(x_test)
roc_auc_score(y_test, pred[:,1])
# tune scale_pos_weight

model = XGBClassifier()

scale_pos_weight = [3.45, 3.5, 3.55, 3.6, 3.65]
param = dict(scale_pos_weight = scale_pos_weight)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
grid_search = GridSearchCV(model, param, scoring='roc_auc', n_jobs = -1, cv=kfold)
grid_result = grid_search.fit(X, y)

print('best cv score:', grid_result.best_score_, grid_result.best_params_)

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, std, param in zip(means, stds, params):
    print('%f, (%f),%r' % (mean, std, param))

plt.errorbar(scale_pos_weight, means, yerr=stds)
plt.title('CV error vs n_estimators')
plt.show()
# # tune n_estimator

# model = XGBClassifier(scale_pos_weight=3.65, eval_metric='auc')

# # n_estimators : 50 ~ 400
# n_estimators = range(50, 400, 50)
# param = dict(n_estimators = n_estimators)
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
# grid_search = GridSearchCV(model, param, scoring='roc_auc', n_jobs = -1, cv=kfold)
# grid_result = grid_search.fit(X, y)

# print('best cv score:', grid_result.best_score_, grid_result.best_params_)

# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']

# for mean, std, param in zip(means, stds, params):
#     print('%f, (%f),%r' % (mean, std, param))

# plt.errorbar(n_estimators, means, yerr=stds)
# plt.title('CV error vs n_estimators')
# plt.show()
# # tune tree size

# model = XGBClassifier(scale_pos_weight=3.65, eval_metric='auc')

# # tree depth = 1,3,5,7,9
# max_depth = [1,3,5,7,9]
# param = dict(max_depth = max_depth)
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
# grid_search = GridSearchCV(model, param, scoring='roc_auc', n_jobs = -1, cv=kfold)
# grid_result = grid_search.fit(X, y)

# print('best cv score:', grid_result.best_score_, grid_result.best_params_)

# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']

# for mean, std, param in zip(means, stds, params):
#     print('%f, (%f),%r' % (mean, std, param))

# plt.errorbar(max_depth, means, yerr=stds)
# plt.title('CV error vs max_depth')
# plt.show()
# # tune learning rate

# model = XGBClassifier(scale_pos_weight=3.65, eval_metric='auc')

# # learning rate
# learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
# param = dict(learning_rate = learning_rate)
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
# grid_search = GridSearchCV(model, param, scoring='roc_auc', n_jobs = -1, cv=kfold)
# grid_result = grid_search.fit(X, y)

# print('best cv score:', grid_result.best_score_, grid_result.best_params_)

# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']

# for mean, std, param in zip(means, stds, params):
#     print('%f, (%f),%r' % (mean, std, param))

# plt.errorbar(learning_rate, means, yerr=stds)
# plt.title('CV error vs learning rate')
# plt.show()
# # tune learning rate and n_estimator

# model = XGBClassifier(scale_pos_weight=3.65, eval_metric='auc')

# # learning rate and n_estimator
# learning_rate = [0.005, 0.01, 0.05, 0.1, 0.2]
# n_estimators = [50, 100, 200, 300, 400]
# param = dict(learning_rate = learning_rate, n_estimators=n_estimators)
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
# grid_search = GridSearchCV(model, param, scoring='roc_auc', n_jobs = -1, cv=kfold)
# grid_result = grid_search.fit(X, y)

# print('best cv score:', grid_result.best_score_, grid_result.best_params_)

# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']

# for mean, std, param in zip(means, stds, params):
#     print('%f, (%f),%r' % (mean, std, param))


# # row sampling

# model = XGBClassifier(scale_pos_weight=3.65, eval_metric='auc')

# # learning rate and n_estimator
# subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
# param = dict(subsample = subsample)
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
# grid_search = GridSearchCV(model, param, scoring='roc_auc', n_jobs = -1, cv=kfold)
# grid_result = grid_search.fit(X, y)

# print('best cv score:', grid_result.best_score_, grid_result.best_params_)

# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']

# for mean, std, param in zip(means, stds, params):
#     print('%f, (%f),%r' % (mean, std, param))

# plt.errorbar(subsample, means, yerr=stds)
# plt.title('CV error vs max_depth')
# plt.show()
# # tune column sampling

# model = XGBClassifier(scale_pos_weight=3.65, eval_metric='auc')

# colsample_bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
# param = dict(colsample_bytree = colsample_bytree)
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
# grid_search = GridSearchCV(model, param, scoring='roc_auc', n_jobs = -1, cv=kfold)
# grid_result = grid_search.fit(X, y)

# print('best cv score:', grid_result.best_score_, grid_result.best_params_)

# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']

# for mean, std, param in zip(means, stds, params):
#     print('%f, (%f),%r' % (mean, std, param))

# plt.errorbar(colsample_bytree, means, yerr=stds)
# plt.title('CV error vs max_depth')
# plt.show()
# # tune column sub sampling

# model = XGBClassifier(scale_pos_weight=3.65, eval_metric='auc')

# colsample_bylevel = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
# param = dict(colsample_bylevel = colsample_bylevel)
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
# grid_search = GridSearchCV(model, param, scoring='roc_auc', n_jobs = -1, cv=kfold)
# grid_result = grid_search.fit(X, y)

# print('best cv score:', grid_result.best_score_, grid_result.best_params_)

# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']

# for mean, std, param in zip(means, stds, params):
#     print('%f, (%f),%r' % (mean, std, param))

# plt.errorbar(colsample_bylevel, means, yerr=stds)
# plt.title('CV error vs max_depth')
# plt.show()
# # tuning all parameters togather

# model = XGBClassifier(n_estimators=100, scale_pos_weight=3.65, eval_metric='auc')

# learning_rate = [0.04, 0.045, 0.05, 0.055, 0.06]
# max_depth = [2, 3]
# colsample_bytree = [0.2, 0.3, 0.4, 0.5]
# min_child_weight = [1, 1.8, 2.0, 2.2]
# param = dict(learning_rate=learning_rate, max_depth = max_depth, colsample_bytree=colsample_bytree, min_child_weight=min_child_weight)
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
# grid_search = GridSearchCV(model, param, scoring='roc_auc', n_jobs = -1, cv=kfold)
# grid_result = grid_search.fit(X, y)

# print('best cv score:', grid_result.best_score_, grid_result.best_params_)

# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']

# for mean, std, param in zip(means, stds, params):
#     print('%f, (%f),%r' % (mean, std, param))
# final model
model = XGBClassifier(n_estimators=100, learning_rate=0.06, colsample_bytree=0.4, max_depth=3, min_child_weight=1.8,scale_pos_weight=3.65 )
model.fit(x_train, y_train)

pred = model.predict_proba(x_test)
print('auc:', roc_auc_score(y_test, pred[:,1]))

pred = model.predict(x_test)
conf_mx = pd.crosstab(pred, y_test, rownames=['Prediction'], colnames=['True Value'])

plt.figure(figsize=(6,6))
sns.heatmap(conf_mx, annot=True, fmt='g', square=True,
            xticklabels=['non-default', 'default'],
           yticklabels=['non-default', 'default'],
           cmap="Greens")
plt.title('Confusion Matrix', fontsize=18)
plt.show()
fig, ax = plt.subplots(figsize=(9,7))
xgb.plot_importance(model, height=0.6, ax=ax, color='red')
plt.show()