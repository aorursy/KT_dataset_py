import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot')

import warnings
def ignore_warn(*args, **kwagrs):
    pass
warnings.warn = ignore_warn
data = pd.read_csv('../input/bank.csv')
data.head()
data.shape
data.describe()
data.info()
data.columns
count_list = ['age', 'job', 'marital', 'education', 'contact', 'month', 'poutcome']
for feature in count_list:
    plt.figure(figsize=(12, 7))
    sns.countplot(feature, data=data)
    plt.xticks(rotation=90)
    plt.show()
data.rename(columns={'y': 'client'}, inplace=True)
data['client'] = data['client'].map({'no': 0, 'yes': 1})
data.head()
sns.countplot('client', data=data)
plt.show()
age = sns.FacetGrid(data, col='client')
age.map(plt.hist, 'age', bins=20)
plt.show()
job = data[['job', 'client']].groupby(['job'], as_index=False).mean().sort_values(ascending=False, by='client')
job
plt.figure(figsize=(8, 5))
sns.barplot(x='job', y='client', data=job)
plt.xticks(rotation=90)
plt.show()
g = sns.pointplot(x='job', y='client', data=data)
plt.xticks(rotation=90)
data[['education', 'client']].groupby(['education'], as_index=False).mean().sort_values(by='client', ascending=False)
g = sns.catplot(x='job', y='age', col='education', hue='client', kind='point', data=data)
for ax in g.axes.flat:
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, rotation=90)
plt.show()
g = sns.FacetGrid(col='client', data=data)
g.map(plt.hist, 'balance', bins=20)
g.set(xlim=(0, 25000))
data.head()
data[['client', 'default']].groupby(['default'], as_index=False).mean()
data[['client', 'housing']].groupby(['housing'], as_index=False).mean()
data[['client', 'loan']].groupby(['loan'], as_index=False).mean()
g = sns.PairGrid(data, x_vars=['loan', 'default', 'housing'], y_vars='balance', hue='client')
g.map(sns.pointplot)
g.add_legend()
g = sns.FacetGrid(col='client', data=data)
g.map(plt.hist, 'duration', bins=20)
g.set(xlim=(0, 2000))
sns.factorplot(x='month', y='day', hue='client', col='contact', data=data)
g = sns.FacetGrid(data, col='client')
g.map(plt.hist, 'campaign', bins=25)
g.set(xlim=(0, 30))
sns.factorplot(x='contact', y='campaign', hue='client', data=data)
plt.figure(figsize=(8, 5))
sns.scatterplot(x='pdays', y='previous', hue='poutcome', data=data)
plt.show()
plt.figure(figsize=(10, 7))
sns.countplot('previous', data=data)
data.loc[data['previous'] > 250, 'previous'] = np.nan
data['previous'] = data['previous'].fillna(data['previous'].mode()[0])
plt.figure(figsize=(8, 5))
sns.scatterplot(x='pdays', y='previous', hue='poutcome', data=data)
plt.show()
data.describe()
data['age_band'] = pd.cut(data['age'], 10)
data[['age_band', 'client']].groupby(['age_band'], as_index=False).mean().sort_values(by='age_band', ascending=False)
data.loc[data['age'] < 25.7, 'age'] = 0
data.loc[(data['age'] > 25.7) & (data['age'] < 33.4), 'age'] = 1
data.loc[(data['age'] > 33.4) & (data['age'] < 41.1), 'age'] = 2
data.loc[(data['age'] > 41.1) & (data['age'] < 48.8), 'age'] = 3
data.loc[(data['age'] > 48.8) & (data['age'] < 56.5), 'age'] = 4
data.loc[(data['age'] > 56.5) & (data['age'] < 64.2), 'age'] = 5
data.loc[(data['age'] > 64.2) & (data['age'] < 71.9), 'age'] = 6
data.loc[(data['age'] > 71.9) & (data['age'] < 79.6), 'age'] = 7
data.loc[(data['age'] > 79.6) & (data['age'] < 87.3), 'age'] = 8
data.loc[data['age'] > 87.3, 'age'] = 9
data['balance_band'] = pd.cut(data['balance'], 10)
data[['balance_band', 'client']].groupby(['balance_band'], as_index=False).mean().sort_values(by='balance_band', ascending=False)
data.loc[data['balance'] < 2995.6, 'balance'] = 0
data.loc[(data['balance'] > 2995.6) & (data['balance'] < 14010.2), 'balance'] = 1
data.loc[(data['balance'] > 14010.2) & (data['balance'] < 25024.8), 'balance'] = 2
data.loc[(data['balance'] > 25024.8) & (data['balance'] < 36039.4), 'balance'] = 3
data.loc[(data['balance'] > 36039.4) & (data['balance'] < 47054), 'balance'] = 4
data.loc[(data['balance'] > 47054) & (data['balance'] < 58068.6), 'balance'] = 5
data.loc[(data['balance'] > 58068.6) & (data['balance'] < 69083.2), 'balance'] = 6
data.loc[(data['balance'] > 69083.2) & (data['balance'] < 80097.8), 'balance'] = 7
data.loc[(data['balance'] > 80097.8) & (data['balance'] < 91112.4), 'balance'] = 8
data.loc[data['balance'] > 91112.4, 'balance'] = 9
data['day_band'] = pd.cut(data['day'], 5)
data[['day_band', 'client']].groupby(['day_band'], as_index=False).mean().sort_values('day_band', ascending=False)
data.loc[data['day'] <= 7, 'day'] = 0
data.loc[(data['day'] > 7) & (data['day'] <= 13), 'day'] = 1
data.loc[(data['day'] > 13) & (data['day'] <= 19), 'day'] = 2
data.loc[(data['day'] > 19) & (data['day'] <= 25), 'day'] = 3
data.loc[data['day'] > 25, 'day'] = 4
data['duration_band'] = pd.cut(data['duration'], 5)
data[['duration_band', 'client']].groupby(['duration_band'], as_index=False).mean().sort_values('duration_band', ascending=False)
data.loc[data['duration'] < 983.6, 'duration'] = 0
data.loc[(data['duration'] > 983.6) & (data['duration'] < 1967.2), 'duration'] = 1
data.loc[(data['duration'] > 1967.2) & (data['duration'] < 2950.8), 'duration'] = 2
data.loc[(data['duration'] > 2950.8) & (data['duration'] < 3934.4), 'duration'] = 3
data.loc[data['duration'] > 3934.4, 'duration'] = 4
data['campaign_band'] = pd.cut(data['campaign'], 6)
data[['campaign_band', 'client']].groupby(['campaign_band'], as_index=False).mean().sort_values('campaign_band', ascending=False)
data.loc[data['campaign'] < 11.3, 'campaign'] = 0
data.loc[(data['campaign'] > 11.3) & (data['campaign'] < 21.6), 'campaign'] = 1
data.loc[(data['campaign'] > 21.6) & (data['campaign'] <= 32), 'campaign'] = 2
data.loc[(data['campaign'] > 32) & (data['campaign'] < 42.3), 'campaign'] = 3
data.loc[(data['campaign'] > 42.3) & (data['campaign'] < 52.6), 'campaign'] = 4
data.loc[data['campaign'] > 52.6, 'campaign'] = 5
data['pdays_band'] = pd.cut(data['pdays'], 9)
data[['pdays_band', 'client']].groupby(['pdays_band'], as_index=False).mean().sort_values('pdays_band', ascending=False)
data.loc[data['pdays'] == -1, 'pdays'] = 0
data.loc[(data['pdays'] > 0) & (data['pdays'] < 95.8), 'pdays'] = 1
data.loc[(data['pdays'] > 95.8) & (data['pdays'] < 95.8), 'pdays'] = 2
data.loc[(data['pdays'] > 192.7) & (data['pdays'] < 289.6), 'pdays'] = 3
data.loc[(data['pdays'] > 289.6) & (data['pdays'] < 386.5), 'pdays'] = 4
data.loc[(data['pdays'] > 386.5) & (data['pdays'] < 483.4), 'pdays'] = 5
data.loc[(data['pdays'] > 483.4) & (data['pdays'] < 580.3), 'pdays'] = 6
data.loc[(data['pdays'] > 580.3) & (data['pdays'] < 677.2), 'pdays'] = 7
data.loc[(data['pdays'] > 677.2) & (data['pdays'] < 774.1), 'pdays'] = 8
data.loc[data['pdays'] > 774.1, 'pdays'] = 9
data['previous_band'] = pd.cut(data['previous'], 6)
data[['previous_band', 'client']].groupby(['previous_band'], as_index=False).mean().sort_values('previous_band', ascending=False)
data.loc[data['previous'] < 9.6, 'previous'] = 0
data.loc[(data['previous'] > 9.6) & (data['previous'] < 19.3), 'previous'] = 1
data.loc[(data['previous'] > 19.3) & (data['previous'] <= 29), 'previous'] = 2
data.loc[(data['previous'] > 29) & (data['previous'] < 38.6), 'previous'] = 3
data.loc[(data['previous'] > 38.6) & (data['previous'] < 48.3), 'previous'] = 4
data.loc[data['previous'] > 48.3, 'previous'] = 5
data.drop(['age_band', 'balance_band', 'day_band', 'duration_band', 'campaign_band', 'pdays_band', 'previous_band'], axis=1, inplace=True)
data['previous'] = data['previous'].astype(int)
data.head()
cols = data.dtypes[data.dtypes == 'O'].index
cols
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
for col in cols:
    data[col] = label.fit(data[col]).transform(data[col])
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
import xgboost as xgb
target = 'client'
train = data.drop(['client'], axis=1)
test = data['client']
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=.3)
log = LogisticRegression()
log.fit(X_train, y_train)
y_pred = log.predict(X_test)
score = 100*accuracy_score(y_test, y_pred)
score
svm = SVC()
svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)
score = 100*accuracy_score(y_test, y_pred)
score
model_xgb = xgb.XGBClassifier(
    learning_rate = 0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=10)
model_xgb.fit(X_train, y_train)
y_pred = model_xgb.predict(X_test)
score = 100*accuracy_score(y_test, y_pred)
score
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
score = 100*accuracy_score(y_test, y_pred)
score
train.head()
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
score = 100*accuracy_score(y_test, y_pred)
score
gboost = GradientBoostingClassifier()
gboost.fit(X_train, y_train)
y_pred = gboost.predict(X_test)
score = 100*accuracy_score(y_test, y_pred)
score
def modelfit(model, train, predictors, useTrainCV=True, cv_folds = 5, early_stopping_rounds=100):
    if useTrainCV:
        model_params = model.get_xgb_params()
        xgbtrain = xgb.DMatrix(train[predictors].values, label=train[target].values)
        cvresult = xgb.cv(model_params, xgbtrain, num_boost_round=model.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds, metrics='auc', verbose_eval=False)
        model.set_params(n_estimators=cvresult.shape[0])
    # fit model
    model.fit(train[predictors], train[target], eval_metric='auc')
    # predict
    predictions = model.predict(train[predictors])
    probs = model.predict_proba(train[predictors])[:, 1]
    # print report model
    print('Model report:')
    print('Accuracy: {}'.format(metrics.accuracy_score(train[target].values, predictions)))
    print('AUC score: {}'.format(metrics.roc_auc_score(train[target], probs)))
    # plot feature importance
    index = train.drop([target], axis=1).columns
    feature_imp = pd.Series(data=model.feature_importances_,
                            index = index).sort_values(ascending=False)
    plt.figure(figsize=(16, 9))
    feature_imp.plot(kind='bar', color='blue', title='Feature importances')
    plt.ylabel('Score')
    plt.xlabel('Feature importance')
    
# predictors = [x for x in data.columns if x not in [target]]
# model = XGBClassifier(
#     learning_rate = 0.1,
#     n_estimators=1000,
#     max_depth=5,
#     min_child_weight=1,
#     gamma=0,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective='binary:logistic',
#     nthread=4,
#     scale_pos_weight=1,
#     seed=10)
# modelfit(model, data, predictors)
# params_test = {
#     'max_depth': [3, 4, 5, 6, 7, 8, 9],
#     'min_child_weight': [1, 2, 3, 4, 5, 6]
# }
# gsearch = GridSearchCV(estimator = XGBClassifier(
#     learning_rate = 0.1,
#     n_estimators=200,
#     max_depth=5,
#     min_child_weight=1,
#     gamma=0,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective='binary:logistic',
#     nthread=4,
#     scale_pos_weight=1,
#     seed=10), param_grid = params_test, scoring = 'roc_auc', n_jobs=4, iid=False, cv=5)
# gsearch.fit(data[predictors], data[target])
# gsearch.best_estimator_, gsearch.best_params_, gsearch.best_score_
adaboost = AdaBoostClassifier()
adaboost.fit(X_train, y_train)
y_pred = adaboost.predict(X_test)
score = 100*accuracy_score(y_test, y_pred)
score
# params_test = {
#     'max_depth': [3, 4, 5, 6, 7, 8, 9],
#     'n_estimators': [50, 100, 150, 200, 250]
# }
# model = XGBClassifier()
# #kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
# gsearch = GridSearchCV(model, params_test, scoring='roc_auc', n_jobs=-1, cv=5, verbose=1)
# result = gsearch.fit(data[predictors], data[target])
# result.best_score_, result.best_params_