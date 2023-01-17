import pandas as pd

import numpy as np

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_log_error as msle

from math import sqrt

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.info()
train.describe()
train.head()
target = train['SalePrice']

train.drop({'Id','SalePrice'}, axis = 1, inplace = True)

test.drop({'Id'}, axis = 1, inplace = True)

print(train.shape)

print(test.shape)
train_test = pd.concat([train, test])

dummies = pd.get_dummies(train_test, columns = train.columns, drop_first = True, sparse = True)

train_ohe = dummies.iloc[:train.shape[0], :]

test_ohe = dummies.iloc[train.shape[0]:, :]



print(train_ohe.shape)

print(test_ohe.shape)
train_pca = train_ohe

test_pca = test_ohe
train_ohe = train_ohe.sparse.to_coo().tocsr()

test_ohe = test_ohe.sparse.to_coo().tocsr()
pca = PCA(n_components = 2)

train_pca = pca.fit_transform(train_pca)

test_pca = pca.transform(test_pca)
def runRFR(train_X, train_y, test_X):

    model = RandomForestRegressor(n_estimators = 50, random_state = 100)

    modefl.fit(train_X, train_y)

    pred_test_y = model.predict(test_X)

    return pred_test_y
def rmsle(true_val, pred_val):

    return sqrt(msle(true_val, pred_val))
def run_cv_model(train, test, target, model_fn, eval_fn= None, params = {}, label = 'model'):

    kf = KFold(n_splits = 5)

    fold_splits = kf.split(train, target)

    cv_scores = []

    pred_full_test = 0

    pred_train = np.zeros((train.shape[0]))

    i = 1

    for whole_index, part_index in fold_splits:

        print('Started '+label+' fold '+str(i)+'/5')

        print('Train '+label)

        whole_train, part_train = train[whole_index], train[part_index]

        whole_target, part_target = target[whole_index], target[part_index]

        pred_train_part, pred_test_y = model_fn(whole_train, whole_target, part_train, part_target, test, params)

        pred_train[part_index] = pred_train_part

        pred_full_test = pred_full_test + pred_test_y

        if eval_fn is not None:

            cv_score = eval_fn(part_target, pred_train_part)

            cv_scores.append(cv_score)

            print(label + ' cv score {}: {}'.format(i, cv_score))

        i+=1

    print('{} cv score: {}'.format(label, cv_scores))

    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))

    print('{} cv std score : {}'.format(label, np.std(cv_scores)))

    pred_full_test = pred_full_test / 5

    results = {'train': pred_train,

               'test': pred_full_test,

               'cv scores': cv_scores,

               'cv mean score':np.mean(cv_scores),

               'label': label}

    return results
def runRF(train_X, train_y, test_X, test_y, test, params):

    model = RandomForestRegressor(**params)

    model.fit(train_X, train_y)

    print('Predict 1/2')

    pred_test_y = model.predict(test_X)

    print('Predict 2/2')

    pred_test_y2 = model.predict(test)

    return pred_test_y, pred_test_y2
def runDT(train_X, train_y, test_X, test_y, test, params):

    model = DecisionTreeRegressor(**params)

    model.fit(train_X, train_y)

    print('Predict 1/2')

    pred_test_y = model.predict(test_X)

    print('Predict 2/2')

    pred_test_y2 = model.predict(test)

    return pred_test_y, pred_test_y2
def runSVM(train_X, train_y, test_X, test_y, test, params):

    model = SVR(**params)

    model.fit(train_X, train_y)

    print('Predict 1/2')

    pred_test_y = model.predict(test_X)

    print('Predict 2/2')

    pred_test_y2 = model.predict(test)

    return pred_test_y, pred_test_y2
def runXGB(train_X, train_y, test_X, test_y, test, params):

    model = XGBRegressor(**params)

    model.fit(train_X, train_y)

    print('Predict 1/2')

    pred_test_y = model.predict(test_X)

    print('Predict 2/2')

    pred_test_y2 = model.predict(test)

    return pred_test_y, pred_test_y2
def switch(x):

    if x == 'RF':

        rf_params = {'n_estimators': 100, 'random_state': 100}

        results = run_cv_model(train_ohe, test_ohe, target, runRF, rmsle, rf_params, 'RF')            

    if x == 'SVM':

        svm_params = {'kernel':'rbf'}

        results = run_cv_model(train_ohe, test_ohe, target, runSVM, rmsle, svm_params, 'SVM')

    if x == 'DT':

        dt_params = {'random_state':0}

        results = run_cv_model(train_ohe, test_ohe, target, runDT, rmsle, dt_params, 'DT')

    if x == 'XGB':

        xgb_params = {'booster':'gbtree', 'random_state':0}

        results = run_cv_model(train_ohe, test_ohe, target, runXGB, rmsle, xgb_params, 'XGB')

    return results
model = ['RF', 'SVM', 'DT','XGB']

results = []



for x in model:

    r = switch(x)

    results.append(r)
submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission['SalePrice'] = results[3]['test']

submission.to_csv('submission.csv', index = False)
plt.scatter(test_pca[:, 0], test_pca[:, 1])