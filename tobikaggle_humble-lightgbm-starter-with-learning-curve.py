# -----------------------------------------------------------------------------

# LightGBM regression example

# __author__ = "DDgg"

# https://www.kaggle.com/c/mercedes-benz-greener-manufacturing

# -----------------------------------------------------------------------------

import numpy as np

import lightgbm as lgb

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

# PCA -------------------------------------------------------------------------

# add to see skewness

# -----------------------------------------------------------------------------

# data imnport 

# fork of forks from https://www.kaggle.com/jaybob20/starter-xgboost

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



#pca_3D_plot(test)



for c in train.columns:

    if train[c].dtype == 'object':

        lbl = LabelEncoder() 

        lbl.fit(list(train[c].values) + list(test[c].values)) 

        train[c] = lbl.transform(list(train[c].values))

        test[c] =  lbl.transform(list(test[c].values))

        

y_train = train["y"]

y_mean = np.mean(y_train)

train.drop('y', axis=1, inplace=True)
# split into training and validation set

# the data has a number of outliers, so the validation size needs

# to be large enough plus cross-validation is needed

X_train, X_valid, y_train, y_valid = train_test_split(

        train, y_train, test_size=0.2, random_state=12345)



# create dataset for lightgbm

lgb_train = lgb.Dataset(X_train, y_train)

lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)



# to record eval results for plotting

evals_result = {} 

# The rmse of prediction is: 7.75675312274

# specify your configurations as a dict

params = {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': {'l2'},

    'num_leaves': 20,

    'learning_rate': 0.05,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.8,

    'bagging_freq': 5,

    'min_data_in_leaf':4,

     #'min_sum_hessian_in_leaf': 5,

    'verbose':10

}



print('Start training...')



# train

gbm = lgb.train(params,

                lgb_train,

                num_boost_round=200,

                valid_sets=[lgb_train, lgb_valid],

                evals_result=evals_result,

                verbose_eval=10,

                early_stopping_rounds=50)



# print('\nSave model...')

# save model to file

# gbm.save_model('model.txt')

print('Start predicting...')

# predict

y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)



# eval rmse

print('\nThe rmse of prediction is:', mean_squared_error(y_valid, y_pred) ** 0.5)
# print feature names

print('\nFeature names:', gbm.feature_name())
print('\nCalculate feature importances...')



# feature importances

print('Feature importances:', list(gbm.feature_importance()))
# -------------------------------------------------------

print('Plot metrics during training...')

ax = lgb.plot_metric(evals_result, metric='l2')

plt.show()

# -------------------------------------------------------
print('Plot feature importances...')

ax = lgb.plot_importance(gbm, max_num_features=10)

plt.show()
print('\nPredicting test set...')

y_pred = gbm.predict(test, num_iteration=gbm.best_iteration)



# y_pred = model.predict(dtest)

output = pd.DataFrame({'id': test['ID'], 'y': y_pred})

output.to_csv('submit-lightgbm.csv', index=False)



print("Finished.")

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Grid search example // uncomment block if needed

# The parameters estimated here need to be fed into the upper code

# can also be rearranged for demo purposes only

# once this part is executed the ipython notebook has to be rerun

# in external pyton editor the code section can be executed alone



from sklearn.model_selection import GridSearchCV

estimator = lgb.LGBMRegressor()



# get possible parameters

estimator.get_params().keys()



# fill parameters ad libitum

param_grid = {

'num_leaves': [20, 30],    

'learning_rate': [0.01, 0.1],

#     'n_estimators': [],

#     'colsample_bytree' :[],

#     'min_split_gain' :[],

#     'subsample_for_bin' :[],

#     'max_depth' :[],

#     'subsample' :[], 

#     'reg_alpha' :[], 

#     'max_drop' :[], 

#     'gaussian_eta' :[], 

#     'drop_rate' :[], 

#     'silent' :[], 

#     'boosting_type' :[], 

#     'min_child_weight' :[], 

#     'skip_drop' :[], 

#     'learning_rate' :[], 

#     'fair_c' :[], 

#     'seed' :[], 

#     'poisson_max_delta_step' :[], 

#     'subsample_freq' :[], 

#     'max_bin' :[], 

#     'n_estimators' :[], 

#     'nthread' :[], 

#     'min_child_samples' :[], 

#     'huber_delta' :[], 

#     'use_missing' :[], 

#     'uniform_drop' :[], 

#     'reg_lambda' :[], 

#     'xgboost_dart_mode' :[], 

#     'objective'

}





gbm = GridSearchCV(estimator, param_grid)



gbm.fit(X_train, y_train)



# list them

print('Best parameters found by grid search are:', gbm.best_params_)
