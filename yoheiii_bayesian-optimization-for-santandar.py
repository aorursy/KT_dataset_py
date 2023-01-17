import pandas as pd

import numpy as np

import warnings

import time

warnings.filterwarnings("ignore")

import lightgbm as lgb

from bayes_opt import BayesianOptimization

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/train.csv')

train, extratrain = train_test_split(train, test_size=0.2, random_state=0)
X = train.drop('target', axis=1).drop('ID_code', axis=1)

y = train.target

def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=5, random_seed=6, n_estimators=10000, output_process=False):

    # prepare data

    train_data = lgb.Dataset(data=X, label=y, free_raw_data=False)

    # parameters

    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, min_split_gain, min_child_weight, learning_rate, num_threads, min_data_in_leaf, min_sum_hessian_in_leaf):

        # fixed parameters

        params = {'application':'binary',

                  'num_iterations': n_estimators,

                  'learning_rate':learning_rate,

                  'early_stopping_round':100,

                  'metric':'auc',

                  'max_depth':-1,

                  'bagging_freq':7,

                  'verbosity':-1}

        # variables

        params["num_leaves"] = int(round(num_leaves))

        params['feature_fraction'] = max(min(feature_fraction, 1), 0)

        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)

        params['learning_rate'] = learning_rate

        params['min_split_gain'] = min_split_gain

        params['min_child_weight'] = min_child_weight

        params['num_threads'] = int(num_threads)

        params['min_data_in_leaf'] = int(min_data_in_leaf)

        params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf

        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])

        return max(cv_result['auc-mean'])

    # range of variables

    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (22, 50),

                                            'feature_fraction': (0.01, 0.9),

                                            'bagging_fraction': (0.8, 1),

                                            'min_split_gain': (0.001, 0.1),

                                            'min_child_weight': (5, 50),

                                            'learning_rate': (0.001, 0.01),

                                            'num_threads': (6, 10),

                                            'min_data_in_leaf': (60, 100),

                                            'min_sum_hessian_in_leaf': (5.0 , 15.0)},

                                             random_state=0)

    # optimize!

    lgbBO.maximize(init_points=init_round, n_iter=opt_round)

    

    # output optimization process

    if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")

    

    # return best parameters

    return lgbBO.res

opt_params = bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=5, random_seed=6, n_estimators=100) #15&25