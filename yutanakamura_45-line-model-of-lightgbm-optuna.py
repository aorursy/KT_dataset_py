import numpy as np

import pandas as pd

import copy

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

import lightgbm as lgb

import optuna



def convert(df_input):

    df = copy.deepcopy(df_input)

    df.loc[:, 'bin_3':'bin_4'] = df.loc[:, 'bin_3':'bin_4'].applymap(lambda x: 'FTNY'.find(x) % 2)

    df = pd.get_dummies(df, columns=['nom_{}'.format(i) for i in range(5)])

    df.loc[:, 'nom_5':'nom_9'] = df.loc[:, 'nom_5':'nom_9'].applymap(lambda x: int(x, 16))

    df['ord_1'] = df['ord_1'].map(lambda x: ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'].index(x))

    df['ord_2'] = df['ord_2'].map(lambda x: ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot'].index(x)) 

    df['ord_3'] = df['ord_3'].map(lambda x: (ord(x) - ord('a')))

    df['ord_4'] = df['ord_4'].map(lambda x: (ord(x) - ord('A')))

    df['ord_6'] = df['ord_5'].map(lambda x: (ord(x[1]) - ord('A')))

    df['ord_5'] = df['ord_5'].map(lambda x: (ord(x[0]) - ord('A')))

    return df



df_train, df_test = (convert(pd.read_csv('../input/cat-in-the-dat/train.csv')), convert(pd.read_csv('../input/cat-in-the-dat/test.csv')))

X_train, X_valid, T_train, T_valid = train_test_split(df_train.drop(['id', 'target'], axis=1), df_train['target'], random_state=0)

X_test = df_test.drop('id', axis=1)

params_fixed = {'objective':'binary', 'metric':'binary_logloss', 'boosting_type':'gbdt', 'num_iterations':10000, 'early_stopping_round':10,\

                'max_depth':7, 'max_bin':255, 'reg_alpha':0., 'min_split_gain':0., 'learning_rate':0.01, 'random_state':0}

models = []



def objective(trial):

    global params_fixed, models

    params_tuning = {'num_leaves' : trial.suggest_int('num_leaves', 2, 100), \

                     'subsample' : trial.suggest_uniform('subsample', 0.5, 1.0), \

                     'subsample_freq' : trial.suggest_int('subsample_freq', 1, 20), \

                     'colsample_bytree' : trial.suggest_uniform('colsample_bytree', 0.01, 1.0), \

                     'min_child_samples' : trial.suggest_int('min_child_samples', 1, 50), \

                     'min_child_weight' : trial.suggest_loguniform('min_child_weight', 1e-3, 1e+1), \

                     'reg_lambda' : trial.suggest_loguniform('reg_lambda', 1e-2, 1e+3)}

    model = lgb.train({**params_fixed, **params_tuning}, lgb.Dataset(X_train, T_train), valid_sets=lgb.Dataset(X_valid, T_valid), verbose_eval=100)

    models.append(model)

    score = roc_auc_score(T_valid, model.predict(X_valid))

    return score



study = optuna.create_study(direction='maximize')

study.optimize(objective, n_trials=10)

pd.concat([df_test['id'], pd.Series(models[study.best_trial.number].predict(X_test), name='target')], axis=1).to_csv('submission.csv', index=False)