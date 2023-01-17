print('loading libs...')

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb

import warnings

warnings.filterwarnings("ignore")

import os

import gc

import datetime

import time

from tqdm import tqdm

from scipy import stats

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

print('done')
# loading the funcs
# func for loading data

def DataLoading(path, df_name):

    files = os.listdir(f'{path}')

    for i in range(len(files)):

        s0 = files[i]

        s1 = files[i][:-4]

        s2 = files[i][-4:]

        if s2 =='.csv':

            print('loading:'+ s1 + '...')

            globals()[s1] = pd.read_csv(f'{path}'+ s0)

            df_name.append(s1)

        elif s2 == '.pkl':

            print('loading:'+ s1 + '...')

            globals()[s1] = pd.read_pickle(f'{path}'+ s0)

            df_name.append(s1)

        else:

            pass

    print('successfully loading: ')

    print(df_name)

    print('done')

    return df_name



                    

# func for training the data using LGB

def DataTraining(df1, df2, df3):

    folds = KFold(n_splits=NFOLDS)

    columns = df1.columns

    splits = folds.split(df1, df2)

    y_preds = np.zeros(df3.shape[0])

    y_oof = np.zeros(df1.shape[0])

    score = 0

  

    for fold_n, (train_index, valid_index) in enumerate(splits):

        X_tr, X_val = df1[columns].iloc[train_index], df1[columns].iloc[valid_index]

        y_tr, y_val = df2.iloc[train_index], df2.iloc[valid_index]    

        dtrain = lgb.Dataset(X_tr, label=y_tr)

        dvalid = lgb.Dataset(X_val, label=y_val)

        clf = lgb.train(LGB_PARAMS, dtrain,  valid_sets = [dtrain, dvalid], verbose_eval=50)        

        y_pred_valid = clf.predict(X_val)

        y_oof[valid_index] = y_pred_valid

        print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_val, y_pred_valid)}")   

        score += roc_auc_score(y_val, y_pred_valid) / NFOLDS

        del X_tr, X_val, y_tr, y_val

        gc.collect() 

        y_preds += clf.predict(df3) / NFOLDS       

    print(f"\nMean AUC = {score}")

    print(f"Out of folds AUC = {roc_auc_score(df2, y_oof)}")

    return y_preds

   

                    

# func for result submission

def ResultSubmitting(res):

    print('submission...')

    sample_submission['isFraud'] = res

    sample_submission.to_csv("submission_lgb.csv", index=False)

    print('done')

# setting the params

DF_NAME= []

PATH = '../input/data-preparation/'

NFOLDS = 5

LGB_PARAMS = {

          'objective':'binary',

          'boosting_type':'gbdt',

          'metric':'auc',

          'n_jobs':-1,

          'max_depth':-1,

          'tree_learner':'serial',

          'min_data_in_leaf':30,

          'n_estimators':2000,

          'max_bin':255,

          'verbose':-1,

          'seed': 1229,

          'learning_rate': 0.01,

          'early_stopping_rounds':200,

          'colsample_bytree': 0.5,          

          'num_leaves': 256, 

          'reg_alpha': 0.35, 

         }
# loading data

df_name = DataLoading(PATH, DF_NAME)
# training data

y_preds = DataTraining(X_train, y_train, X_test)
# submitting result

ResultSubmitting(y_preds)