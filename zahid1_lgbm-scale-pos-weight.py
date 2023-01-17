# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.
train = pd.read_parquet('../input/feature-engineering/train_eng.parquet')
# Parquet file format doesn't handle float16 so we will need to only re-type these columns

numerics = ['float32', 'float64']



def reduce_mem_usage(df, float_cols, verbose=True):

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in float_cols:

        col_type = df[col].dtypes

        if col_type == 'float32':

            df[col] = df[col].astype(np.float16)

            c_min = df[col].min()

            c_max = df[col].max()

        elif col_type == 'float64':

            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                df[col] = df[col].astype(np.float16)

            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                df[col] = df[col].astype(np.float32)

            else:

                pass  

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
float_cols = [col for col, dtype in train.dtypes.items() if dtype in numerics]
%%time

train = reduce_mem_usage(train, float_cols=float_cols)
train = train.sort_values(by=['TransactionDT'])
y = train['isFraud'].copy()

X = train.drop(columns=['TransactionDT', 'index', 'isFraud', 'TransactionID'])
params = {

          'num_leaves': 175,

          'feature_fraction': 0.5,

          'bagging_freq': 50,

          'bagging_fraction': 0.75,

          'min_data_in_leaf': 40,

          'objective': 'binary',

          'max_bin': 255,

          'max_depth': -1,

          'learning_rate': 0.02,

          'scale_pos_weight': 25,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": 'auc',

          "verbosity": -1,

          'random_state': 47

}
import lightgbm as lgb
predictions_all = {}
%%time

from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit



folds = TimeSeriesSplit(n_splits=5)

for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):

    if fold == 1:

        for weight in [1, 3, 15, 45, 85]:

            test_idx_3 = test_idx

            X_test = X.iloc[test_idx]

            y_test = y.iloc[test_idx]

            params['scale_pos_weight'] = weight

            print('Training on {:,} records'.format(len(trn_idx)))

            trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])

            val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])

            clf = lgb.train(params, trn_data, 400, 

                            valid_sets = [trn_data, val_data], 

                            verbose_eval=100)

            predictions = clf.predict(data=X_test)

            predictions_all[str(weight)] = predictions
import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, f1_score, roc_curve, average_precision_score
plt.figure(figsize=(15,12))

plt.subplots_adjust(left=0.25, right=1.25, top=1.5)

current = 1

for i, weight in enumerate([1, 3, 15, 45, 85]):

    preds = pd.DataFrame(data={'y': y.iloc[test_idx_3], 'preds': predictions_all[str(weight)]})

    f1 = f1_score(y_true=y.iloc[test_idx_3], y_pred=np.round(predictions_all[str(weight)]))

    auc = roc_auc_score(y_true=y.iloc[test_idx_3], y_score=predictions_all[str(weight)])

    recall = recall_score(y_true=y.iloc[test_idx_3], y_pred=np.round(predictions_all[str(weight)]))

    precision = precision_score(y_true=y.iloc[test_idx_3], y_pred=np.round(predictions_all[str(weight)]))

    avg_precision = average_precision_score(y_true=y.iloc[test_idx_3], y_score=predictions_all[str(weight)])

    

    plt.subplot(3, 2, current)

    current += 1

    plt.title(f'Scale: {weight}, AUC: {np.round(auc, 3)}, F1: {np.round(f1, 3)}, ' +

              f'\n Recall: {np.round(recall, 3)} Precision: {np.round(precision, 3)} Avg Precision: {np.round(avg_precision, 3)}')

    sns.distplot(preds[preds['y']==1]['preds'], label='1')

    sns.distplot(preds[preds['y']==0]['preds'], label='0')

    

plt.legend()