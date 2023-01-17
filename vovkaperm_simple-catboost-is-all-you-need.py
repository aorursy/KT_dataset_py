import pandas as pd

import numpy as np

from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
train_df = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv', index_col='id')

test_df = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv', index_col='id')

y_true = train_df['target'].squeeze().copy()

idx = train_df.shape[0] # We should know where to cut train/test from full

full_df = pd.concat([train_df.drop('target', axis=1), test_df])

del train_df, test_df

full_df['n_of_mis_values'] = full_df.isna().sum(axis=1)

cat_feat_idxs = np.where(full_df.dtypes == 'object')[0]

num_feat_idxs = np.where(full_df.dtypes != 'object')[0]



bin_cols = [f'bin_{i}' for i in range(0,5)]

nom_cols = [f'nom_{i}' for i in range(0,10)]

ord_cols = [f'nom_{i}' for i in range(0,6)]



ctb = CatBoostClassifier(random_seed=17,

                         silent=True,

                         task_type="GPU",

                         border_count=254)



[full_df[col].fillna('MISSING_STRING', inplace=True) for col in full_df.columns[cat_feat_idxs]];

[full_df[col].fillna(0, inplace=True) for col in full_df.columns[num_feat_idxs]];
%%time

ctb.fit(full_df[:idx], y_true, cat_features=cat_feat_idxs)

y_pred = ctb.predict_proba(full_df[idx:])[:, 1]

submission = pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv', index_col='id')

submission['target'] = y_pred

submission.to_csv('catboost_submission.csv')