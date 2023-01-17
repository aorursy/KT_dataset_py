import pandas as pd

import numpy as np

from catboost import CatBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import OrdinalEncoder, StandardScaler

import category_encoders as ce
train_df = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv', index_col='id')

test_df = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv', index_col='id')

y_true = train_df['target'].squeeze().copy()

idx = train_df.shape[0] # We should know where to cut train/test from full

full_df = pd.concat([train_df.drop('target', axis=1), test_df])

del train_df, test_df

full_df['n_of_mis_values'] = full_df.isna().sum(axis=1)

full_df['%_of_mis_values'] = (full_df['n_of_mis_values']/(len(full_df.columns) - 1))

ctb_X = full_df.copy()



cat_feat_idxs = np.where(ctb_X.dtypes == 'object')[0]

num_feat_idxs = np.where(ctb_X.dtypes != 'object')[0]



bin_cols = [f'bin_{i}' for i in range(0,5)]

nom_cols = [f'nom_{i}' for i in range(0,10)]

ord_cols = [f'nom_{i}' for i in range(0,6)]



ctb = CatBoostClassifier(random_seed=17,

                         silent=True,

                         task_type="GPU",

                         border_count=254)



logit = LogisticRegression(random_state=2020, solver='lbfgs', max_iter=10000)

rfc = RandomForestClassifier(random_state=2020, n_estimators=200, max_depth=30)

scaler = StandardScaler()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)



[ctb_X[col].fillna('MISSING_STRING', inplace=True) for col in ctb_X.columns[cat_feat_idxs]];

[ctb_X[col].fillna(0, inplace=True) for col in ctb_X.columns[num_feat_idxs]];

full_df[bin_cols] = full_df[bin_cols].fillna(0)

full_df['day'] = full_df['day'].fillna(0)

full_df['month'] = full_df['month'].fillna(0)
full_df[bin_cols] = full_df[bin_cols].replace({'F': 0, 'T': 1, 'N': 0, 'Y': 1})
encoder = ce.TargetEncoder(cols=['nom_0', 'nom_1', 'nom_2',

       'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_0',

       'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5'], smoothing=0.2)



encoder.fit(full_df[:idx], y_true)



X = encoder.transform(full_df)

X = scaler.fit_transform(X)
%%time

models = [logit, rfc]



meta_mtrx = np.zeros(shape=(X[:idx].shape[0], len(models)))

meta_mtrx_test = np.zeros(shape=(X[idx:].shape[0], len(models)))



for i, model in enumerate(models):

    meta_mtrx[:, i] = cross_val_predict(model, X=X[:idx], y=y_true, cv=skf)

    model.fit(X[:idx], y_true)

    meta_mtrx_test[:, i] = model.predict(X[idx:])

meta_mtrx
meta_full = np.vstack([meta_mtrx, meta_mtrx_test])

ctb_X = pd.concat([ctb_X, pd.DataFrame(meta_full, columns=['meta_lr', 'meta_rfc'])], axis=1)
%%time

ctb.fit(ctb_X[:idx], y_true, cat_features=cat_feat_idxs)

y_pred = ctb.predict_proba(ctb_X[idx:])[:, 1]

submission = pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv', index_col='id')

submission['target'] = y_pred

submission.to_csv('catboost_submission.csv')