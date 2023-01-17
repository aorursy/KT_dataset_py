import os

os.listdir('../input/')
import dask.dataframe as dd

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd
import gc
pd.set_option('max_columns', 500)
import warnings

warnings.filterwarnings('ignore')
tr_trx = dd.read_csv('../input/train_transaction.csv', blocksize=25e6)

ts_trx = dd.read_csv('../input/test_transaction.csv', blocksize=25e6)
tr_id = dd.read_csv('../input/train_identity.csv', blocksize=25e6)

ts_id = dd.read_csv('../input/test_identity.csv', blocksize=25e6)
# tr_trx.head(npartitions=1)
# tr_id.head(npartitions=1)
tr = dd.merge(tr_trx, tr_id, on='TransactionID', how='inner')

ts = dd.merge(ts_trx, ts_id, on='TransactionID', how='inner')
del(tr_trx)

del(tr_id)

del(ts_id)

del(ts_trx)

gc.collect()
seed = 44
tr = tr.sample(frac=1, random_state=seed)
# tr['isFraud'].value_counts().compute()
# df_0 = tr[tr['isFraud'] == 0]

# df_1 = tr[tr['isFraud'] == 1]
# df_0_u = df_0.sample(frac = (20663/590540), random_state=seed)
# df = dd.concat([df_0_u, df_1], 0)
# df = df.persist()
train = tr.persist()
test = ts.persist()
del(tr)

del(ts)

gc.collect()
# df.compute().shape
train.compute().shape, test.compute().shape
# df['isFraud'].value_counts().compute()
train.index = train['TransactionID']

test.index = test['TransactionID']
train = train.drop(['TransactionID'], 1)

test = test.drop(['TransactionID'], 1)
train.head()
test.head()
target = train['isFraud']

features = train.drop(['isFraud'], 1)
data = dd.concat([features, test], 0)
cat_cols = data.select_dtypes('object').columns
data = dd.get_dummies(data.categorize(), cat_cols, drop_first=True)
# data.head()
data = data.drop(['TransactionDT'], 1)
train.compute().shape, test.compute().shape
features = data.compute().head(144233)

test_df = data.compute().tail(141907)
del(data)

del(train)

# del(test)

gc.collect()
features = features.fillna(features.mean())

test_df = test_df.fillna(test_df.mean())



features = features.fillna(0)

test_df = test_df.fillna(0)
from sklearn.decomposition import PCA



pca = PCA(n_components=100)

X = pca.fit_transform(features)
pca.explained_variance_ratio_
features.shape, target.compute().shape
from sklearn.model_selection import train_test_split



x, x_test, y, y_test = train_test_split(X, target.compute(), test_size=0.2, random_state=seed)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)
from imblearn.over_sampling import SMOTE



smt = SMOTE(random_state=seed)

x_train, y_train = smt.fit_sample(x_train, y_train)
# from imblearn.under_sampling import NearMiss



# nm = NearMiss(random_state=seed)

# x_train, y_train = nm.fit_sample(x_train, y_train)
import numpy as np

np.bincount(y_train)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from xgboost import XGBClassifier

from sklearn.multiclass import OneVsRestClassifier





model = RandomForestClassifier(random_state=seed)

model.fit(x_train, y_train)
y_val_pred = model.predict(x_val)

y_test_pred = model.predict(x_test)
print(model.score(x_val, y_val))

print(model.score(x_test, y_test))
from sklearn.metrics import confusion_matrix, f1_score



print(sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='g'))

print(f1_score(y_test, y_test_pred))
np.bincount(y_test)
test_features = pca.transform(test_df)
predictions = model.predict(test_features)
submission = pd.DataFrame({'TransactionID': test.compute().index, 'isFraud': predictions})
submission['isFraud'].unique()
submission.to_csv('submission.csv', index=False)