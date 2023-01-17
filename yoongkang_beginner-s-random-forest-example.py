# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_transaction_df = pd.read_csv('/kaggle/input/train_transaction.csv')

train_identity_df = pd.read_csv('/kaggle/input/train_identity.csv')
train_transaction_df.head()
train_identity_df.head()
df_train = train_transaction_df.merge(train_identity_df, on='TransactionID', how='left')
df_train.shape
df_train.head().transpose()
df_train.memory_usage().sum()
# Get the categorical and numeric columns

cat_cols = [

    'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',

    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo',

] + [f'M{n}' for n in range(1, 10)] + [f'id_{n}' for n in range(12, 39)]

num_cols = list(set(df_train.columns) - set(cat_cols))
a = df_train[num_cols].isnull().any()

train_null_num_cols = a[a].index
nas = {}

for n in train_null_num_cols:

    df_train[f'{n}_isna'] = df_train[n].isnull()

    median = df_train[n].median()

    df_train[n].fillna(median, inplace=True)

    nas[n] = median
integer_cols = []

for c in num_cols:

    try:

        if df_train[c].fillna(-1.0).apply(float.is_integer).all():

            integer_cols += [c]

    except Exception as e:

        print("error: ", c, e)
stats = df_train[integer_cols].describe().transpose()

stats
int8columns = stats[stats['max'] < 256].index

print(int8columns.shape)

print(int8columns)

int16columns = stats[(stats['max'] >= 256) & (stats['max'] <= 32767)].index

print(int16columns.shape)

print(int16columns)
for c in int8columns:

    df_train[c] = df_train[c].astype('int8')

    

for c in int16columns:

    df_train[c] = df_train[c].astype('int16')
df_train.memory_usage().sum()
test_transaction_df = pd.read_csv('/kaggle/input/test_transaction.csv')

test_identity_df = pd.read_csv('/kaggle/input/test_identity.csv')

df_test = test_transaction_df.merge(test_identity_df, on='TransactionID', how='left')
for k, v in nas.items():

    df_test[f'{k}_isna'] = df_test[k].isnull()

    df_test[k].fillna(v, inplace=True)
test_num_cols = list(set(num_cols) - set(['isFraud']))

a = df_test[test_num_cols].isnull().any()

test_null_num_cols = a[a].index
for n in test_null_num_cols:

    df_test[n].fillna(df_train[n].median(), inplace=True)  # use the training set's median!
# copied from above cells



integer_cols = []

for c in test_num_cols:

    try:

        if df_test[c].fillna(-1.0).apply(float.is_integer).all():

            integer_cols += [c]

    except Exception as e:

        print("error: ", c, e)

stats = df_test[integer_cols].describe().transpose()

int8columns = stats[stats['max'] < 256].index

int16columns = stats[(stats['max'] >= 256) & (stats['max'] <= 32767)].index

for c in int8columns:

    df_test[c] = df_test[c].astype('int8')

    

for c in int16columns:

    df_test[c] = df_test[c].astype('int16')
for c in cat_cols:

    df_train[c] = df_train[c].fillna("missing")

    

for c in cat_cols:   

    df_test[c] = df_test[c].fillna("missing")
cats = {}

for c in cat_cols:

    df_train[c] = df_train[c].astype("category")

    df_train[c].cat.add_categories('unknown', inplace=True)

    cats[c] = df_train[c].cat.categories
for k, v in cats.items():

    df_test[k][~df_test[k].isin(v)] = 'unknown'
from pandas.api.types import CategoricalDtype



for k, v in cats.items():

    new_dtype = CategoricalDtype(categories=v, ordered=True)

    df_test[k] = df_test[k].astype(new_dtype)
for c in cat_cols:

    df_train[c] = df_train[c].cat.codes

    df_test[c] = df_test[c].cat.codes

    
df_train.to_feather('df_train')
df_test.to_feather('df_test')
idx = int(len(df_train) * 0.8)

training_set, validation_set = df_train[:idx], df_train[idx:]
y_train = training_set['isFraud']

X_train = training_set.drop('isFraud', axis=1)

y_valid = validation_set['isFraud']

X_valid = validation_set.drop('isFraud', axis=1)
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score
training_sample = training_set[-100000:]

y_train_sample = training_sample['isFraud']

X_train_sample = training_sample.drop('isFraud', axis=1)
model = RandomForestRegressor(

    n_estimators=400, max_features=0.3,

    min_samples_leaf=20, n_jobs=-1, verbose=1)
model.fit(X_train_sample, y_train_sample)
preds_valid = model.predict(X_valid)
roc_auc_score(y_valid, preds_valid)
model = RandomForestRegressor(

    n_estimators=400, max_features=0.3,

    min_samples_leaf=20, n_jobs=-1, verbose=1)
y = df_train['isFraud']

X = df_train.drop('isFraud', axis=1)
model.fit(X, y)
y_preds = model.predict(df_test)
submission = pd.read_csv('/kaggle/input/sample_submission.csv')

submission['isFraud'] = y_preds

submission.to_csv('submission.csv', index=False)