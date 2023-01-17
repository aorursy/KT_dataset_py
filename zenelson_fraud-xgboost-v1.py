# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn import preprocessing

import xgboost as xgb
path = '/kaggle/input/ieee-fraud-detection/'



# Importing data files



train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')



train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')
# Create dataframe combining ID and transaction information

    # Left merge combines all transaction data and only matching key ID information 

trn = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

tst = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
# Reducing size for testability

#trn = trn.iloc[:1000]

#tst = tst.iloc[:1000]
y_trn = trn['isFraud'].copy()



# Drop target

X_trn = trn.drop('isFraud', axis=1)

X_tst = tst.copy()
del trn, tst, train_transaction, train_identity, test_transaction, test_identity
# Finds features with more than 20% data missing

tot_rows = X_trn.shape[0]

empty_features = []

for col in X_trn.columns.values:

    num_empty = X_trn[X_trn[col].isnull()].shape[0]

    if num_empty / tot_rows > .4:

        empty_features.append(col)



# Drops columns

X_trn = X_trn.drop(columns = empty_features)

X_tst = X_tst.drop(columns = empty_features)
categorical_features = []

for col in X_trn.columns.values:

    if X_trn[col].dtype == 'object' or X_tst[col].dtype == 'object':

        categorical_features.append(col)



# Fills categorical features null entries

for col in categorical_features:

    X_trn[col].fillna(value = 'missing', inplace = True)

    X_tst[col].fillna(value = 'missing', inplace = True)



# Encodes categorical features numerically

# Label encoding for each string-type categorical feature

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

for col in categorical_features:

    X_trn[col] = le.fit_transform(X_trn[col])

    X_tst[col] = le.fit_transform(X_tst[col])



gc.collect()
from sklearn.impute import SimpleImputer



columns = X_trn.columns.values



imp_mean = SimpleImputer(strategy = 'mean')

imp_mean.fit(X_trn)

X_trn = pd.DataFrame(imp_mean.transform(X_trn), columns = columns)

X_tst = pd.DataFrame(imp_mean.transform(X_tst), columns = columns)
model = xgb.XGBClassifier(

    #booster = 'gbtree',

    objective = 'binary:logistic',

    n_estimators=300,

    max_depth=15,

    learning_rate=0.01,

    subsample=0.5,

    colsample_bytree=0.8,

    random_state=1,

    #tree_method='gpu_hist'

)
model.fit(X_trn, y_trn)
# Make predictions for test data

tst_preds = model.predict_proba(X_tst)[:,1]
# Use sample submission to find format for submission output

sample_submission = pd.read_csv(path + 'sample_submission.csv')



# Rewrite sample submission using tst_preds

sub1 = sample_submission

sub1['isFraud'] = tst_preds



# Save output to csv file

pd.DataFrame(sub1).to_csv('sub_predict', index = False)