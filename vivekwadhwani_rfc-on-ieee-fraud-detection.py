# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import roc_auc_score
import time
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')
sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')
#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings            
            # Print current column type
#             print("******************************")
#             print("Column: ",col)
#             print("dtype before: ",df[col].dtype)            
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
#             print("min for this col: ",mn)
#             print("max for this col: ",mx)
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
            
            # Print new column type
#             print("dtype after: ",df[col].dtype)
#             print("******************************")
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist
print(train_transaction.shape)
print(test_transaction.shape)
# Reducing memory for transaction data set
train, NAlist = reduce_mem_usage(train_transaction)
# Reducing memory for test data set
test, NAlist = reduce_mem_usage(test_transaction)
# Memory comparison

# train_transaction.memory_usage().sum() / 1024**2 
# train.memory_usage().sum() / 1024**2 

# test_transaction.memory_usage().sum() / 1024**2 
# test.memory_usage().sum() / 1024**2 
del train_transaction, test_transaction
# Data description
# print(train.shape)
# print(test.shape)

print(train.head())
print(test.head())
# Manually pick 23 important features

# ignore "ProductCD"

features = ['isFraud','TransactionDT','TransactionAmt','card1','card2', 'card3', 'card5', 'addr1', 'addr2', 'dist1','C1', 'C2','C3', 'C4','C5', 'C6','C7', 'C8','C9', 'C10', 'C11', 'C12','C13', 'C14']
len(features)

print(train.shape)
print(test.shape)

train = train.loc[:,features]
features.remove('isFraud')
test = test.loc[:,features]

print(train.shape)
print(test.shape)
# Assigning X and y vairables

X_train = train.drop('isFraud',axis=1)
y_train = train['isFraud']

X_test = test
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.3, random_state=0)


print(X_train.shape)
print(X_val.shape)
## Reducing Memory again 
X_train, NAlist = reduce_mem_usage(X_train)
X_val, NAlist = reduce_mem_usage(X_val)
model = RandomForestClassifier(n_jobs=-1, n_estimators=200)
model.fit(X_train, y_train)

print(roc_auc_score(y_val,model.predict_proba(X_val)[:,1] ))
# creating a base sample submission

sample_submission['isFraud'] = model.predict_proba(X_test)[:,1]
sample_submission.to_csv("base_sample_submission.csv")
N = 10
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)

# create a dataframe
importances_df = pd.DataFrame({'variable':X_train.columns, 'importance': importances})

top_N = importances_df.sort_values(by=['importance'], ascending=False).head(N)

top_N