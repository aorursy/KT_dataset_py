#Importing libraries.



import os

import gc

import time

import datetime



import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



import matplotlib.pyplot as plt

import seaborn as sns



import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, GridSearchCV, train_test_split

from sklearn import metrics





#Reading the data.



train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
print('train_identity has {} columns'.format(len(train_identity.columns)))

train_identity.head()
print('train_transaction has {} columns'.format(len(train_transaction.columns)))

train_transaction.head()
print("There are {} fraudulent transactions out of {}".format(train_transaction['isFraud'].sum(),train_transaction.shape[0]))

_ = sns.countplot(x="isFraud", data=train_transaction)
identity_cols_withNA = 100*train_identity.isnull().sum().sort_values(ascending=False)/train_identity.shape[0]

identity_cols_withNA = identity_cols_withNA[identity_cols_withNA > 0]

print('train_identity has {} columns with null entries. The 10 most empty columns are shown below'.format(identity_cols_withNA.shape[0]))

identity_cols_withNA[:10]
transaction_cols_withNA = 100*train_transaction.isnull().sum().sort_values(ascending=False)/train_transaction.shape[0]

transaction_cols_withNA = transaction_cols_withNA[transaction_cols_withNA > 0]

print('train_transaction has {} columns with null entries. The 10 most empty columns are shown below'.format(transaction_cols_withNA.shape[0]))

transaction_cols_withNA[:10]
identity_cat_cols = ['DeviceType','DeviceInfo']

for i in range(12,39):

    identity_cat_cols.append('id_'+str(i))



transaction_cat_cols = ['ProductCD','addr1','addr2','P_emaildomain','R_emaildomain']

for i in range(1,7):

    transaction_cat_cols.append('card'+str(i))

    

for i in range(1,10):

    transaction_cat_cols.append('M'+str(i))
print('Number of unique entries per categorical column (including nulls) in train_identity:\n')

for col in identity_cat_cols:

    cat_number = len(train_identity[col].unique())

    print(col+': '+str(cat_number))
print('Number of unique entries per categorical column (including nulls) in train_transaction:\n')

for col in transaction_cat_cols:

    cat_number = len(train_transaction[col].unique())

    print(col+': '+str(cat_number))
train_identity_mem_usage = train_identity.memory_usage(index = True).sum()/(1024**2)

train_transaction_mem_usage = train_transaction.memory_usage(index = True).sum()/(1024**2)

test_identity_mem_usage = test_identity.memory_usage(index = True).sum()/(1024**2)

test_transaction_mem_usage = test_transaction.memory_usage(index = True).sum()/(1024**2)

print('Memory usage of train_identity is {:.2f} MB'.format(train_identity_mem_usage))

print('Memory usage of train_transaction is {:.2f} MB'.format(train_transaction_mem_usage))

print('Memory usage of test_identity is {:.2f} MB'.format(test_identity_mem_usage))

print('Memory usage of test_transaction is {:.2f} MB'.format(test_transaction_mem_usage))
def reduce_mem_usage(df,cat_cols):

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in cat_cols:

        df[col] = df[col].astype('category')

        

    num_cols = list(set(df.columns) - set(cat_cols))

    for col in num_cols:

        col_type = df[col].dtype        

        c_min = df[col].min()

        c_max = df[col].max()

        if str(col_type)[:3] == 'int':

            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                df[col] = df[col].astype(np.int8)

            elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:

                df[col] = df[col].astype(np.uint8)

            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                df[col] = df[col].astype(np.int16)

            elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:

                df[col] = df[col].astype(np.uint16)

            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                df[col] = df[col].astype(np.int32)

            elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:

                df[col] = df[col].astype(np.uint32)                    

            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                df[col] = df[col].astype(np.int64)

            elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:

                df[col] = df[col].astype(np.uint64)  

        else:

            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                df[col] = df[col].astype(np.float16)

            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                df[col] = df[col].astype(np.float32)

            else:

                df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
print('Reducing memory usage of train_identity\n')

train_identity = reduce_mem_usage(train_identity,identity_cat_cols)

print('\nReducing memory usage of train_transaction\n')

train_transaction = reduce_mem_usage(train_transaction,transaction_cat_cols)



train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

train_mem_usage = train.memory_usage(index = True).sum()/(1024**2)

print('\nMemory usage of train is {:.2f} MB'.format(train_mem_usage))

train.head()
cols_to_rename = {}

for i in range(10):    

    cols_to_rename['id-0'+str(i)] = 'id_0'+str(i)

for i in range(10,39):    

    cols_to_rename['id-'+str(i)] = 'id_'+str(i)

    

test_identity = test_identity.rename(columns=cols_to_rename)



print('Reducing memory usage of test_identity\n')

test_identity = reduce_mem_usage(test_identity,identity_cat_cols)

print('\nReducing memory usage of test_transaction\n')

test_transaction = reduce_mem_usage(test_transaction,transaction_cat_cols)



test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

test_mem_usage = test.memory_usage(index = True).sum()/(1024**2)

print('\nMemory usage of test is {:.2f} MB'.format(test_mem_usage))



del train_transaction

del train_identity

del test_identity

del test_transaction

gc.collect()

test.head()
#Step 1



cat_cols = transaction_cat_cols + identity_cat_cols

num_cols = list(set(train.drop(['TransactionID','isFraud','TransactionDT'],axis=1).columns) - set(cat_cols))



train_copy = train.copy()

test_copy = test.copy()

for col in cat_cols:

    le = LabelEncoder()

    le.fit(list(train_copy[col].astype(str).values) + list(test_copy[col].astype(str).values))

    train_copy[col] = le.transform(list(train_copy[col].astype(str).values))

    test_copy[col] = le.transform(list(test_copy[col].astype(str).values))



X = train_copy[cat_cols + num_cols]

y = train_copy['isFraud'] 

X_test = test_copy[cat_cols + num_cols]



del train_copy

del test_copy

gc.collect()
#Step 2

params = {'num_leaves': 256,

          'min_child_samples': 79,

          'objective': 'binary',

          'max_depth': 8,

          'learning_rate': 0.02,

          "boosting_type": "gbdt",

          "subsample_freq": 3,

          "subsample": 0.9,

          "bagging_seed": 11,

          "metric": 'auc',

          "verbosity": -1,

          'reg_alpha': 0.3,

          'reg_lambda': 0.3,

          'colsample_bytree': 0.9,

         }



#Step 3

n_splits = 4

skf = StratifiedKFold(n_splits=n_splits,shuffle=True)

skf.get_n_splits(X, y)



scores = []

i = 1

y_pred = np.zeros(len(X_test.iloc[:,0]))

feat_imp = pd.Series(0, index=X.columns)

for train_index, valid_index in skf.split(X, y):

    print('Training the model using fold number {}...'.format(i))

    X_train, X_valid = X.iloc[train_index,:], X.iloc[valid_index,:]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]



    model = lgb.LGBMClassifier(**params, n_estimators=500, n_jobs = -1)

    model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='auc', early_stopping_rounds=200,verbose=False)

    y_pred_valid = model.predict_proba(X_valid)[:, 1]

    y_pred += model.predict_proba(X_test)[:, 1]/n_splits

    score = metrics.roc_auc_score(y_valid, y_pred_valid)

    scores.append(score)

    feat_imp += pd.Series(model.feature_importances_, index=X.columns)/n_splits

    print('AUC score: {:.3f}'.format(score))

    i += 1

    

print('mean AUC score over the validation sets: {:.3f}\n'.format(np.mean(scores)))    

fig,ax = plt.subplots(figsize=(10,10))

_ = feat_imp.nlargest(20).sort_values().plot(kind='barh',ax=ax)

_ = ax.set_xlabel('Average importance (a.u.)')

_ = ax.set_ylabel('Features')
#Step 4

output = pd.DataFrame({'TransactionID': test.TransactionID, 'isFraud': y_pred})

output.to_csv('simple_lightgbm_submission.csv', index=False)

print("Submission successfully saved!")
#Cleaning the mess

del X

del X_test

del y

del X_train

del X_valid

del y_train

del y_valid

del y_pred

del y_pred_valid

gc.collect()
#Sorting the numerical columns by name

num_cols.sort()



# Create correlation matrix

corr_matrix = train[num_cols].corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]



print('Showing the first 15 elements of to_drop (out of {}):\n{}'.format(len(to_drop),to_drop[:15]))
to_drop_notV = to_drop[:14]

to_drop_V = to_drop[14:]



for col in to_drop_notV:

    na_proportion = train[col].isna().sum()/train.shape[0]

    if na_proportion >= 0.4:

        train = train.drop(col,axis=1)

        test = test.drop(col,axis=1)

        print('Column {} was dropped'.format(col))

        

train = train.drop(columns = to_drop_V)

test = test.drop(columns = to_drop_V)

print('\n{} V columns were dropped'.format(len(to_drop_V)))

#Excluding the removed features from the definition of num_cols

num_cols = list(set(num_cols) - set(['D2','D6','D7']) - set(to_drop_V))



train[cat_cols] = train[cat_cols].astype('str').replace('nan','unknown').astype('category')

test[cat_cols] = test[cat_cols].astype('str').replace('nan','unknown').astype('category')

train[num_cols] = train[num_cols].fillna(-999)

test[num_cols] = test[num_cols].fillna(-999)
print('Showing the first 10 unique elements of DeviceInfo (out of {}):\n{}'\

      .format(len(list(train['DeviceInfo'].unique())),\

              list(train['DeviceInfo'].unique())[:10]))
def setDevice(df):

    df['DeviceInfo'] = df['DeviceInfo'].str.lower()   

    df['device_name'] = df['DeviceInfo'].str.split('/', expand=True)[0]

    

    df.loc[df['device_name'].str.contains('sm', na=False), 'device_name'] = 'samsung'

    df.loc[df['device_name'].str.contains('samsung', na=False), 'device_name'] = 'samsung'

    df.loc[df['device_name'].str.contains('gt-', na=False), 'device_name'] = 'samsung'

    df.loc[df['device_name'].str.contains('moto', na=False), 'device_name'] = 'motorola'

    df.loc[df['device_name'].str.contains('lg', na=False), 'device_name'] = 'lg'

    df.loc[df['device_name'].str.contains('rv:', na=False), 'device_name'] = 'rv'

    df.loc[df['device_name'].str.contains('huawei', na=False), 'device_name'] = 'huawei'

    df.loc[df['device_name'].str.contains('ale-', na=False), 'device_name'] = 'huawei'

    df.loc[df['device_name'].str.contains('-l', na=False), 'device_name'] = 'huawei'

    df.loc[df['device_name'].str.contains('blade', na=False), 'device_name'] = 'zte'

    df.loc[df['device_name'].str.contains('linux', na=False), 'device_name'] = 'linux'

    df.loc[df['device_name'].str.contains('xt', na=False), 'device_name'] = 'sony'

    df.loc[df['device_name'].str.contains('htc', na=False), 'device_name'] = 'htc'

    df.loc[df['device_name'].str.contains('asus', na=False), 'device_name'] = 'asus'

    df.loc[df['device_name'].str.contains('lenovo', na=False), 'device_name'] = 'lenovo'

    df.loc[df['device_name'].str.contains('pixel', na=False), 'device_name'] = 'google'

    df.loc[df['device_name'].str.contains('redmi', na=False), 'device_name'] = 'xiaomi'

    df.loc[df['device_name'].str.contains('windows', na=False), 'device_name'] = 'microsoft'

    df.loc[df['device_name'].str.contains('microsoft', na=False), 'device_name'] = 'microsoft'

    df.loc[df['device_name'].str.contains('nexus', na=False), 'device_name'] = 'google'

    df.loc[df['device_name'].str.contains('ilium', na=False), 'device_name'] = 'lanix'

    df.loc[df['device_name'].str.contains('lumia', na=False), 'device_name'] = 'nokia'

    

    df['DeviceInfo'] = df['DeviceInfo'].astype('category')

    df['device_name'] = df['device_name'].astype('category')

    

    return df
train = setDevice(train)

test = setDevice(test)

print('The new column, device_name, has {} unique values'\

      .format(len(list(train['device_name'].unique()))))
print('Showing the first 10 unique elements of id_31 (out of {}):\n{}'\

      .format(len(list(train['id_31'].unique())),\

              list(train['id_31'].unique())[:10]))
train['id_31'] = train['id_31'].str.lower().astype('category')

test['id_31'] = test['id_31'].str.lower().astype('category')
print('Showing the first 10 unique elements of id_30 (out of {}):\n{}'\

      .format(len(list(train['id_30'].unique())),\

              list(train['id_30'].unique())[:10]))
train['id_30'] = train['id_30'].str.lower()

test['id_30'] = test['id_30'].str.lower()

train['id_30_word1'] = train['id_30'].str.split(' ', expand=True)[0].astype('category')

test['id_30_word1'] = test['id_30'].str.split(' ', expand=True)[0].astype('category')

train['id_30'] = train['id_30'].astype('category')

test['id_30'] = test['id_30'].astype('category')



print('The new column, id_30_word1, has {} unique values'\

      .format(len(list(train['id_30_word1'].unique()))))
print('Showing the unique elements of id_15:\n{}'\

      .format(list(train['id_15'].unique())))
train['id_15'] = train['id_15'].str.lower().astype('category')

test['id_15'] = test['id_15'].str.lower().astype('category')
print('Showing the first 10 unique elements of P_emaildomain (out of {}):\n{}'\

      .format(len(list(train['P_emaildomain'].unique())),\

              list(train['P_emaildomain'].unique())[:10]))
def SplitEmailDomain(df):

    

    df['P_emaildomain_prefix'] = df['P_emaildomain'].str.split('.',1, expand=True)[0].astype('category')

    df['P_emaildomain_sufix'] = df['P_emaildomain'].str.split('.',1, expand=True)[1].fillna('unknown').astype('category')

    df['R_emaildomain_prefix'] = df['R_emaildomain'].str.split('.',1, expand=True)[0].astype('category')

    df['R_emaildomain_sufix'] = df['R_emaildomain'].str.split('.',1, expand=True)[1].fillna('unknown').astype('category')

    

    return df
train = SplitEmailDomain(train)

test = SplitEmailDomain(test)



train.head()
print('Total number of days:',round((train.TransactionDT.max() - train.TransactionDT.min())/(24*60*60),3))
# NORMALIZE D COLUMNS

for i in range(1,16):

    if i not in [2,6,7]:

        train['D'+str(i)] = train['D'+str(i)] - train.TransactionDT/np.float32(24*60*60)

        test['D'+str(i)] = test['D'+str(i)] - test.TransactionDT/np.float32(24*60*60)
cat_cols = cat_cols + ['device_name','id_30_word1','P_emaildomain_prefix','P_emaildomain_sufix',\

                      'R_emaildomain_prefix','R_emaildomain_sufix']



train_copy = train.copy()

test_copy = test.copy()

for col in cat_cols:

    le = LabelEncoder()

    le.fit(list(train_copy[col].astype(str).values) + list(test_copy[col].astype(str).values))

    train_copy[col] = le.transform(list(train_copy[col].astype(str).values))

    test_copy[col] = le.transform(list(test_copy[col].astype(str).values))



first_day = train_copy['TransactionDT'].min()

five_months = first_day + 5*30*24*3600

X_train = train_copy.loc[train_copy['TransactionDT'] <= five_months,cat_cols + num_cols]

y_train = train_copy.loc[train_copy['TransactionDT'] <= five_months,'isFraud']

X_valid = train_copy.loc[train_copy['TransactionDT'] > five_months,cat_cols + num_cols]

y_valid = train_copy.loc[train_copy['TransactionDT'] > five_months,'isFraud'] 

X_test = test_copy[cat_cols + num_cols]



del train_copy

del test_copy

gc.collect()
params = {'num_leaves': 256,

          'min_child_samples': 79,

          'objective': 'binary',

          'max_depth': 8,

          'learning_rate': 0.02,

          "boosting_type": "gbdt",

          "subsample_freq": 3,

          "subsample": 0.9,

          "bagging_seed": 11,

          "metric": 'auc',

          'reg_alpha': 0.3,

          'reg_lambda': 0.3,

          'colsample_bytree': 0.9,

         }





model = lgb.LGBMClassifier(**params, n_estimators=2000, n_jobs = -1)

model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='auc', early_stopping_rounds=100,verbose=50)

y_pred_valid = model.predict_proba(X_valid)[:, 1]

y_pred = model.predict_proba(X_test)[:, 1]

score = metrics.roc_auc_score(y_valid, y_pred_valid)

feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)



print('AUC score: {:.3f}'.format(score))

  

fig,ax = plt.subplots(figsize=(10,10))

_ = feat_imp.nlargest(20).sort_values().plot(kind='barh',ax=ax)

_ = ax.set_xlabel('Average importance (a.u.)')

_ = ax.set_ylabel('Features')
output = pd.DataFrame({'TransactionID': test.TransactionID, 'isFraud': y_pred})

output.to_csv('lightgbm_submission.csv', index=False)

print("Submission successfully saved!")