# !pip install --upgrade pip
# !pip3 install lightgbm
#### basic

import pandas as pd

import numpy as np



#### Visulization

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline



#### ML

import sklearn

from sklearn import tree

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit, train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.tree import DecisionTreeClassifier

import lightgbm as lgb

# import xgboost as xgb



#### Others

import datetime

import os, warnings, random

warnings.filterwarnings('ignore')
transaction_train = pd.read_csv('./datasets/train_transaction.csv',index_col='TransactionID')

transaction_test = pd.read_csv('./datasets/test_transaction.csv',index_col='TransactionID')

identity_train = pd.read_csv('./datasets/train_identity.csv',index_col='TransactionID')

identity_test = pd.read_csv('./datasets/test_identity.csv',index_col='TransactionID')
print('training set # for transaction: ' + str(len(transaction_train)))

transaction_train.head()
print('test set # for transaction: ' + str(len(transaction_test)))

transaction_test.head()
print('training set # for identity: ' + str(len(identity_train)))

identity_train.head(3)
train = pd.merge(transaction_train, identity_train, on='TransactionID', how='left',indicator = True)

test = pd.merge(transaction_test, identity_test, on='TransactionID', how='left',indicator = True)
train['_merge'].value_counts()
print(len(train))

train.head()
#helper functions



## Seeder

# :seed to make all processes deterministic 

def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)



## Memory Reducer                                     

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
SEED = 42

seed_everything(SEED)
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
#################################################################################

# Converting Strings to ints(or floats if nan in column) using frequency encoding

# We will be able to use these columns as category or as numerical feature

cat_cols = ['DeviceType', 'DeviceInfo', 'ProductCD', 

            'card1', 'card2', 'card3',  'card4','card5', 'card6','addr1', 'addr2']



for col in cat_cols:

    print('Encoding', col)

    temp_df = pd.concat([train[[col]], test[[col]]])

    col_encoded = temp_df[col].value_counts().to_dict()   

    train[col] = train[col].map(col_encoded)

    test[col]  = test[col].map(col_encoded)

    print(col_encoded)
#################################################################################

# Converting Strings to ints(or floats if nan in column) using frequency encoding

# for id information in indentity table

# encoding seperately



id_cols = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 

            'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',

            'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38']



for col in id_cols:

    print('Encoding', col)

    print('training set:')

    col_encoded = train[col].value_counts().to_dict()

    print(col_encoded)

    train[col] = train[col].map(col_encoded)

    

    print('test set:')

    col_encoded = test[col].value_counts().to_dict()

    test[col]  = test[col].map(col_encoded)

    print(col_encoded)
# M columns

#################################################################################

# Converting Strings to ints(or floats if nan in column)



for col in ['M1','M2','M3','M5','M6','M7','M8','M9']:

    train[col] = train[col].map({'T':1, 'F':0})

    test[col]  = test[col].map({'T':1, 'F':0})



for col in ['M4']:

    print('Encoding', col)

    temp_df = pd.concat([train[[col]], test[[col]]])

    col_encoded = temp_df[col].value_counts().to_dict()   

    train[col] = train[col].map(col_encoded)

    test[col]  = test[col].map(col_encoded)

    print(col_encoded)
#final minification

# train = reduce_mem_usage(train)

# test = reduce_mem_usage(test)
#export

train.to_pickle('train_mini.pkl')

test.to_pickle('test_mini.pkl')
train = pd.read_pickle('train_mini.pkl')

test = pd.read_pickle('test_mini.pkl')
base_columns = list(train) + list(identity_train)
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

for df in [train, test]:

    # Temporary

    df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

    df['DT_M'] = (df['DT'].dt.year-2017)*12 + df['DT'].dt.month

    df['DT_W'] = (df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear

    df['DT_D'] = (df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear

    

    df['Transaction_hour'] = df['DT'].dt.hour

    df['Transaction_day_of_week'] = df['DT'].dt.dayofweek

    df['DT_day'] = df['DT'].dt.day

    

    # D9 column

    df['D9'] = np.where(df['D9'].isna(),0,1)
######################################################

#calculate transaction amount by group

train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')

train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')



test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')

test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')

test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')

test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')



train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')

train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')

train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')

train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')



test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')

test['id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')

test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')

test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')



train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')

train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')

train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')

train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')



test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')

test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')

test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')

test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')



train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')

train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')



test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')

test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')


# New feature - decimal part of the transaction amount.

train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)

test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)



# New feature - day of week in which a transaction happened.

# train['Transaction_day_of_week'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1) % 7)

# test['Transaction_day_of_week'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)



# # New feature - hour of the day in which a transaction happened.

# train['Transaction_hour'] = np.floor(train['TransactionDT'] / 3600) % 24

# test['Transaction_hour'] = np.floor(test['TransactionDT'] / 3600) % 24



# Some arbitrary features interaction

for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 'P_emaildomain__C2', 

                'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:



    f1, f2 = feature.split('__')

    train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)

    test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)



    le = LabelEncoder()

    le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))

    train[feature] = le.transform(list(train[feature].astype(str).values))

    test[feature] = le.transform(list(test[feature].astype(str).values))
#new feature (8.21)

#add more transaction amount groupby

train['TransactionAmt_to_mean_card2'] = train['TransactionAmt'] / train.groupby(['card2'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_std_card2'] = train['TransactionAmt'] / train.groupby(['card2'])['TransactionAmt'].transform('std')



test['TransactionAmt_to_mean_card2'] = test['TransactionAmt'] / test.groupby(['card2'])['TransactionAmt'].transform('mean')

test['TransactionAmt_to_std_card2'] = test['TransactionAmt'] / test.groupby(['card2'])['TransactionAmt'].transform('std')

#new feature (8.21)

#add transaction amount groupby product type(productCD)

train['TransactionAmt_to_mean_product'] = train['TransactionAmt'] / train.groupby(['ProductCD'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_std_product'] = train['TransactionAmt'] / train.groupby(['ProductCD'])['TransactionAmt'].transform('std')



test['TransactionAmt_to_mean_product'] = test['TransactionAmt'] / test.groupby(['ProductCD'])['TransactionAmt'].transform('mean')

test['TransactionAmt_to_std_product'] = test['TransactionAmt'] / test.groupby(['ProductCD'])['TransactionAmt'].transform('std')
#new feature(8.21)

# New feature: max & min transaction amount by groups

train['TransactionAmt_to_max_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('max')

train['TransactionAmt_to_max_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('max')

train['TransactionAmt_to_max_card2'] = train['TransactionAmt'] / train.groupby(['card2'])['TransactionAmt'].transform('max')

train['TransactionAmt_to_min_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('min')

train['TransactionAmt_to_min_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('min')

train['TransactionAmt_to_min_card2'] = train['TransactionAmt'] / train.groupby(['card2'])['TransactionAmt'].transform('min')



test['TransactionAmt_to_max_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('max')

test['TransactionAmt_to_max_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('max')

test['TransactionAmt_to_max_card2'] = test['TransactionAmt'] / test.groupby(['card2'])['TransactionAmt'].transform('max')

test['TransactionAmt_to_min_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('min')

test['TransactionAmt_to_min_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('min')

test['TransactionAmt_to_min_card2'] = test['TransactionAmt'] / test.groupby(['card2'])['TransactionAmt'].transform('min')
# New feature - log of transaction amount. ()

train['TransactionAmt'] = np.log(train['TransactionAmt'])

test['TransactionAmt'] = np.log(test['TransactionAmt'])
#new feature(8.21)

#interaction between product type and transaction time, location info(address, distance)

for feature in ['Transaction_day_of_week__ProductCD', 'Transaction_hour__ProductCD', 

                'addr1__ProductCD', 'addr2__ProductCD', 'dist1__ProductCD', 'dist2__ProductCD']:



    f1, f2 = feature.split('__')

    train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)

    test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)



    le = LabelEncoder()

    le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))

    train[feature] = le.transform(list(train[feature].astype(str).values))

    test[feature] = le.transform(list(test[feature].astype(str).values))
#new feature(8.21)

#https://www.kaggle.com/nroman/eda-for-cis-fraud-detection#New-feature:-number-of-NaN's

#https://www.kaggle.com/c/ieee-fraud-detection/discussion/105130#latest-604661

#number of NaNs 

#V1 ~ V11

#V12 ~ V34

#V12 ~ V34

#V35 ~ V52

#V53 ~ V74

#V75 ~ V94

#V95 ~ V137

#V138 ~ V166 (high null ratio)

#V167 ~ V216 (high null ratio)

#V217 ~ V278 (high null ratio, 2 different null ratios)

#V279 ~ V321 (2 different null ratios)

#V322 ~ V339 (high null ratio)

#haven't figure out how to use this finding

train['Total_nulls'] = train.isnull().sum(axis=1)

test['Total_nulls'] = test.isnull().sum(axis=1)
#new feature(8.21)





########################### 'P_emaildomain' - 'R_emaildomain'

p = 'P_emaildomain'

r = 'R_emaildomain'

uknown = 'email_not_provided'



for df in [train, test]:

    df[p] = df[p].fillna(uknown)

    df[r] = df[r].fillna(uknown)

    

    # Check if P_emaildomain matches R_emaildomain

    df['email_check'] = np.where((df[p]==df[r])&(df[p]!=uknown),1,0)



    

# https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499

# bin email address

# do not use frequency encoding for email before

emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 

          'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 

          'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 

          'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 

          'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 

          'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 

          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 

          'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 

          'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 

          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 

          'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 

          'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 

          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 

          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 

          'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

us_emails = ['gmail', 'net', 'edu']





for c in ['P_emaildomain', 'R_emaildomain']:

    train[c + '_bin'] = train[c].map(emails)

    test[c + '_bin'] = test[c].map(emails)

    

    train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])

    test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])

    

    train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')



for col in ['P_emaildomain', 'R_emaildomain','P_emaildomain_bin', 'R_emaildomain_bin',

            'P_emaildomain_suffix', 'R_emaildomain_suffix']:

    print('Encoding', col)

    temp_df = pd.concat([train[[col]], test[[col]]])

    col_encoded = temp_df[col].value_counts().to_dict()   

    train[col] = train[col].map(col_encoded)

    test[col]  = test[col].map(col_encoded)

    print(col_encoded)
train.head()
# #############8.24 

# ########################### Model Features 

# ## We can use set().difference() but the order matters

# ## Matters only for deterministic results

# ## In case of remove() we will not change order

# ## even if variable will be renamed

# ## please see this link to see how set is ordered

# ## https://stackoverflow.com/questions/12165200/order-of-unordered-python-sets

# TARGET = 'isFraud'

# rm_cols = [

#     'TransactionID','TransactionDT', # These columns are pure noise right now

#      TARGET,                         # Not target in features))

#     'uid','uid2','uid3',             # Our new client uID -> very noisy data

#     'bank_type',                     # Victims bank could differ by time

#     'DT','DT_M','DT_W','DT_D',       # Temporary Variables

#     'DT_hour','DT_day_week','DT_day',

#     'DT_D_total','DT_W_total','DT_M_total',

#     'id_30','id_31','id_33',

# ]





# ########################### Features elimination 

# from scipy.stats import ks_2samp

# features_check = []

# columns_to_check = set(list(train)).difference(rm_cols)

# for i in columns_to_check:

#     features_check.append(ks_2samp(test[i], train[i])[1])



# features_check = pd.Series(features_check, index=columns_to_check).sort_values() 

# features_discard = list(features_check[features_check==0].index)

# print(features_discard)



# # We will reset this list for now (use local test drop),

# # Good droping will be in other kernels

# # with better checking

# features_discard = [] 



# # Final features list

# features_columns = [col for col in list(train_df) if col not in rm_cols + features_discard]
for column in train:

    total = len(train)

    print('{0} : {1}'.format(column, train[column].isnull().sum()/total))
def drop_sparse_column(threshold, df_train, df_test):

    new_train = df_train.copy()

    new_test = df_test.copy()

    total = len(df_train)

    for column in df_train:

        percent = df_train[column].isnull().sum()/total

        if percent > threshold:

            new_train = new_train.drop(columns = [column])

            new_test = new_test.drop(columns = [column])

    return new_train, new_test
train, test = drop_sparse_column(0.97, train,test)
def drop_single_dominant(threshold, df_train, df_test):

    new_train = df_train.copy()

    isfraud = new_train['isFraud']

    new_train = new_train.drop(columns = ['isFraud'])

    new_test = df_test.copy()

    for column in new_train:

        if train[column].value_counts(dropna = False, normalize = True).values[0] > threshold:

            new_train = new_train.drop(columns = [column])

            new_test = new_test.drop(columns = [column])

    new_train['isFraud'] = isfraud

    return new_train, new_test
train, test = drop_single_dominant(0.97, train, test)
def drop_one_value(df_train, df_test):

    new_train = df_train.copy()

    new_test = df_test.copy()

    for column in new_train:

        if train[column].nunique() <= 1:

            new_train = new_train.drop(columns = [column])

            new_test = new_test.drop(columns = [column])

    return new_train, new_test
train, test = drop_one_value(train, test)
train.head()
useful_features = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1',

                   'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',

                   'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M2', 'M3',

                   'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V17',

                   'V19', 'V20', 'V29', 'V30', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V40', 'V44', 'V45', 'V46', 'V47', 'V48',

                   'V49', 'V51', 'V52', 'V53', 'V54', 'V56', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V69', 'V70', 'V71',

                   'V72', 'V73', 'V74', 'V75', 'V76', 'V78', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V87', 'V90', 'V91', 'V92',

                   'V93', 'V94', 'V95', 'V96', 'V97', 'V99', 'V100', 'V126', 'V127', 'V128', 'V130', 'V131', 'V138', 'V139', 'V140',

                   'V143', 'V145', 'V146', 'V147', 'V149', 'V150', 'V151', 'V152', 'V154', 'V156', 'V158', 'V159', 'V160', 'V161',

                   'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V169', 'V170', 'V171', 'V172', 'V173', 'V175', 'V176', 'V177',

                   'V178', 'V180', 'V182', 'V184', 'V187', 'V188', 'V189', 'V195', 'V197', 'V200', 'V201', 'V202', 'V203', 'V204',

                   'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V219', 'V220',

                   'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V231', 'V233', 'V234', 'V238', 'V239',

                   'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V249', 'V251', 'V253', 'V256', 'V257', 'V258', 'V259', 'V261',

                   'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276',

                   'V277', 'V278', 'V279', 'V280', 'V282', 'V283', 'V285', 'V287', 'V288', 'V289', 'V291', 'V292', 'V294', 'V303',

                   'V304', 'V306', 'V307', 'V308', 'V310', 'V312', 'V313', 'V314', 'V315', 'V317', 'V322', 'V323', 'V324', 'V326',

                   'V329', 'V331', 'V332', 'V333', 'V335', 'V336', 'V338', 'id_01', 'id_02', 'id_03', 'id_05', 'id_06', 'id_09',

                   'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_17', 'id_19', 'id_20', 'id_30', 'id_31', 'id_32', 'id_33',

                   'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'device_name', 'device_version', 'OS_id_30', 'version_id_30',

                   'browser_id_31', 'version_id_31', 'screen_width', 'screen_height', 'had_id']
for col in train:

    if col not in useful_features:

        print("'{}'".format(col)+',',end='')
####### Random Sampling

#train = train.sample(n=10000, replace=True, random_state=1)

#test = test.sample(n=1000, replace=True, random_state=2)
### Final feature selection
train_X = train.drop(['TransactionDT','isFraud','_merge'],axis=1)

train_Y = train['isFraud']

test_X = test.drop(['TransactionDT','_merge'],axis=1)
train_X = train_X.drop(['DT'],axis=1)

test_X = test_X.drop(['DT'],axis=1)
# train_X = train_X.drop(['DeviceInfo_device', 'DeviceInfo_version', 'id_30_device', 'id_30_version', 'id_31_device'],axis=1)

# test_X = test_X.drop(['DeviceInfo_device', 'DeviceInfo_version', 'id_30_device', 'id_30_version', 'id_31_device'],axis=1)
train_X = train_X.drop(['DT_D'],axis=1)

test_X = test_X.drop(['DT_D'],axis=1)

train_X.head()
train_X.to_csv('train_FE.csv',index=False)

test_X.to_csv('test_FE.csv',index=False)
train_Y.to_csv('train_FE_Y_3.csv',index=False,header=False)
len(train_Y)
#use new_train & new_test to train model



# Set Parameters

# params = {'num_leaves': 2**8,

#           'min_child_weight': 0.03454472573214212,

#           'feature_fraction': 0.3797454081646243,

#           'bagging_fraction': 0.4181193142567742,

#           'min_data_in_leaf': 106,

#           'objective': 'binary',

#           'max_depth': -1,

#           'learning_rate': 0.01,

#           "boosting_type": "gbdt",

#           "bagging_seed": 11,

#           "metric": 'auc',

#           "verbosity": -1,

#           'reg_alpha': 0.3899927210061127,

#           'reg_lambda': 0.6485237330340494,

#           'random_state': 47,

#           'colsample_bytree': 0.7,

#           'n_estimators':800,

#           'max_bin':255

          

#          }



params = {'num_leaves': 2**8,

          'min_child_weight': 0.03454472573214212,

          'feature_fraction': 0.3797454081646243,

          'bagging_fraction': 0.4181193142567742,

          'min_data_in_leaf': 106,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.01,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": 'auc',

          "verbosity": -1,

          'reg_alpha': 0.3899927210061127,

          'reg_lambda': 0.6485237330340494,

          'random_state': 47,

         }

# lgb_params = {

#                     'objective':'binary',

#                     'boosting_type':'gbdt',

#                     'metric':'auc',

#                     'n_jobs':-1,

#                     'learning_rate':0.01,

#                     'num_leaves': 2**8,

#                     'max_depth':-1,

#                     'tree_learner':'serial',

#                     'colsample_bytree': 0.7,

#                     'subsample_freq':1,

#                     'subsample':0.7,

#                     'n_estimators':800,

#                     'max_bin':255,

#                     'verbose':-1,

#                     'seed': SEED,

#                     'early_stopping_rounds':100, 

#                 }
%%time

import gc







##### Cross Validation

NFOLDS = 5

folds = KFold(n_splits=NFOLDS,random_state=42)



columns = train_X.columns

splits = folds.split(train_X, train_Y)



y_pred_test_vectors = np.zeros(test_X.shape[0])

y_pred_valid_vectors = np.zeros(train_X.shape[0])

score = 0





feature_importances = pd.DataFrame()

feature_importances['feature'] = columns

  

for fold_n, (train_index, valid_index) in enumerate(splits):

    X_train, X_valid = train_X[columns].iloc[train_index], train_X[columns].iloc[valid_index]

    y_train, y_valid = train_Y.iloc[train_index], train_Y.iloc[valid_index]

    

    dtrain = lgb.Dataset(X_train, label=y_train)

    dvalid = lgb.Dataset(X_valid, label=y_valid)



    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)

    

    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

    

    y_pred_valid = clf.predict(X_valid)

    y_pred_valid_vectors[valid_index] = y_pred_valid

    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")

    

    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS

    y_pred_test_vectors += clf.predict(test_X) / NFOLDS

    

    del X_train, X_valid, y_train, y_valid

    

    #gabage collector

    gc.collect()

    

print(f"\nMean AUC = {score}")

print(f"Out of folds AUC = {roc_auc_score(train_Y, y_pred_valid_vectors)}")
pred = pd.read_csv('./datasets/sample_submission.csv')



pred['isFraud'] = y_pred_test_vectors

pred.to_csv("submission_5.csv", index=False)
feature_importances['average'] = feature_importances.mean(axis = 1)

feature_importances
feature_importances['average'] = feature_importances.mean(axis = 1)

plt.figure(figsize=(16, 16))

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');

plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));
# the final submission
