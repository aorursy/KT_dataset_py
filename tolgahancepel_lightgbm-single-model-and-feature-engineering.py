def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

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

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

from sklearn import preprocessing

import gc, datetime, random

import lightgbm as lgb

import xgboost as xgb

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import roc_auc_score



pd.options.display.max_rows = 4000



def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)



SEED = 42

seed_everything(SEED)

START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
%%time

train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')



train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')



sample_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')



train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



print("Train shape: ", train.shape)

print("Test shape: ", test.shape)



y = train['isFraud'].copy()

del train_transaction, train_identity, test_transaction, test_identity



# Drop target, fill in NaNs

train = train.drop('isFraud', axis=1)
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
def addNewFeatures(data): 

    data['uid'] = data['card1'].astype(str)+'_'+data['card2'].astype(str)



    data['uid2'] = data['uid'].astype(str)+'_'+data['card3'].astype(str)+'_'+data['card5'].astype(str)



    data['uid3'] = data['uid2'].astype(str)+'_'+data['addr1'].astype(str)+'_'+data['addr2'].astype(str)



    data['D9'] = np.where(data['D9'].isna(),0,1)

    

    return data



train = addNewFeatures(train)

test = addNewFeatures(test)
i_cols = ['card1','card2','card3','card5','uid','uid2','uid3']



for col in i_cols:

    for agg_type in ['mean','std']:

        new_col_name = col+'_TransactionAmt_'+agg_type

        temp_df = pd.concat([train[[col, 'TransactionAmt']], test[[col,'TransactionAmt']]])

        #temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)

        temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(

                                                columns={agg_type: new_col_name})



        temp_df.index = list(temp_df[col])

        temp_df = temp_df[new_col_name].to_dict()   



        train[new_col_name] = train[col].map(temp_df)

        test[new_col_name]  = test[col].map(temp_df)



train = train.replace(np.inf,999)

test = test.replace(np.inf,999)
train['TransactionAmt'] = np.log1p(train['TransactionAmt'])

test['TransactionAmt'] = np.log1p(test['TransactionAmt'])
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other',

          'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',

          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft',

          'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 

          'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other',

          'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo',

          'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',

          'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo',

          'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo',

          'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo',

          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other',

          'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple',

          'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other',

          'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

us_emails = ['gmail', 'net', 'edu']



for c in ['P_emaildomain', 'R_emaildomain']:

    train[c + '_bin'] = train[c].map(emails)

    test[c + '_bin'] = test[c].map(emails)

    

    train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])

    test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])

    

    train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
p = 'P_emaildomain'

r = 'R_emaildomain'

uknown = 'email_not_provided'



def setDomain(df):

    df[p] = df[p].fillna(uknown)

    df[r] = df[r].fillna(uknown)

    

    # Check if P_emaildomain matches R_emaildomain

    df['email_check'] = np.where((df[p]==df[r])&(df[p]!=uknown),1,0)



    df[p+'_prefix'] = df[p].apply(lambda x: x.split('.')[0])

    df[r+'_prefix'] = df[r].apply(lambda x: x.split('.')[0])

    

    return df

    

train=setDomain(train)

test=setDomain(test)
def setTime(df):

    df['TransactionDT'] = df['TransactionDT'].fillna(df['TransactionDT'].median())

    # Temporary

    df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

    df['DT_M'] = (df['DT'].dt.year-2017)*12 + df['DT'].dt.month

    df['DT_W'] = (df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear

    df['DT_D'] = (df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear

    

    df['DT_hour'] = df['DT'].dt.hour

    df['DT_day_week'] = df['DT'].dt.dayofweek

    df['DT_day'] = df['DT'].dt.day

    

    return df

    

train=setTime(train)

test=setTime(test)
train["lastest_browser"] = np.zeros(train.shape[0])

test["lastest_browser"] = np.zeros(test.shape[0])



def setBrowser(df):

    df.loc[df["id_31"]=="samsung browser 7.0",'lastest_browser']=1

    df.loc[df["id_31"]=="opera 53.0",'lastest_browser']=1

    df.loc[df["id_31"]=="mobile safari 10.0",'lastest_browser']=1

    df.loc[df["id_31"]=="google search application 49.0",'lastest_browser']=1

    df.loc[df["id_31"]=="firefox 60.0",'lastest_browser']=1

    df.loc[df["id_31"]=="edge 17.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 69.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 67.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 63.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 63.0 for ios",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 64.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 64.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 64.0 for ios",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 65.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 65.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 65.0 for ios",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 66.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 66.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 66.0 for ios",'lastest_browser']=1

    return df



train=setBrowser(train)

test=setBrowser(test)
def setDevice(df):

    df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()

    

    df['device_name'] = df['DeviceInfo'].str.split('/', expand=True)[0]



    df.loc[df['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'

    df.loc[df['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'

    df.loc[df['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'

    df.loc[df['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'

    df.loc[df['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'

    df.loc[df['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'

    df.loc[df['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'

    df.loc[df['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'

    df.loc[df['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'

    df.loc[df['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'

    df.loc[df['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'

    df.loc[df['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'

    df.loc[df['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'

    df.loc[df['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'

    df.loc[df['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'

    df.loc[df['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'

    df.loc[df['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'



    #df.loc[df.device_name.isin(df.device_name.value_counts()[df.device_name.value_counts() < 200].index), 'device_name'] = "Others"

    df['had_id'] = 1

    gc.collect()

    

    return df



train=setDevice(train)

test=setDevice(test)
i_cols = ['card1','card2','card3','card5',

          'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',

          'D1','D2','D3','D4','D5','D6','D7','D8',

          'addr1','addr2',

          'dist1','dist2',

          'P_emaildomain', 'R_emaildomain',

          'DeviceInfo','device_name',

          'id_30','id_33',

          'uid','uid2','uid3',

         ]



for col in i_cols:

    temp_df = pd.concat([train[[col]], test[[col]]])

    fq_encode = temp_df[col].value_counts(dropna=False).to_dict()   

    train[col+'_fq_enc'] = train[col].map(fq_encode)

    test[col+'_fq_enc']  = test[col].map(fq_encode)





for col in ['DT_M','DT_W','DT_D']:

    temp_df = pd.concat([train[[col]], test[[col]]])

    fq_encode = temp_df[col].value_counts().to_dict()

            

    train[col+'_total'] = train[col].map(fq_encode)

    test[col+'_total']  = test[col].map(fq_encode)

        



periods = ['DT_M','DT_W','DT_D']

i_cols = ['uid']

for period in periods:

    for col in i_cols:

        new_column = col + '_' + period

            

        temp_df = pd.concat([train[[col,period]], test[[col,period]]])

        temp_df[new_column] = temp_df[col].astype(str) + '_' + (temp_df[period]).astype(str)

        fq_encode = temp_df[new_column].value_counts().to_dict()

            

        train[new_column] = (train[col].astype(str) + '_' + train[period].astype(str)).map(fq_encode)

        test[new_column]  = (test[col].astype(str) + '_' + test[period].astype(str)).map(fq_encode)

        

        train[new_column] /= train[period+'_total']

        test[new_column]  /= test[period+'_total']
def get_too_many_null_attr(data):

    many_null_cols = [col for col in data.columns if data[col].isnull().sum() / data.shape[0] > 0.9]

    return many_null_cols



def get_too_many_repeated_val(data):

    big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

    return big_top_value_cols



def get_useless_columns(data):

    too_many_null = get_too_many_null_attr(data)

    print("More than 90% null: " + str(len(too_many_null)))

    too_many_repeated = get_too_many_repeated_val(data)

    print("More than 90% repeated value: " + str(len(too_many_repeated)))

    cols_to_drop = list(set(too_many_null + too_many_repeated))

    #cols_to_drop.remove('isFraud')

    return cols_to_drop
cols_to_drop = get_useless_columns(train)
train = train.drop(cols_to_drop, axis=1)

test = test.drop(cols_to_drop, axis=1)
print(train.shape)

print(test.shape)

print(y.shape)
numerical_cols = train.select_dtypes(exclude = 'object').columns

categorical_cols = train.select_dtypes(include = 'object').columns
categorical_cols[:5]
# Label Encoding

for f in train.columns:

    if train[f].dtype.name =='object' or test[f].dtype.name =='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train[f].values) + list(test[f].values))

        train[f] = lbl.transform(list(train[f].values))

        test[f] = lbl.transform(list(test[f].values))
train = train.fillna(-999)

test = test.fillna(-999)
print(train.isnull().sum().max())

print(test.isnull().sum().max())
train.head()
X = train.drop(['TransactionDT', 'DT'], axis=1)

X_test = test.drop(['TransactionDT', 'DT'], axis=1)
print("X:", X.shape)

print("y_train:", y.shape)

print("X_test:", X_test.shape)
params = {'num_leaves': 546,

          'min_child_weight': 0.03454472573214212,

          'feature_fraction': 0.1797454081646243,

          'bagging_fraction': 0.2181193142567742,

          'min_data_in_leaf': 106,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.005883242363721497,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": 'auc',

          "verbosity": -1,

          'reg_alpha': 0.3299927210061127,

          'reg_lambda': 0.3885237330340494,

          'random_state': 42,

}
%%time



NFOLDS = 5

folds = KFold(n_splits=NFOLDS)



columns = X.columns

splits = folds.split(X, y)

y_preds = np.zeros(X_test.shape[0])

y_oof = np.zeros(X.shape[0])

score = 0



feature_importances = pd.DataFrame()

feature_importances['feature'] = columns

  

for fold_n, (train_index, valid_index) in enumerate(splits):

    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    dtrain = lgb.Dataset(X_train, label=y_train)

    dvalid = lgb.Dataset(X_valid, label=y_valid)



    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)

    

    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

    

    y_pred_valid = clf.predict(X_valid)

    y_oof[valid_index] = y_pred_valid

    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")

    

    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS

    y_preds += clf.predict(X_test) / NFOLDS

    

    del X_train, X_valid, y_train, y_valid

    gc.collect()

    

print(f"\nMean AUC = {score}")

print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")
submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
submission['isFraud'] = y_preds
submission.head()
submission.to_csv('submission.csv', index=False)
feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)

feature_importances.to_csv('feature_importances.csv')



plt.figure(figsize=(16, 16))

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');

plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));