import gc, os, logging, datetime, warnings, pickle, optuna

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from tqdm import tqdm_notebook

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn import preprocessing

warnings.filterwarnings('ignore')
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
os.listdir('../input/ieee-fraud-detection')
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')

test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')
print("train_transaction shape : ", train_transaction.shape)

print("train_identity shape : ", train_identity.shape)

print("test_transaction shape : ", test_transaction.shape)

print("test_identity shape : ", test_identity.shape)
df_train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

df_test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
for df in [train_transaction, train_identity, test_transaction, test_identity]:

    del df
drop_col = ['V300','V309','V111','V124','V106','V125','V315','V134','V102','V123','V316','V113',

            'V136','V305','V110','V299','V289','V286','V318','V103','V304','V116','V29','V284','V293',

            'V137','V295','V301','V104','V311','V115','V109','V119','V321','V114','V133','V122','V319',

            'V105','V112','V118','V117','V121','V108','V135','V320','V303','V297','V120',

            'V1','V14','V41','V65','V88','V107']

for df in [df_train, df_test]:

    df = df.drop(drop_col, axis=1)
print("df_train shape :", df_train.shape)

print("df_test shape :", df_test.shape)
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

dates_range = pd.date_range(start='2017-10-01', end='2019-01-01')

us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())
def make_time_feature(df):

    START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

    df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

    df['DT_M'] = ((df['DT'].dt.year-2017)*12 + df['DT'].dt.month).astype(np.int8)

    df['DT_W'] = ((df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear).astype(np.int8)

    df['DT_D'] = ((df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear).astype(np.int16)

    df['DT_hour'] = (df['DT'].dt.hour).astype(np.int8)

    df['DT_day_week'] = (df['DT'].dt.dayofweek).astype(np.int8)

    df['DT_day_month'] = (df['DT'].dt.day).astype(np.int8)

    df['is_holiday'] = (df['DT'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)

    return df
for df in [df_train, df_test]:

    df = make_time_feature(df=df)
df_deviceinfo = pd.read_csv('../input/ieee-deviceinfo/DeviceInfo.csv')
df_train = pd.merge(df_train, df_deviceinfo, on='DeviceInfo', how='left')

df_test = pd.merge(df_test, df_deviceinfo, on='DeviceInfo', how='left')
del df_deviceinfo
for df in [df_train, df_test]:

    df['OS'] = df['id_30'].apply(lambda x: str(x).split(' ')[0])

    df['OS_version'] = df['id_30'].apply(lambda x: str(x).split(' ')[-1])

    df['OS_version'] = df['OS_version'].apply(lambda x: str(x).replace('_', '').replace('.', ''))

    df['OS_version'] = df['OS_version'].apply(lambda x: int(x) if x.isdigit() else np.nan)
browser_release_date = {'android browser 4.0' : '2011-10-18',

                        'android webview 4.0' : '2011-10-18',

                        'chrome 43.0 for android' : '2015-04-15',

                        'chrome 46.0 for android' : '2015-10-14',

                        'chrome 49.0' : '2016-03-02',

                        'chrome 49.0 for android' : '2016-03-09',

                        'chrome 50.0 for android' : '2016-04-26',

                        'chrome 51.0' : '2016-05-25',

                        'chrome 51.0 for android' : '2016-06-01',

                        'chrome 52.0 for android' : '2016-07-27',

                        'chrome 53.0 for android' : '2016-09-07',

                        'chrome 54.0 for android' : '2016-10-19',

                        'chrome 55.0' : '2016-12-01',

                        'chrome 55.0 for android' : '2016-12-06',

                        'chrome 56.0' : '2017-01-25',

                        'chrome 56.0 for android' : '2017-02-01',

                        'chrome 57.0' : '2017-03-09',

                        'chrome 57.0 for android' : '2017-03-16',

                        'chrome 58.0' : '2017-04-19',

                        'chrome 58.0 for android' : '2017-04-20',

                        'chrome 59.0' : '2017-06-05',

                        'chrome 59.0 for android' : '2017-06-06',

                        'chrome 60.0' : '2017-07-25',

                        'chrome 60.0 for android' : '2017-07-31',

                        'chrome 61.0' : '2017-09-05',

                        'chrome 61.0 for android' : '2017-09-05',

                        'chrome 62.0' : '2017-10-17',

                        'chrome 62.0 for android' : '2017-10-19',

                        'chrome 62.0 for ios' : '2017-10-18',

                        'chrome 63.0' : '2017-12-06',

                        'chrome 63.0 for android' : '2017-12-05',

                        'chrome 63.0 for ios' : '2017-12-05',

                        'chrome 64.0' : '2018-01-24',

                        'chrome 64.0 for android' : '2018-01-23',

                        'chrome 64.0 for ios' : '2018-01-24',

                        'chrome 65.0' : '2018-03-06',

                        'chrome 65.0 for android' : '2018-03-06',

                        'chrome 65.0 for ios' : '2018-03-06',

                        'chrome 66.0' : '2018-04-17',

                        'chrome 66.0 for android' : '2018-04-17',

                        'chrome 66.0 for ios' : '2018-04-17',

                        'chrome 67.0' : '2018-05-29',

                        'chrome 67.0 for android' : '2018-05-31',

                        'chrome 69.0' : '2018-09-04',

                        'edge 13.0' : '2015-09-15',

                        'edge 14.0' : '2016-02-18',

                        'edge 15.0' : '2016-10-07',

                        'edge 16.0' : '2017-09-26',

                        'edge 17.0' : '2018-04-30',

                        'firefox 47.0' : '2016-06-07',

                        'firefox 48.0' : '2016-08-02',

                        'firefox 52.0' : '2017-03-07',

                        'firefox 55.0' : '2017-08-08',

                        'firefox 56.0' : '2017-09-28',

                        'firefox 57.0' : '2017-11-14',

                        'firefox 58.0' : '2018-01-23',

                        'firefox 59.0' : '2018-03-13',

                        'firefox 60.0' : '2018-05-09',

                        'firefox mobile 61.0' : '2018-06-26',

                        'google search application 48.0' : '2016-01-20',

                        'google search application 49.0' : '2016-03-02',

                        'ie 11.0 for desktop' : '2013-10-17',

                        'ie 11.0 for tablet' : '2013-10-17',

                        'mobile safari 10.0' : '2016-09-14',

                        'mobile safari 11.0' : '2017-09-20',

                        'mobile safari 8.0' : '2014-09-17',

                        'mobile safari 9.0' : '2015-09-16',

                        'opera 49.0' : '2017-11-08',

                        'opera 51.0' : '2018-02-07',

                        'opera 52.0' : '2018-03-14',

                        'opera 53.0' : '2018-05-10',

                        'safari 10.0' : '2016-09-20',

                        'safari 11.0' : '2017-09-19',

                        'safari 9.0' : '2015-09-30',

                        'samsung browser 3.3' : '2015-08-01',

                        'samsung browser 4.0' : '2016-03-11',

                        'samsung browser 4.2' : '2016-08-19',

                        'samsung browser 5.2' : '2016-11-01',

                        'samsung browser 5.4' : '2017-05-01',

                        'samsung browser 6.2' : '2017-08-01',

                        'samsung browser 6.4' : '2018-02-19',

                        'samsung browser 7.0' : '2018-03-01',

                       }
def time_split(val):

    try:

        return datetime.datetime.strptime(str(x), '%Y-%m-%d')

    except:

        return pd.NaT
for df in [df_train, df_test]:

    df['browser_elapsed_time'] = df['id_31'].map(browser_release_date)

    df['browser_elapsed_time'].apply(lambda x: time_split(x))

    df['browser_elapsed_time'] = df['DT'] - df['browser_elapsed_time'].astype('datetime64[D]')

    df['browser_elapsed_time'] = df['browser_elapsed_time'].dt.days

    del df['DT']
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 

          'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft',

          'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',

          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 

          'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',

          'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other',

          'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 

          'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 

          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',

          'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',

          'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft',

          'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 

          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 

          'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 

          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 

          'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 

          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 

          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other',

          'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

us_emails = ['gmail', 'net', 'edu']
def make_email_feature(df):

    for c in ['P_emaildomain', 'R_emaildomain']:

        df[c + '_bin'] = df[c].map(emails)

        df[c + '_suffix'] = df[c].map(lambda x: str(x).split('.')[-1])

        df[c + '_suffix'] = df[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
for df in [df_train, df_test]:

    df = make_email_feature(df=df)
def siev_card_feature(df_train, df_test, col):

    valid_card = pd.concat([df_train[[col]], df_test[[col]]])

    valid_card = valid_card[col].value_counts()

    valid_card_std = valid_card.values.std()

    invalid_cards = valid_card[valid_card < 2]

    print('{0}{1}\nNumber of Rare card : '.format(col, '*'*10), len(invalid_cards))

    

    valid_card = valid_card[valid_card >= 2]

    valid_card = list(valid_card.index)

    print('Number of not intersection in Train : ', len(df_train[~df_train[col].isin(df_test[col])]))

    print('Nmber of intersection in Train : ', len(df_train[df_train[col].isin(df_test[col])]))

    

    df_train[col] = np.where(df_train[col].isin(df_test[col]), df_train[col], np.nan)

    df_test[col]  = np.where(df_test[col].isin(df_train[col]), df_test[col], np.nan)



    df_train[col] = np.where(df_train[col].isin(valid_card), df_train[col], np.nan)

    df_test[col]  = np.where(df_test[col].isin(valid_card), df_test[col], np.nan)
for col in ['card1','card2','card3','card4','card5','card6']:

    siev_card_feature(df_train, df_test, col)
def make_composite_id(new_id, id1, id2):

    df_train[new_id] = df_train[id1].astype(str)+'_' + df_train[id2].astype(str)

    df_test[new_id] = df_test[id1].astype(str)+'_' + df_test[id2].astype(str)
make_composite_id('uid1', 'card1', 'card2') #card1 + card2 -> uid1

make_composite_id('uid2', 'uid1', 'card3')

make_composite_id('uid2', 'uid2', 'card5') #card1 + card3 +card5 -> uid2

make_composite_id('uid3', 'uid2', 'addr1')

make_composite_id('uid3', 'uid3', 'addr2') #card2 + addr1 + addr2 -> uid3

make_composite_id('uid4', 'uid3', 'P_emaildomain') #uid3 + P_emaildomain -> uid4

make_composite_id('uid5', 'uid3', 'R_emaildomain') #uid3 + R_emaildomain -> uid5

make_composite_id('bank_type', 'card3', 'card5') #card3 + card5 -> bank_type
gc.collect()
for df in [df_train, df_test]:

    df = reduce_mem_usage(df=df)
def create_new_columns(key, aggs):

    return [key + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
aggs = {}

for col in ['DT_D','DT_W','DT_M']:

    aggs[col] = ['nunique', 'mean', 'median', 'var', 'skew']

aggs
df_temp = pd.concat([df_train, df_test])

for key in tqdm_notebook(['card3', 'card5', 'bank_type']):

    new_columns = create_new_columns(key, aggs)

    df_grouped = df_temp.groupby(key).agg(aggs)

    df_grouped.columns = new_columns

    df_grouped.reset_index(drop=False,inplace=True)

    df_train = df_train.merge(df_grouped,on=key,how='left')

    df_test = df_test.merge(df_grouped,on=key,how='left')
aggs = {}

for col in ['TransactionAmt', 'D15']:

    aggs[col] = ['nunique', 'mean', 'median', 'var', 'skew', 'min', 'max']

aggs
for key in tqdm_notebook(['card1','card2','card3','card5','uid1','uid2','uid3','uid4','uid5','bank_type']):

    new_columns = create_new_columns(key, aggs)

    df_grouped = df_temp.groupby(key).agg(aggs)

    df_grouped.columns = new_columns

    df_grouped.reset_index(drop=False,inplace=True)

    df_train = df_train.merge(df_grouped,on=key,how='left')

    df_test = df_test.merge(df_grouped,on=key,how='left')
aggs = {}

for col in ['TransactionAmt', 'D15']:

    aggs[col] = ['mean', 'median', 'min', 'max']

    

for key in tqdm_notebook(['card1','card2','card3','card5','uid1','uid2','uid3','uid4','uid5','bank_type']):

    columns = create_new_columns(key, aggs)

    for i in columns:

        diff_column = 'diff_' + i

        ratio_column = 'ratio_' + i

        df_train[diff_column] = df_train['TransactionAmt'] - df_train[i]

        df_test[diff_column] = df_test['TransactionAmt'] - df_test[i]

        df_train[ratio_column] = df_train['TransactionAmt'] / df_train[i]

        df_test[ratio_column] = df_test['TransactionAmt'] / df_test[i]
del df_temp

del df_grouped

gc.collect()
df_train['nulls1'] = df_train.isna().sum(axis=1)

df_test['nulls1'] = df_test.isna().sum(axis=1)
categorical_features = list()

for i in df_train.columns:

    if(df_train[i].dtype == object):

        categorical_features.append(i)

print(categorical_features)
def label_encoding(df_train, df_test, feature):

    le = preprocessing.LabelEncoder()

    le.fit(list(df_train[feature].astype(str).values) + list(df_test[feature].astype(str).values))

    df_train[feature] = le.transform(list(df_train[feature].astype(str).values))

    df_test[feature] = le.transform(list(df_test[feature].astype(str).values))
for feature in tqdm_notebook(categorical_features):

    label_encoding(df_train, df_test, feature)
def frequency_encoding(df_train, df_test, feature, self_encoding=False):

    df_temp = pd.concat([df_train[[feature]], df_test[[feature]]])

    fq_encode = df_temp[feature].value_counts(dropna=False).to_dict()

    if self_encoding:

        df_train[feature] = df_train[feature].map(fq_encode)

        df_test[feature] = df_test[feature].map(fq_encode)            

    else:

        df_train[feature+'_fq_enc'] = df_train[feature].map(fq_encode)

        df_test[feature+'_fq_enc'] = df_test[feature].map(fq_encode)
for feature in tqdm_notebook(categorical_features):

    frequency_encoding(df_train, df_test, feature, self_encoding=False)
for feature in tqdm_notebook(['DT_M','DT_W','DT_D']):

    frequency_encoding(df_train, df_test, feature, self_encoding=False)
for feature in tqdm_notebook(['card1','card2','card3','card5','uid1','uid2','uid3','uid4','uid5']):

    frequency_encoding(df_train, df_test, feature, self_encoding=False)
for feature in tqdm_notebook(['general_vendor','general_platform','general_type']):

    frequency_encoding(df_train, df_test, feature, self_encoding=False)
for df in [df_train, df_test]:

    df = reduce_mem_usage(df=df)
gc.collect()
df_train.to_pickle('df_train.pkl')

df_test.to_pickle('df_test.pkl')