import gc
import glob
import os

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from contextlib import contextmanager
from multiprocessing.pool import ThreadPool, Pool
from joblib import Parallel, delayed
import time

%matplotlib inline
pd.set_option("display.max_columns", 96)
pd.set_option("display.max_rows", 96)

plt.rcParams['figure.figsize'] = (12, 9)
debug = True
n_debug_samples = 200000
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
print('Done!')

(market_train_df, news_train_df) = env.get_training_data()

df_ = market_train_df.copy()
if debug:
    df_ = df_.iloc[:n_debug_samples, :]
print(df_.shape)
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('{0} done in {1:.3f} seconds.'.format(name, time.time() - t0))
    

def split_df(df, num_splits):
    
    df_list = []
    rows_splits = np.linspace(0, df.shape[0], num_splits).astype(np.int)
    print('Split into {} parts'.format(num_splits))
    print('Row splits:\n{}'.format(rows_splits))
    
    for i in range(len(rows_splits) - 1):
        df_list.append(df.iloc[rows_splits[i]:rows_splits[i+1]])
        
    return df_list
dfs_split = split_df(df_, 4)
dfs_ = pd.concat(dfs_split, ignore_index=True, sort=False)
random_index = np.random.randint(0, df_.shape[0])

print(random_index)
print(df_.iloc[random_index, :])
print('\n', dfs_.iloc[random_index, :])
# Process time column to datetime format
def datetime_proc(df):
    df['time'] = pd.to_datetime(df['time'])
    return df


# Create features containing information about time dimension
def create_time_resolutions(df):
    
    df = df.copy()
    df['dt_hour'] = df.time.dt.floor('h')
    df['dt_day'] = df.time.dt.floor('d')
    df['dt_weekofyear'] = df.apply(
        lambda x: '{}_{}'.format(x['time'].weekofyear, x['time'].year), axis=1)
    
    return df


# Rename columns after grouping for easy merge and access
def rename_columns(df):
    
    df.columns = pd.Index(['{}{}'.format(
        c[0], c[1].upper()) for c in df.columns.tolist()])
    
    return df


# Create grouped features
def create_grouped_df(df, group_by, columns_set):
    
    df_grouped = df.groupby(group_by)[columns_set].agg(aggs_num).reset_index()
    df_grouped = rename_columns(df_grouped)
    
    return df_grouped


# Create grouped features with shift 
def create_grouped_df_shifted(df, group_by, columns_set, shift_name=['assetName'], shift=1):
    
    df_grouped = df.groupby(group_by)[columns_set].agg(aggs_num).groupby(
        shift_name).shift(shift).reset_index()
    df_grouped = rename_columns(df_grouped)
    
    return df_grouped
# News columns:
news_cols_agg_num = ['urgency', 'sentenceCount', 'wordCount',
                    'firstMentionSentence', 'relevance',
                   'sentimentClass', 'sentimentNegative',
                   'sentimentNeutral', 'sentimentPositive',
                   'sentimentWordCount']

news_cols_agg_cat = ['sourceId', 'provider', 'headlineTag',
                    'marketCommentary']


# Market columns:
market_cols_agg_num = ['volume', 'close', 'open',
                      'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',]

market_cols_agg_cat = ['assetCode']


aggs_num = ['mean', 'std']
aggs_cat = ['count']
%%time

dfs_ = df_.assign(time = pd.to_datetime(df_['time']))
with timer('datetime processing:'):
    df_ = create_time_resolutions(df_)
with timer('pool datetime processing:'):
    with Pool(processes=4) as pool:
        dfs_proc = pool.map(create_time_resolutions, dfs_split)
        

with timer('pool datetime processing threads:'):
    with ThreadPool(processes=4) as pool:
        dfs_proc = pool.map(create_time_resolutions, dfs_split)
with timer('joblib parallel datetime processing:'):
    dfs_proc = Parallel(n_jobs=4)(delayed(create_time_resolutions)(i) for i in dfs_split)
    
    
with timer('joblib parallel datetime processing threads:'):
    dfs_proc = Parallel(n_jobs=4, prefer='threads')(delayed(create_time_resolutions)(i) for i in dfs_split)
# Now, let's concat processed DFs into one:
df_time = pd.concat(dfs_proc, ignore_index=True, sort=False)
# 
groupings = [
    ['dt_day', 'assetName'],
    ['dt_weekofyear', 'assetName'],
    ['assetName', 'open'],
    ['assetName', 'close'],
]
dfs_proc = []

with timer('grouping features:'):
    for i in groupings:
        dfs_proc.append(create_grouped_df(df_time, group_by=i, columns_set=market_cols_agg_num))
        
        
dfs_proc_shift = []

with timer('grouping features shifted:'):
    for i in groupings:
        dfs_proc_shift.append(create_grouped_df_shifted(df_time, group_by=i, columns_set=market_cols_agg_num))
with timer('pool grouping features parallel:'):
    with Pool(processes=4) as pool:
        dfs_proc = pool.starmap(create_grouped_df,
                                [(df_time, groupings[0], market_cols_agg_num),
                                (df_time, groupings[1], market_cols_agg_num),
                                (df_time, groupings[2], market_cols_agg_num),
                                (df_time, groupings[3], market_cols_agg_num)])
        
with timer('pool grouping features parallel threads:'):
    with ThreadPool(processes=4) as pool:
        dfs_proc_shift = pool.starmap(create_grouped_df,
                                [(df_time, groupings[0], market_cols_agg_num),
                                (df_time, groupings[1], market_cols_agg_num),
                                (df_time, groupings[2], market_cols_agg_num),
                                (df_time, groupings[3], market_cols_agg_num)])
        

with timer('pool grouping features shifted parallel:'):
    with Pool(processes=4) as pool:
        dfs_proc = pool.starmap(create_grouped_df_shifted,
                                [(df_time, groupings[0], market_cols_agg_num),
                                (df_time, groupings[1], market_cols_agg_num),
                                (df_time, groupings[2], market_cols_agg_num),
                                (df_time, groupings[3], market_cols_agg_num)])
        
with timer('pool grouping features shifted parallel threads:'):
    with ThreadPool(processes=4) as pool:
        dfs_proc_shift = pool.starmap(create_grouped_df_shifted,
                                [(df_time, groupings[0], market_cols_agg_num),
                                (df_time, groupings[1], market_cols_agg_num),
                                (df_time, groupings[2], market_cols_agg_num),
                                (df_time, groupings[3], market_cols_agg_num)])
# Without prefer='threads' joblib throws an error.

with timer('joblib grouping features parallel threads:'):
    dfs_proc = Parallel(n_jobs=4, prefer='threads')(delayed(create_grouped_df)(
        df_time, group_by=i, columns_set=market_cols_agg_num) for i in groupings)
    
    
with timer('joblib grouping features shifted parallel threads:'):
    dfs_proc_shift = Parallel(n_jobs=4, prefer='threads')(delayed(create_grouped_df_shifted)(
        df_time, group_by=i, columns_set=market_cols_agg_num) for i in groupings)
X = df_time.copy()

for i in range(len(groupings)):
    X = X.merge(dfs_proc[i], how='left',
                on=groupings[i], suffixes=('', '_basic_{}'.format('_'.join(groupings[i]))))
    X = X.merge(dfs_proc_shift[i], how='left',
                on=groupings[i], suffixes=('', '_shift1_{}'.format('_'.join(groupings[i]))))
    
X
# Floor all datetime objects to a specified resolution:
# 'd' - days
# 'h' - hours
X['time_split_resolution'] = X.time.dt.floor('d')

# Select unique values based on market set:
train_times_unique = X.time_split_resolution.unique()
# Split unique dates into 80/20% training/validation split:
tr_times, valid_times = train_test_split(train_times_unique, test_size=0.2, random_state=1337)

# Create subsets for market and news datasets:
X_tr = X[X.time_split_resolution.isin(tr_times)]
X_val = X[X.time_split_resolution.isin(valid_times)]

print('Dataset shapes: train - {}, valid - {}'.format(X_tr.shape, X_val.shape))
def get_input(df_, to_drop):
    X = df_.drop(to_drop, axis=1)
    y = (df_.loc[:, 'returnsOpenNextMktres10'] >= 0).values.astype(np.uint8)
    # y = df_.loc[:, 'returnsOpenNextMktres10'].values
    r = df_.loc[:, 'returnsOpenNextMktres10'].values
    u = df_.loc[:, 'universe']
    d = df_.loc[:, 'time'].dt.date
    return X, y, r, u, d


to_drop = ['returnsOpenNextMktres10',
          'universe',
          'time',
          'assetCode',
          'assetName',
          'dt_weekofyear',
          'dt_hour',
          'dt_day',
          'time_split_resolution']
X_train, y_train, r_train, u_train, d_train = get_input(X_tr, to_drop)
X_valid, y_valid, r_valid, u_valid, d_valid = get_input(X_val, to_drop)
train_cols = X_train.columns.tolist()

dtrain = lgb.Dataset(X_train.values, y_train, feature_name=train_cols)
dvalid = lgb.Dataset(X_valid.values, y_valid,
                     feature_name=train_cols, reference=dtrain)


params = {'learning_rate': 0.05,
          'boosting': 'gbdt', 
          'objective': 'binary', 
          'seed': 2018,
          'nthreads': 1}

with timer('basic model training:'):
    lgb_model = lgb.train(params, dtrain, 
                          num_boost_round=1000, 
                          valid_sets=(dvalid,), 
                          valid_names=('valid',), 
                          verbose_eval=25, 
                          early_stopping_rounds=20)
params1 = {'learning_rate': 0.05,
          'boosting': 'gbdt', 
          'objective': 'binary', 
          'num_leaves': 8,
          'min_data_in_leaf': 4,
          'max_bin': 255,
          'bagging_fraction': 0.7,
          'lambda_l2': 0.01,
          'max_depth': 12,
          'seed': 2018,
          'nthreads': 1}

params2 = {'learning_rate': 0.005,
          'boosting': 'gbdt', 
          'objective': 'binary', 
          'num_leaves': 16,
          'min_data_in_leaf': 2,
          'max_bin': 255,
          'bagging_fraction': 0.4,
          'lambda_l2': 0.001,
          'max_depth': 16,
          'seed': 2018,
          'nthreads': 1}

params3 = {'learning_rate': 0.1,
          'boosting': 'gbdt', 
          'objective': 'binary', 
          'num_leaves': 6,
          'min_data_in_leaf': 4,
          'max_bin': 128,
          'bagging_fraction': 0.9,
          'lambda_l2': 0.03,
          'max_depth': 7,
          'seed': 2018,
          'nthreads': 1}

params4 = {'learning_rate': 0.001,
          'boosting': 'gbdt', 
          'objective': 'binary', 
          'num_leaves': 64,
          'min_data_in_leaf': 16,
          'max_bin': 255,
          'bagging_fraction': 0.75,
          'lambda_l2': 1e-3,
          'max_depth': 14,
          'seed': 2018,
          'nthreads': 1}


params_list = [
    params1,
    params2,
    params3,
    params4,
]


def train_lgb(params, dtrain, dvalid):
    
    lgb_model = lgb.train(params, dtrain, 
                      num_boost_round=1000, 
                      valid_sets=(dvalid,), 
                      valid_names=('valid',), 
                      verbose_eval=25, 
                      early_stopping_rounds=20)
    
    return lgb_model
with timer('joblib lgbm models training:'):
    lgb_models = Parallel(n_jobs=4, prefer='threads')(delayed(train_lgb)(
        i, dtrain, dvalid) for i in params_list)
for l in lgb_models:
    y_pred_conf_valid = l.predict(X_valid) * 2 - 1
    y_pred_conf_valid_binary = (y_pred_conf_valid > 0).astype(np.uint8)
    print('Valid accuracy: {:.4f}'.format(accuracy_score(y_valid, y_pred_conf_valid_binary)))

    confidence_valid = l.predict(X_valid) * 2 - 1
    r_valid = r_valid.clip(-1,1)
    x_t_i = confidence_valid * r_valid * u_valid
    data = {'day' : d_valid, 'x_t_i' : x_t_i}
    df = pd.DataFrame(data)
    x_t = df.groupby('day').sum().values.flatten()
    mean = np.mean(x_t)
    std = np.std(x_t)
    score_valid = mean / std
    print('Valid score: {:.4f}'.format(score_valid))
