import os
import sys
import time
import gc
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt

%matplotlib inline


"""This code is based on this script by baris:
https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977?scriptVersionId=3224614
"""

DEBUG = 0

logger = logging.getLogger()
logger.handlers = [logging.StreamHandler(sys.stdout)]
logger.setLevel(20 - DEBUG * 10)

predictors=[]
def do_next_Click(df, agg_suffix='nextClick', agg_type='float32'):
    """Extracting next click feature.
    Taken help from https://www.kaggle.com/nanomathias/feature-engineering-importance-testing  
    """
    logger.info("Extracting {} time calculation features...".format(agg_suffix))
    
    GROUP_BY_NEXT_CLICKS = [
        {'groupby': ['ip', 'os', 'device', 'app']},
        {'groupby': ['ip', 'os', 'device', 'app', 'channel']},
        {'groupby': ['app', 'device', 'channel']},
        {'groupby': ['ip', 'os', 'device']},
        {'groupby': ['device', 'hour']},
        
       # {'groupby': ['device']},
        
        {'groupby': ['ip', 'app']},
        {'groupby': ['ip', 'channel']},
        {'groupby': ['device', 'channel']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
    
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        logger.info(">> Grouping by {}".format(spec['groupby']))
        df[new_feature] = (df[all_features]
                           .groupby(spec['groupby'])
                           .click_time.shift(-1) - df.click_time).dt.seconds.astype(agg_type)
        predictors.append(new_feature)
        gc.collect()
    return df


def do_prev_Click(df, agg_suffix='prevClick', agg_type='float32'):
    """Extracting previous click feature.
    Taken help from https://www.kaggle.com/nanomathias/feature-engineering-importance-testing  
    """
    logger.info(">> Extracting {} time calculation features...".format(agg_suffix))
    
    GROUP_BY_NEXT_CLICKS = [
        {'groupby': ['ip', 'channel']},
        {'groupby': ['ip', 'os']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
    
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        logger.info(">> Grouping by {}".format(spec['groupby']))
        df[new_feature] = (df.click_time - 
                           df[all_features]
                           .groupby(spec['groupby'])
                           .click_time.shift(+1)).dt.seconds.astype(agg_type)
        
        predictors.append(new_feature)
        gc.collect()
    return df


def do_count(df, group_cols, agg_type='uint32', show_max=False, show_agg=True):
    """Add a new column with the count of another one after 
    grouping on a set of columns.
    """
    agg_name='{}_count'.format('_'.join(group_cols))
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    logger.info("{} max value = {}".format(agg_name, df[agg_name].max()))
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    gc.collect()
    return df


def do_countuniq(df, group_cols, counted, agg_type='uint32', show_max=False, show_agg=True):
    """Add a new column with the unique count of another one after 
    grouping on a set of columns.
    """
    agg_name= '{}_by_{}_countuniq'.format(('_'.join(group_cols)),(counted))  
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    logger.info("{} max value = {}".format(agg_name, df[agg_name].max()))
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    gc.collect()
    return df


def do_cumcount(df, group_cols, counted,agg_type='uint32', show_max=False, show_agg=True):
    """Add a new column with the cumulative count of another one after 
    grouping on a set of columns.
    """
    agg_name = '{}_by_{}_cumcount'.format(('_'.join(group_cols)),(counted)) 
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name] = gp.values
    del gp
    logger.info("{} max value = {}.".format(agg_name, df[agg_name].max()))
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    gc.collect()
    return df


def do_mean(df, group_cols, counted, agg_type='float32', show_max=False, show_agg=True):
    """Add a new column with the mean value of a another one after 
    grouping on a set of columns.
    """
    agg_name= '{}_by_{}_mean'.format(('_'.join(group_cols)),(counted))  
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    logger.info("{} max value = {}".format(agg_name, df[agg_name].max()))
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    gc.collect()
    return df


def do_var(df, group_cols, counted, agg_type='float32', show_max=False, show_agg=True):
    """Add a new column with the variance value of another one after
    grouping on a set of columns.
    """
    agg_name= '{}_by_{}_var'.format(('_'.join(group_cols)),(counted)) 
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    logger.info("{} max value = {}".format(agg_name, df[agg_name].max()))
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    gc.collect()
    return df


def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target',
                      objective='binary', metrics='auc', feval=None,
                      early_stopping_rounds=50, num_boost_round=3000,
                      verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.05,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 0,
    }

    lgb_params.update(params)

    xgtrain = lgb.Dataset(dtrain[predictors].values,
                          label=dtrain[target].values,
                          feature_name=predictors)
    
    xgvalid = lgb.Dataset(dvalid[predictors].values,
                          label=dvalid[target].values,
                          feature_name=predictors)
    
    del dtrain
    del dvalid
    gc.collect()
    
    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgvalid], 
                     valid_names=['valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)

    logger.info("Model Report")
    logger.info("bst1.best_iteration: {}".format(bst1.best_iteration))
    logger.info("{}:{}".format(metrics, evals_results['valid'][metrics][bst1.best_iteration-1]))
    return bst1, bst1.best_iteration


def sample_positive(df: pd.DataFrame, positive_ratio: float = 0.1) -> pd.DataFrame:
    """Over sample positive events.
    :param positive_ratio: The ratio of positive events to maintain.
    :return: Over sampled `DataFrame`.
    """
    positive = df[df.is_attributed == 1]  # Select positive events
    negative = df[df.is_attributed == 0]  # And negative events
    
    negative_sampled = negative.sample(len(positive) * 99)  # Sample negative events with negative 1 : 9 positive ratio
    logger.info('Sampled data: {:,} positive, {:,} => {:,} negative.'
                .format(positive.shape[0], negative.shape[0], negative_sampled.shape[0]))
    return (positive
            .append(negative_sampled)
            .sort_values(by='click_time')
            .reset_index(drop=True))  # Combine negative and positive 
def main():
    dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint8',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
    }

    logger.debug('*** Running in DEBUG mode. ***')
    nrows = 100000 if logger.getEffectiveLevel() == logging.DEBUG else None
    
    logger.info("Loading training data...")
    train = pd.read_csv('../input/train.csv',
                        parse_dates=['click_time'],
                        nrows=nrows,
                        dtype=dtypes,
                        usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
    
    train_df = sample_positive(train) # Oversampling positive events
    del(train)
        
    logger.info('Loading test data...')
    test_df = pd.read_csv("../input/test.csv",
                          nrows=nrows,
                          parse_dates=['click_time'],
                          dtype=dtypes,
                          usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

    train_size = len(train_df)
    val_size = int(train_size * 0.4)

    all_df = train_df.append(test_df).reset_index(drop=True)
    del test_df

    gc.collect()
    all_df['hour'] = pd.to_datetime(all_df.click_time).dt.hour.astype('int8')
    all_df['day'] = pd.to_datetime(all_df.click_time).dt.day.astype('int8') 
    
    all_df = do_countuniq(all_df, ['ip'], 'app'); gc.collect()
    all_df = do_var(all_df, ['channel', 'day'], 'hour'); gc.collect()
    all_df = do_mean(all_df, ['ip', 'app', 'channel'], 'hour'); gc.collect()
    all_df = do_countuniq(all_df, ['ip'], 'device'); gc.collect()
    all_df = do_next_Click(all_df, agg_suffix='nextClick', agg_type='float32'); gc.collect()
    all_df = do_count(all_df, ['ip', 'day', 'hour']); gc.collect()
    
    all_df = do_countuniq(all_df, ['ip'], 'channel'); gc.collect()
    all_df = do_count(all_df, ['ip'], 'app'); gc.collect()
    all_df = do_countuniq(all_df, ['ip', 'device', 'os'], 'app'); gc.collect()
    all_df = do_cumcount(all_df, ['ip', 'device', 'os'], 'app'); gc.collect()
    
    # all_df = do_countuniq(all_df, ['device'], 'day'); gc.collect()
    # all_df = do_var(all_df, ['device', 'day'], 'hour'); gc.collect()
    # all_df = do_count(all_df, ['app']); gc.collect()
    # all_df = do_count(all_df, ['channel']); gc.collect()
    
    
    # all_df = do_var(all_df, ['device'], 'day'); gc.collect()
    # all_df = do_countuniq(all_df, ['ip', 'channel'], 'app'); gc.collect()
    # all_df = do_countuniq(all_df, ['channel', 'day'], 'hour'); gc.collect()
    # all_df = do_countuniq(all_df, ['device', 'day'], 'hour'); gc.collect()
    
    del all_df['day']
    gc.collect()
    
    logger.info('Before appending predictors...{}'.format(sorted(predictors)))
    target = 'is_attributed'
    word = ['app','device','os', 'channel', 'hour']
    for feature in word:
        if feature not in predictors:
            predictors.append(feature)
    categorical = ['app', 'device', 'os', 'channel', 'hour']
    logger.info('After appending predictors...{}'.format(sorted(predictors)))

    train_df = all_df.iloc[:(train_size - val_size)]
    val_df = all_df.iloc[(train_size - val_size) : train_size]
    test_df = all_df.iloc[train_size:]

    logger.info("Training size: {}".format(len(train_df)))
    logger.info("Validation size: {}".format(len(val_df)))
    logger.info("Test size : {}".format(len(test_df)))

    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    gc.collect()
    start_time = time.time()

    params = {
        'learning_rate': 0.10,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':200 # because training data is extremely unbalanced 
    }
    
    bst, best_iteration = lgb_modelfit_nocv(params,
                                            train_df,
                                            val_df,
                                            predictors,
                                            target,
                                            objective='binary',
                                            metrics='auc',
                                            early_stopping_rounds=30,
                                            verbose_eval=True,
                                            num_boost_round=1000,
                                            categorical_features=categorical)

    logger.info('[{}]: model training time'.format(time.time() - start_time))
    del train_df
    del val_df
    gc.collect()

    ax = lgb.plot_importance(bst, max_num_features=300)
    plt.show()

    logger.info("Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors], num_iteration=best_iteration)
    sub.to_csv('sub_{}.csv'.format(str(int(time.time()))), index=False, float_format='%.9f')
    logger.info("Done...")
    return sub


if __name__ == '__main__':
    main()