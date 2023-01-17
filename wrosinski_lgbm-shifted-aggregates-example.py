import gc
import glob
import os
import time

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kaggle.competitions import twosigmanews
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

%matplotlib inline
pd.set_option("display.max_columns", 96)
pd.set_option("display.max_rows", 96)

plt.rcParams['figure.figsize'] = (12, 9)
env = twosigmanews.make_env()
train_market, train_news = env.get_training_data()
days = env.get_prediction_days()
debug = True
n_debug_samples = 100000
train_market['time'] = pd.to_datetime(train_market['time'])
train_news['time'] = pd.to_datetime(train_news['time'])
train_news['sourceTimestamp'] = pd.to_datetime(train_news['sourceTimestamp'])
train_news['firstCreated'] = pd.to_datetime(train_news['firstCreated'])

train_market = train_market.sort_values('time')
train_news = train_news.sort_values('time')

if debug:
    print('Debugging.')
    train_market = train_market.iloc[:n_debug_samples, :]
    train_news = train_news.iloc[:n_debug_samples, :]
train_news = train_news.drop(['headline', 'subjects', 'audiences'], axis=1)
print('Market NaN structure:\n{}'.format(np.sum(pd.isnull(train_market))))
print('\nNews NaN structure:\n{}'.format(np.sum(pd.isnull(train_news))))
print(train_news.headlineTag.value_counts())

train_news.headlineTag = train_news.headlineTag.replace('', 'NoInfo')

print('\nAfter filling in:\n{}'.format(train_news.headlineTag.value_counts()))
def create_time_resolutions(df):
    
    df = df.copy()
    # Basic hour and day:
    df['dt_hour'] = df.time.dt.floor('h')
    df['dt_day'] = df.time.dt.floor('d')

    # Week of year, cannot be created in an above way due to not being a fixed frequency (according to pandas docs):
    df['dt_weekofyear'] = df.apply(
        lambda x: '{}_{}'.format(x['time'].weekofyear, x['time'].year), axis=1)
    
    return df


train_news = create_time_resolutions(train_news)
train_market = create_time_resolutions(train_market)
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
def rename_columns(df):
    
    df.columns = pd.Index(['{}{}'.format(
        c[0], c[1].upper()) for c in df.columns.tolist()])
    
    return df
# Mean aggregates for day for each of the assets:
news_asset_day = train_news.groupby(['dt_day', 'assetName'])[news_cols_agg_num].agg(aggs_num).reset_index()
# Similarly, for each week of year:
news_asset_week = train_news.groupby(['dt_weekofyear', 'assetName'])[news_cols_agg_num].agg(aggs_num).reset_index()

# Now, let's incorporate some history.
# We will shift week by one and create features describing last week state of the asset:
news_asset_last_day = train_news.groupby(
    ['assetName', 'dt_day'])[news_cols_agg_num].agg(aggs_num).groupby(['assetName']).shift(1).reset_index()
news_asset_last_week = train_news.groupby(
    ['assetName', 'dt_weekofyear'])[news_cols_agg_num].agg(aggs_num).groupby(['assetName']).shift(1).reset_index()


news_asset_day = rename_columns(news_asset_day)
news_asset_week = rename_columns(news_asset_week)
news_asset_last_day = rename_columns(news_asset_last_day)
news_asset_last_week = rename_columns(news_asset_last_week)
# Mean aggregates for day for each of the assets:
market_asset_day = train_market.groupby(['dt_day', 'assetName'])[market_cols_agg_num].agg(aggs_num).reset_index()
# Similarly, for each week of year:
market_asset_week = train_market.groupby(['dt_weekofyear', 'assetName'])[market_cols_agg_num].agg(aggs_num).reset_index()

# Now, let's incorporate some history.
# We will shift week by one and create features describing last week state of the asset:
market_asset_last_day = train_market.groupby(
    ['assetName', 'dt_day'])[market_cols_agg_num].agg(aggs_num).groupby(['assetName']).shift(1).reset_index()
market_asset_last_week = train_market.groupby(
    ['assetName', 'dt_weekofyear'])[market_cols_agg_num].agg(aggs_num).groupby(['assetName']).shift(1).reset_index()


market_asset_day = rename_columns(market_asset_day)
market_asset_week = rename_columns(market_asset_week)
market_asset_last_day = rename_columns(market_asset_last_day)
market_asset_last_week = rename_columns(market_asset_last_week)
X = train_market.copy()

# Merge news features:
# At this point we will perform a suboptimal merge on assetName and day:
X = X.merge(
    news_asset_day, how='left', on=['assetName', 'dt_day'], suffixes=('', '_news_day'))
X = X.merge(
    news_asset_week, how='left', on=['assetName', 'dt_weekofyear'], suffixes=('', '_news_week'))
X = X.merge(
    news_asset_last_day, how='left', on=['assetName', 'dt_day'], suffixes=('', '_news_last_day'))
X = X.merge(
    news_asset_last_week, how='left', on=['assetName', 'dt_weekofyear'], suffixes=('', '_news_last_week'))

# Merge market features:
X = X.merge(
    market_asset_day, how='left', on=['assetName', 'dt_day'], suffixes=('', '_market_day'))
X = X.merge(
    market_asset_week, how='left', on=['assetName', 'dt_weekofyear'], suffixes=('', '_market_week'))
X = X.merge(
    market_asset_last_day, how='left', on=['assetName', 'dt_day'], suffixes=('', '_market_last_day'))
X = X.merge(
    market_asset_last_week, how='left', on=['assetName', 'dt_weekofyear'], suffixes=('', '_market_last_week'))


# Encode basic categorical variables:
# X['assetCode'] = pd.factorize(X['assetCode'])[0]
# X['assetName'] = pd.factorize(X['assetName'])[0]
print('New train shape: {}'.format(X.shape))
print('\nNew NaN structure:\n{}'.format(np.sum(pd.isnull(X))))
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

# To reduce memory footprint:
X_train = X_train.astype(np.float32)
X_valid = X_valid.astype(np.float32)
train_cols = X_train.columns.tolist()

dtrain = lgb.Dataset(X_train.values, y_train, feature_name=train_cols)
dvalid = lgb.Dataset(X_valid.values, y_valid,
                     feature_name=train_cols, reference=dtrain)


params = {'learning_rate': 0.02,
          'boosting': 'gbdt', 
          'objective': 'binary', 
          'seed': 2018}

lgb_model = lgb.train(params, dtrain, 
                      num_boost_round=1000, 
                      valid_sets=(dvalid,), 
                      valid_names=('valid',), 
                      verbose_eval=50, 
                      early_stopping_rounds=100)
y_pred_conf_valid = lgb_model.predict(X_valid) * 2 - 1
y_pred_conf_valid_binary = (y_pred_conf_valid > 0).astype(np.uint8)
print('Valid confidence accuracy: {}'.format(accuracy_score(y_valid, y_pred_conf_valid_binary)))
confidence_valid = y_pred_conf_valid.copy()
r_valid = r_valid.clip(-1,1)
x_t_i = confidence_valid * r_valid * u_valid
data = {'day' : d_valid, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print('Score valid: {}'.format(score_valid))
n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
predicted_confidences = np.array([])
to_drop_test = [
          'time',
          'assetCode',
          'assetName',
          'dt_weekofyear',
          'dt_hour',
          'dt_day']


for (market_obs_df, news_obs_df, predictions_template_df) in days:

    n_days += 1
    print(n_days, end=' ')

    t = time.time()

    market_obs_df = market_obs_df[market_obs_df.assetCode.isin(predictions_template_df.assetCode)]
    X = market_obs_df.copy()
    X = create_time_resolutions(X)

    # Merge news features:
    X = X.merge(
        news_asset_day, how='left', on=['assetName', 'dt_day'], suffixes=('', '_news_day'))
    X = X.merge(
        news_asset_week, how='left', on=['assetName', 'dt_weekofyear'], suffixes=('', '_news_week'))
    X = X.merge(
        news_asset_last_day, how='left', on=['assetName', 'dt_day'], suffixes=('', '_news_last_day'))
    X = X.merge(
        news_asset_last_week, how='left', on=['assetName', 'dt_weekofyear'], suffixes=('', '_news_last_week'))

    # Merge market features:
    X = X.merge(
        market_asset_day, how='left', on=['assetName', 'dt_day'], suffixes=('', '_market_day'))
    X = X.merge(
        market_asset_week, how='left', on=['assetName', 'dt_weekofyear'], suffixes=('', '_market_week'))
    X = X.merge(
        market_asset_last_day, how='left', on=['assetName', 'dt_day'], suffixes=('', '_market_last_day'))
    X = X.merge(
        market_asset_last_week, how='left', on=['assetName', 'dt_weekofyear'], suffixes=('', '_market_last_week'))

    X = X.drop(to_drop_test, axis=1)
    X = X.astype(np.float32)

    prep_time += time.time() - t

    t = time.time()
    market_prediction = lgb_model.predict(X) * 2 - 1
    predicted_confidences = np.concatenate(
        (predicted_confidences, market_prediction))
    prediction_time += time.time() - t

    t = time.time()
    preds = pd.DataFrame(
        {'assetCode': market_obs_df['assetCode'], 'confidence': market_prediction})
    # insert predictions to template
    predictions_template_df = predictions_template_df.merge(preds, how='left').drop(
        'confidenceValue', axis=1).fillna(0).rename(columns={'confidence': 'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t


env.write_submission_file()
total = prep_time + prediction_time + packaging_time
print(f'Preparing Data: {prep_time:.2f}s')
print(f'Making Predictions: {prediction_time:.2f}s')
print(f'Packing: {packaging_time:.2f}s')
print(f'Total: {total:.2f}s')

plt.hist(predicted_confidences, bins='auto')
plt.title("Test set confidence:")
plt.show()