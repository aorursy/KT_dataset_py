import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import gc
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

import os
print(os.listdir("../input"))
train = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_train.gz', 
                    dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
test = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_test.gz', 
                   dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
train['totals.transactionRevenue'] = train['totals.transactionRevenue'].fillna(0)
y_reg = train['totals.transactionRevenue']

train.shape, test.shape
def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids
y_reg = train['totals.transactionRevenue'].fillna(0)
train.columns
train['target'] = y_reg
for df in [train, test]:
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['sess_date_dow'] = df['date'].dt.dayofweek
    df['sess_date_hours'] = df['date'].dt.hour
    df['sess_date_dom'] = df['date'].dt.day
    df.sort_values(['fullVisitorId', 'date'], ascending=True, inplace=True)
    df['next_session_1'] = (
        df['date'] - df[['fullVisitorId', 'date']].groupby('fullVisitorId')['date'].shift(1)
    ).astype(np.int64) // 1e9 // 60 // 60
    df['next_session_2'] = (
        df['date'] - df[['fullVisitorId', 'date']].groupby('fullVisitorId')['date'].shift(-1)
    ).astype(np.int64) // 1e9 // 60 // 60

y_reg = train['target']
del train['target']
def browser_mapping(x):
    browsers = ['chrome','safari','firefox','internet explorer','edge','opera','coc coc','maxthon','iron']
    if x in browsers:
        return x.lower()
    elif  ('android' in x) or ('samsung' in x) or ('mini' in x) or ('iphone' in x) or ('in-app' in x) or ('playstation' in x):
        return 'mobile browser'
    elif  ('mozilla' in x) or ('chrome' in x) or ('blackberry' in x) or ('nokia' in x) or ('browser' in x) or ('amazon' in x):
        return 'mobile browser'
    elif  ('lunascape' in x) or ('netscape' in x) or ('blackberry' in x) or ('konqueror' in x) or ('puffin' in x) or ('amazon' in x):
        return 'mobile browser'
    elif '(not set)' in x:
        return x
    else:
        return 'others'
def adcontents_mapping(x):
    if  ('google' in x):
        return 'google'
    elif  ('placement' in x) | ('placememnt' in x):
        return 'placement'
    elif '(not set)' in x or 'nan' in x:
        return x
    elif 'ad' in x:
        return 'ad'
    else:
        return 'others'
    
def source_mapping(x):
    if  ('google' in x):
        return 'google'
    elif  ('youtube' in x):
        return 'youtube'
    elif '(not set)' in x or 'nan' in x:
        return x
    elif 'yahoo' in x:
        return 'yahoo'
    elif 'facebook' in x:
        return 'facebook'
    elif 'reddit' in x:
        return 'reddit'
    elif 'bing' in x:
        return 'bing'
    elif 'quora' in x:
        return 'quora'
    elif 'outlook' in x:
        return 'outlook'
    elif 'linkedin' in x:
        return 'linkedin'
    elif 'pinterest' in x:
        return 'pinterest'
    elif 'ask' in x:
        return 'ask'
    elif 'siliconvalley' in x:
        return 'siliconvalley'
    elif 'lunametrics' in x:
        return 'lunametrics'
    elif 'amazon' in x:
        return 'amazon'
    elif 'mysearch' in x:
        return 'mysearch'
    elif 'qiita' in x:
        return 'qiita'
    elif 'messenger' in x:
        return 'messenger'
    elif 'twitter' in x:
        return 'twitter'
    elif 't.co' in x:
        return 't.co'
    elif 'vk.com' in x:
        return 'vk.com'
    elif 'search' in x:
        return 'search'
    elif 'edu' in x:
        return 'edu'
    elif 'mail' in x:
        return 'mail'
    elif 'ad' in x:
        return 'ad'
    elif 'golang' in x:
        return 'golang'
    elif 'direct' in x:
        return 'direct'
    elif 'dealspotr' in x:
        return 'dealspotr'
    elif 'sashihara' in x:
        return 'sashihara'
    elif 'phandroid' in x:
        return 'phandroid'
    elif 'baidu' in x:
        return 'baidu'
    elif 'mdn' in x:
        return 'mdn'
    elif 'duckduckgo' in x:
        return 'duckduckgo'
    elif 'seroundtable' in x:
        return 'seroundtable'
    elif 'metrics' in x:
        return 'metrics'
    elif 'sogou' in x:
        return 'sogou'
    elif 'businessinsider' in x:
        return 'businessinsider'
    elif 'github' in x:
        return 'github'
    elif 'gophergala' in x:
        return 'gophergala'
    elif 'yandex' in x:
        return 'yandex'
    elif 'msn' in x:
        return 'msn'
    elif 'dfa' in x:
        return 'dfa'
    elif '(not set)' in x:
        return '(not set)'
    elif 'feedly' in x:
        return 'feedly'
    elif 'arstechnica' in x:
        return 'arstechnica'
    elif 'squishable' in x:
        return 'squishable'
    elif 'flipboard' in x:
        return 'flipboard'
    elif 't-online.de' in x:
        return 't-online.de'
    elif 'sm.cn' in x:
        return 'sm.cn'
    elif 'wow' in x:
        return 'wow'
    elif 'baidu' in x:
        return 'baidu'
    elif 'partners' in x:
        return 'partners'
    else:
        return 'others'

train['device.browser'] = train['device.browser'].map(lambda x:browser_mapping(str(x).lower())).astype('str')
train['trafficSource.adContent'] = train['trafficSource.adContent'].map(lambda x:adcontents_mapping(str(x).lower())).astype('str')
train['trafficSource.source'] = train['trafficSource.source'].map(lambda x:source_mapping(str(x).lower())).astype('str')

test['device.browser'] = test['device.browser'].map(lambda x:browser_mapping(str(x).lower())).astype('str')
test['trafficSource.adContent'] = test['trafficSource.adContent'].map(lambda x:adcontents_mapping(str(x).lower())).astype('str')
test['trafficSource.source'] = test['trafficSource.source'].map(lambda x:source_mapping(str(x).lower())).astype('str')

def process_device(data_df):
    print("process device ...")
    data_df['source.country'] = data_df['trafficSource.source'] + '_' + data_df['geoNetwork.country']
    data_df['campaign.medium'] = data_df['trafficSource.campaign'] + '_' + data_df['trafficSource.medium']
    data_df['browser.category'] = data_df['device.browser'] + '_' + data_df['device.deviceCategory']
    data_df['browser.os'] = data_df['device.browser'] + '_' + data_df['device.operatingSystem']
    return data_df

train = process_device(train)
test = process_device(test)

def custom(data):
    print('custom..')
    data['device_deviceCategory_channelGrouping'] = data['device.deviceCategory'] + "_" + data['channelGrouping']
    data['channelGrouping_browser'] = data['device.browser'] + "_" + data['channelGrouping']
    data['channelGrouping_OS'] = data['device.operatingSystem'] + "_" + data['channelGrouping']
    
    for i in ['geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country','geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region','geoNetwork.subContinent']:
        for j in ['device.browser','device.deviceCategory', 'device.operatingSystem', 'trafficSource.source']:
            data[i + "_" + j] = data[i] + "_" + data[j]
    
    data['content.source'] = data['trafficSource.adContent'] + "_" + data['source.country']
    data['medium.source'] = data['trafficSource.medium'] + "_" + data['source.country']
    return data

train = custom(train)
test = custom(test)
excluded_features = [
    'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 
    'visitId', 'visitStartTime', 'vis_date', 'nb_sessions', 'max_visits'
]

categorical_features = [
    _f for _f in train.columns
    if (_f not in excluded_features) & (train[_f].dtype == 'object')
]

excluded_features = ['date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 'visitId', 'visitStartTime']

categorical_features = [ _f for _f in train.columns\
    if (_f not in excluded_features) & (train[_f].dtype == 'object')]

dropcolumns =  [c for c in categorical_features if train[c].nunique() == 1]
train = train[[_f for _f in train.columns if _f not in dropcolumns]]
test = test[[_f for _f in test.columns if _f not in dropcolumns]]

categorical_features = [ _f for _f in train.columns\
    if (_f not in excluded_features) & (train[_f].dtype == 'object')]

def process_device2(train,test):
    print("process device2 ...")
    train_objs_num = len(train)
    dataset = pd.concat(objs=[train, test], axis=0)
    for f in categorical_features:
        dataset[f], indexer = pd.factorize(dataset[f])
    train = dataset[:train_objs_num]
    test = dataset[train_objs_num:]
    del dataset
    return train, test

train, test = process_device2(train,test)
gc.collect()
print('load')

temp_train = pd.read_csv('../input/try-so-hard/temp_train_predictions.csv', 
                    dtype={'fullVisitorId': str}, nrows=None)
temp_test = pd.read_csv('../input/try-so-hard/temp_test_predictions.csv',
                   dtype={'fullVisitorId': str}, nrows=None)

print('finished')

y_reg = train['totals.transactionRevenue']

excluded_features = ['date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 'visitId', 'visitStartTime','predictions']
train_features = [_f for _f in train.columns if _f not in excluded_features]
trn_data = train[train_features + ['fullVisitorId']].groupby('fullVisitorId').mean()
trn_data.sort_index(inplace=True)
trn_pred_list = temp_train[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\
    .apply(lambda df: list(df.predictions))\
    .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})
trn_pred_list.sort_index(inplace=True)
trn_all_predictions = pd.DataFrame(list(trn_pred_list.values), index=trn_data.index)
trn_feats = trn_all_predictions.columns
trn_all_predictions['t_mean'] = np.log1p(trn_all_predictions[trn_feats].mean(axis=1))
trn_all_predictions['t_median'] = np.log1p(trn_all_predictions[trn_feats].median(axis=1))
trn_all_predictions['t_sum_log'] = np.log1p(trn_all_predictions[trn_feats]).sum(axis=1)
trn_all_predictions['t_sum_act'] = np.log1p(trn_all_predictions[trn_feats].fillna(0).sum(axis=1))
trn_all_predictions['t_nb_sess'] = trn_all_predictions[trn_feats].isnull().sum(axis=1)
full_data = pd.concat([trn_data, trn_all_predictions], axis=1)
del trn_data, trn_all_predictions, temp_train
gc.collect()
full_data.shape
sub_data = test[train_features + ['fullVisitorId']].groupby('fullVisitorId').mean()
sub_data.sort_index(inplace=True)
sub_pred_list = temp_test[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\
    .apply(lambda df: list(df.predictions))\
    .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})
sub_pred_list.sort_index(inplace=True)
sub_all_predictions = pd.DataFrame(list(sub_pred_list.values), index=sub_data.index)
for f in trn_feats:
    if f not in sub_all_predictions.columns:
        sub_all_predictions[f] = np.nan
sub_all_predictions['t_mean'] = np.log1p(sub_all_predictions[trn_feats].mean(axis=1))
sub_all_predictions['t_median'] = np.log1p(sub_all_predictions[trn_feats].median(axis=1))
sub_all_predictions['t_sum_log'] = np.log1p(sub_all_predictions[trn_feats]).sum(axis=1)
sub_all_predictions['t_sum_act'] = np.log1p(sub_all_predictions[trn_feats].fillna(0).sum(axis=1))
sub_all_predictions['t_nb_sess'] = sub_all_predictions[trn_feats].isnull().sum(axis=1)
sub_full_data = pd.concat([sub_data, sub_all_predictions], axis=1)
del sub_data, sub_all_predictions
gc.collect()
sub_full_data.shape
train['target'] = y_reg
trn_user_target = train[['fullVisitorId', 'target']].groupby('fullVisitorId').sum()
del train, test
gc.collect()
xgb_params = {
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'learning_rate': 0.02,
        'max_depth': 22,
        'min_child_weight': 57,
        'gamma' : 1.45,
        'alpha': 0.0,
        'lambda': 0.0,
        'subsample': 0.67,
        'colsample_bytree': 0.054,
        'colsample_bylevel': 0.50,
        'n_jobs': -1,
        'random_state': 456
    }

from xgboost import XGBRegressor
folds = get_folds(df=full_data[['totals.pageviews']].reset_index(), n_splits=5)

excluded_features = ['date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 'visitId', 'visitStartTime']
for x in range(40,300,1):
    excluded_features.append('pred_'+str(x))
full_features = [_f for _f in full_data.columns if _f not in excluded_features]
print(full_features)

oof_preds = np.zeros(full_data.shape[0])
oof_preds1 = np.zeros(full_data.shape[0])
both_oof = np.zeros(full_data.shape[0])
sub_preds = np.zeros(sub_full_data.shape[0])
lgb_temp = np.zeros(sub_full_data.shape[0])
xgb_temp = np.zeros(sub_full_data.shape[0])
feature_importance_df = pd.DataFrame()

for fold_, (trn_, val_) in enumerate(folds):
    print("-"* 20 + "Fold :"+str(fold_) + "-"* 20)
    trn_x, trn_y = full_data[full_features].iloc[trn_], trn_user_target['target'].iloc[trn_]
    val_x, val_y = full_data[full_features].iloc[val_], trn_user_target['target'].iloc[val_]
    xg = XGBRegressor(**xgb_params, n_estimators=20000)
    reg = lgb.LGBMRegressor(
        num_leaves=40,
        learning_rate=0.02,
        n_estimators=20000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    print("-"* 20 + "LightGBM Training" + "-"* 20)
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(trn_x, np.log1p(trn_y)), (val_x, np.log1p(val_y))],
        eval_names=['TRAIN', 'VALID'],
        early_stopping_rounds=50,
        eval_metric='rmse',
        verbose=100
    )
    print("-"* 20 + "Xgboost Training" + "-"* 20)
    xg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(trn_x, np.log1p(trn_y)), (val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        eval_metric='rmse',
        verbose=100
    )

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = trn_x.columns
    fold_importance_df["importance"] = reg.feature_importances_
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    oof_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_preds1[val_] = xg.predict(val_x)

    oof_preds[oof_preds < 0] = 0
    oof_preds1[oof_preds1 < 0] = 0

    both_oof[val_] = oof_preds[val_] * 0.6 + oof_preds1[val_] * 0.4

    # Make sure features are in the same order
    _preds = reg.predict(sub_full_data[full_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0

    pre = xg.predict(sub_full_data[full_features])
    pre[pre<0]=0

    sub_preds += (_preds / len(folds)) * 0.6 + (pre / len(folds)) * 0.4
    lgb_temp += (_preds / len(folds))
    xgb_temp += (pre / len(folds)) 
    # sub_preds += (_preds / len(folds)) 
    
def display_importances(feature_importance_df_,output_name):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature", as_index=False).mean().sort_values(by="importance", ascending=False)[:40]
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=cols.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features_' +output_name + '_(avg over folds)')
    plt.tight_layout()
    plt.savefig('importance_'+ output_name + '.png')
    plt
display_importances(feature_importance_df,"2nd train")
print("LGB  ", mean_squared_error(np.log1p(trn_user_target['target']), oof_preds) ** .5)
print("XGB  ", mean_squared_error(np.log1p(trn_user_target['target']), oof_preds1) ** .5)
print("Combine  ", mean_squared_error(np.log1p(trn_user_target['target']), both_oof) ** .5)


sub_full_data['PredictedLogRevenue'] = sub_preds
sub_full_data[['PredictedLogRevenue']].to_csv('futureisthere_gogogo3_both.csv', index=True)

sub_full_data['PredictedLogRevenue'] = lgb_temp
sub_full_data[['PredictedLogRevenue']].to_csv('futureisthere_gogogo3_lgb.csv', index=True)

sub_full_data['PredictedLogRevenue'] = xgb_temp
sub_full_data[['PredictedLogRevenue']].to_csv('futureisthere_gogogo3_xgb.csv', index=True)
futureishere_result = pd.read_csv('../input/future-is-here/new_test.csv', 
                    dtype={'fullVisitorId': str}, nrows=None)
# print(sub_full_data)

sub_full_data = sub_full_data.reset_index()

sub_full_data['PredictedLogRevenue_lgb+xgb'] = sub_preds
sub_full_data['PredictedLogRevenue_lgb'] = lgb_temp
sub_full_data['PredictedLogRevenue_xgb'] = xgb_temp
sub_full_data['PredictedLogRevenue_futureishere'] = futureishere_result['PredictedLogRevenue']

# print(futureishere_result)
# print(sub_full_data)

sub1 = sub_full_data[["fullVisitorId", "PredictedLogRevenue_lgb+xgb"]]
sub2 = sub_full_data[["fullVisitorId", "PredictedLogRevenue_lgb"]]
sub3 = sub_full_data[["fullVisitorId", "PredictedLogRevenue_xgb"]]
sub4 = sub_full_data[["fullVisitorId", "PredictedLogRevenue_futureishere"]]

df_base = pd.merge(sub1,sub2,how='inner',on='fullVisitorId')
df_base = pd.merge(df_base,sub3,how='inner',on='fullVisitorId')
df_base = pd.merge(df_base,sub4,how='inner',on='fullVisitorId')
plt.figure(figsize=(6,6))
sns.heatmap(df_base.iloc[:,1:].corr(),annot=True,fmt=".2f")
# plt.show()
df_base.iloc[:,1:].corr()
sub_df = pd.DataFrame({"fullVisitorId":sub1['fullVisitorId']})
sub_df["PredictedLogRevenue"] = df_base['PredictedLogRevenue_futureishere'] * 0.6 + df_base['PredictedLogRevenue_lgb+xgb'] * 0.4
# sub_df["PredictedLogRevenue"] = sub_df["PredictedLogRevenue"]
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df.to_csv('blend3itall.csv', index=False)