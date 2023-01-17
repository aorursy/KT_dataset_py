# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
import json
from pandas.io.json import json_normalize
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
# Loading data and flattening JSON columns
def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     # Set the date, fullVisitorId, sessionId as string of constant
                     dtype={'date': str, 'fullVisitorId': str, 'sessionId': str}, 
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(list(df[column]))
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
%%time
train_df = load_df()
test_df = load_df('../input/test.csv')
def na_detect(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum() / df.isnull().count() * 100 ).sort_values(ascending = False)
    df_opt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    plt.figure(figsize=(20,20))
    fig, ax = plt.subplots()
    col_na = total[total>0]
    bar_na = ax.barh(col_na.index, col_na.values, 0.8)
    for i, v in enumerate(col_na.values):
        ax.text(v + 5, i - .15 , str(v), color='red')#, fontweight='bold')
    plt.title('Variables with Missing Value')
    plt.xlabel('Quantity of Missing Value')
    plt.ylabel('Columns')
    plt.show()
    
    print (df_opt[~(df_opt['Total'] == 0)])
    
    return
na_detect(train_df)
set(train_df.columns).difference(set(test_df.columns))
def date_convert(df):
    df['visitdate'] = pd.to_datetime(df['visitStartTime'], unit='s')
    #df['visitdate'] = pd.datetime.utcfromtimestamp(test_df['visitStartTime'])
    df['wday'] = df['visitdate'].dt.weekday
    df['hour'] = df['visitdate'].dt.hour
    df['day'] = df['visitdate'].dt.day
    df['month'] = df['visitdate'].dt.month
    return    
for df in [train_df, test_df]:
    date_convert(df)
print('TrainSet:', train_df.shape)
print('TestSet:', test_df.shape)
def constant_process(df):
    num_constant = 0
    constant_cols = []
    for col in df.columns:
        if df[col].nunique()==1:
            constant_cols.append(col)
            num_constant = num_constant+1
            
    print('Number of Constant Variables:', num_constant)
    print(constant_cols)
    df = df.drop(constant_cols, axis=1)
    print('Shape: ', df.shape)
    return df
ctrain_df = constant_process(train_df)
ctest_df = constant_process(test_df)
print('Unique Variables in Train:', ctrain_df['sessionId'].nunique())
print('Unique Variables in Test:', ctest_df['sessionId'].nunique())
dup_session = ctrain_df[ctrain_df.duplicated(subset='sessionId', keep=False)].sort_values('sessionId',ascending = False)
dup_session.head(2)
ctrain_df["totals.transactionRevenue"].fillna(0, inplace=True)
ctrain_df['totals.transactionRevenue'] = ctrain_df['totals.transactionRevenue'].astype(int)
ctrain_df['totals.hits'] = ctrain_df['totals.hits'].astype(int)
ctrain_df['totals.pageviews'] = ctrain_df['totals.hits'].astype(int)
ctest_df['totals.hits'] = ctest_df['totals.hits'].astype(int)
ctest_df['totals.pageviews'] = ctest_df['totals.hits'].astype(int)
plt.figure(figsize=(14,5))
plt.subplot(1,2,2)
ax = sns.distplot(np.log1p(ctrain_df[ctrain_df['totals.transactionRevenue'] > 0]["totals.transactionRevenue"]), kde=True)
ax.set_xlabel('Transaction Revenue Log', fontsize=15)
ax.set_ylabel('Distribuition', fontsize=15)
ax.set_title("Distribuition of Revenue Log", fontsize=20)
plt.subplot(1,2,1)
sns.distplot(ctrain_df["totals.transactionRevenue"], kde=True)
plt.xlabel('Transaction Revenue', fontsize=15)
plt.ylabel('Distribuition', fontsize=15)
plt.title("Distribuition of Revenue", fontsize=20)
valid_df = ctrain_df[ctrain_df['totals.transactionRevenue'] > 0]
na_detect(valid_df)
ctrain_df = ctrain_df.drop(['trafficSource.adContent', 'trafficSource.adwordsClickInfo.adNetworkType', 'trafficSource.adwordsClickInfo.gclId', 'trafficSource.adwordsClickInfo.page', 'trafficSource.adwordsClickInfo.slot'], axis=1)
print('Train Shape: ' ,ctrain_df.shape)
ctest_df = ctest_df.drop(['trafficSource.adContent', 'trafficSource.adwordsClickInfo.adNetworkType', 'trafficSource.adwordsClickInfo.gclId', 'trafficSource.adwordsClickInfo.page', 'trafficSource.adwordsClickInfo.slot'], axis=1)
print('Test Shape:' ,ctest_df.shape)
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
sns.distplot(ctrain_df["totals.hits"], kde=True)
plt.xlabel('Hits', fontsize=15)
plt.ylabel('Distribuition', fontsize=15)
plt.title("Distribuition of Hits", fontsize=20)
plt.subplot(1,2,2)
sns.distplot(ctrain_df["totals.pageviews"], kde=True)
plt.xlabel('Page Views', fontsize=15)
plt.ylabel('Distribuition', fontsize=15)
plt.title("Distribuition of Page Views", fontsize=20)
ctrain_df["totals.pageviews"].fillna(value=ctrain_df['totals.pageviews'].median(), inplace=True)
ctest_df["totals.pageviews"].fillna(value=ctest_df['totals.pageviews'].median(), inplace=True)
plt.figure(figsize=(10,5))
sns.distplot(ctrain_df["visitNumber"], kde=True)
plt.xlabel('Visit Number', fontsize=15)
plt.ylabel('Distribuition', fontsize=15)
plt.title("Distribuition of Visit Number", fontsize=20)
non_relevant = ["date", "fullVisitorId", "sessionId", "visitId", "visitStartTime", "visitdate", "totals.transactionRevenue"]
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

categorical_cols = [c for c in ctrain_df.columns if not c.startswith("total")]
categorical_cols = [c for c in categorical_cols if c not in non_relevant]
for c in categorical_cols:

    le = LabelEncoder()
    train_vals = list(ctrain_df[c].values.astype(str))
    test_vals = list(ctest_df[c].values.astype(str))
    
    le.fit(train_vals + test_vals)
    
    ctrain_df[c] = le.transform(train_vals)
    ctest_df[c] = le.transform(test_vals)
train_y = ctrain_df['totals.transactionRevenue']
del ctrain_df['totals.transactionRevenue']
def get_folds(df=None, n_splits=5):
    unique_sessions = np.array(sorted(df['sessionId'].unique()))
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for dev_s, val_s in folds.split(X=unique_sessions, y=unique_sessions, groups=unique_sessions):
        fold_ids.append(
            [
                ids[df['sessionId'].isin(unique_sessions[dev_s])],
                ids[df['sessionId'].isin(unique_sessions[val_s])]
            ]
        )

    return fold_ids

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn import metrics
%%time
features = [f for f in ctrain_df.columns if f not in non_relevant]
print(features)

folds = get_folds(df=ctrain_df, n_splits=5)

importances = pd.DataFrame()
dev_reg_preds = np.zeros(ctrain_df.shape[0])
val_reg_preds = np.zeros(ctest_df.shape[0])

for f, (dev, val) in enumerate(folds):
    dev_x, dev_y = ctrain_df[features].iloc[dev], train_y.iloc[dev]
    val_x, val_y = ctrain_df[features].iloc[val], train_y.iloc[val]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.03,
        n_estimators=1000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    
    reg.fit(
        dev_x, np.log1p(dev_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    
    importance_df = pd.DataFrame()
    importance_df['feature'] = features
    importance_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    importance_df['fold'] = f + 1
    importances = pd.concat([importances, importance_df], axis=0, sort=False)
    dev_reg_preds[val] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    dev_reg_preds[dev_reg_preds < 0] = 0
    preds = reg.predict(ctest_df[features], num_iteration=reg.best_iteration_)
    preds[preds < 0] = 0
    val_reg_preds += np.expm1(preds)/len(folds)
print('RMSE=' ,metrics.mean_squared_error(np.log1p(train_y), dev_reg_preds) ** .5)

val_reg_preds.shape
import warnings
warnings.simplefilter('ignore', FutureWarning)

importances['gain_log'] = np.log1p(importances['gain'])
mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])

plt.figure(figsize=(8, 12))
sns.barplot(x='gain_log', y='feature', data=importances.sort_values('mean_gain', ascending=False))
ctest_df["PredictedLogRevenue"] = val_reg_preds
submission = ctest_df.groupby("fullVisitorId").agg({"PredictedLogRevenue" : "sum"}).reset_index()
submission["PredictedLogRevenue"] = np.log1p(submission["PredictedLogRevenue"])
submission["PredictedLogRevenue"] =  submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
submission.to_csv("baseline.csv", index=False)
submission.head()