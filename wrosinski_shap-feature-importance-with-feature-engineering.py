import gc
import glob
import os
import random
import time
from datetime import date, datetime

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn import model_selection
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

%matplotlib inline
pd.set_option("display.max_columns", 96)
pd.set_option("display.max_rows", 96)

plt.rcParams['figure.figsize'] = (12, 9)
plt.style.use('ggplot')

shap.initjs()
# Size of the dataset is limited to just n_debug_samples, because SHAP calculation is quite time-consuming.

debug = True
n_debug_samples = 10000
from multiprocessing import Pool


def create_lag(df_code, n_lag=[3, 7, 14, ], shift_size=1):
    code = df_code['assetCode'].unique()

    for col in return_features:
        for window in n_lag:
            rolled = df_code[col].shift(shift_size).rolling(window=window)
            lag_mean = rolled.mean()
            lag_max = rolled.max()
            lag_min = rolled.min()
            lag_std = rolled.std()
            df_code['%s_lag_%s_mean' % (col, window)] = lag_mean
            df_code['%s_lag_%s_max' % (col, window)] = lag_max
            df_code['%s_lag_%s_min' % (col, window)] = lag_min

    return df_code.fillna(-1)


def generate_lag_features(df, n_lag=[3, 7, 14]):
    
    
    features = ['time', 'assetCode', 'assetName', 'volume', 'close', 'open',
                'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
                'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
                'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
                'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
                'returnsOpenNextMktres10', 'universe']

    assetCodes = df['assetCode'].unique()
    print(assetCodes)
    all_df = []
    df_codes = df.groupby('assetCode')
    df_codes = [df_code[1][['time', 'assetCode'] + return_features]
                for df_code in df_codes]
    print('total %s df' % len(df_codes))

    pool = Pool(4)
    all_df = pool.map(create_lag, df_codes)

    new_df = pd.concat(all_df)
    new_df.drop(return_features, axis=1, inplace=True)
    pool.close()

    return new_df


def mis_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data


def data_prep(market_train):
    lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())}
    market_train['assetCodeT'] = market_train['assetCode'].map(lbl)
    market_train = market_train.dropna(axis=0)
    return market_train


def exp_loss(p, y):
    y = y.get_label()
    grad = -y * (1.0 - 1.0 / (1.0 + np.exp(-y * p)))
    hess = -(np.exp(y * p) * (y * p - 1) - 1) / ((np.exp(y * p) + 1)**2)
    return grad, hess
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
print('Done!')

(market_train_df, news_train_df) = env.get_training_data()
market_train_df['time'] = market_train_df['time'].dt.date
market_train_df = market_train_df.loc[market_train_df['time'] >= date(
    2010, 1, 1)]


return_features = ['returnsClosePrevMktres10',
                   'returnsClosePrevRaw10', 'open', 'close']
n_lag = [3, 7, 14]
new_df = generate_lag_features(market_train_df, n_lag=n_lag)
market_train_df = pd.merge(market_train_df, new_df,
                           how='left', on=['time', 'assetCode'])

market_train_df = mis_impute(market_train_df)
market_train_df = data_prep(market_train_df)
if debug:
    market_train_df = market_train_df.iloc[:n_debug_samples, :]


up = market_train_df['returnsOpenNextMktres10'] >= 0
universe = market_train_df['universe'].values
d = market_train_df['time']


fcol = [c for c in market_train_df if c not in [
    'assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences',
    'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider',
    'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x',
    'universe', 'sourceTimestamp']]


X = market_train_df[fcol]
up = up.values
r = market_train_df.returnsOpenNextMktres10.values

mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
X = 1 - ((maxs - X) / rng)
assert X.shape[0] == up.shape[0] == r.shape[0]
X_train, X_test, up_train, up_test, r_train, r_test, u_train, u_test, d_train, d_test = model_selection.train_test_split(
    X, up, r, universe, d, test_size=0.25, random_state=99)


train_cols = X_train.columns.tolist()

train_data = lgb.Dataset(X_train, label=up_train.astype(int), 
                         feature_name=train_cols)
test_data = lgb.Dataset(X_test, label=up_test.astype(int), 
                        feature_name=train_cols, reference=train_data)
# LGB parameters:
params = {'learning_rate': 0.05,
          'boosting': 'gbdt', 
          'objective': 'binary',
          'num_leaves': 2000,
          'min_data_in_leaf': 200,
          'max_bin': 200,
          'max_depth': 16,
          'seed': 2018,
          'nthread': 10,}


# LGB training:
lgb_model = lgb.train(params, train_data, 
                      num_boost_round=1000, 
                      valid_sets=(test_data,), 
                      valid_names=('valid',), 
                      verbose_eval=25, 
                      early_stopping_rounds=20)
# DF, based on which importance is checked
X_importance = X_test

# Explain model predictions using shap library:
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_importance)
# Plot summary_plot
shap.summary_plot(shap_values, X_importance)
# Plot summary_plot as barplot:
shap.summary_plot(shap_values, X_importance, plot_type='bar')
X_importance.returnsClosePrevRaw10_lag_3_mean.value_counts()
plt.hist(X_importance.returnsClosePrevRaw10_lag_3_mean, bins=100)
shap.dependence_plot("returnsClosePrevRaw10_lag_3_mean", shap_values, X_importance)
shap.dependence_plot("volume", shap_values, X_importance)
X_interaction = X_importance.iloc[:500,:]

shap_interaction_values = shap.TreeExplainer(lgb_model).shap_interaction_values(X_interaction)
shap.summary_plot(shap_interaction_values, X_interaction)
# Raw dependence plot:

shap.dependence_plot(
    "returnsClosePrevRaw10_lag_3_mean",
    shap_values, X_importance)
# Interaction values dependence plot capturing main effects:

shap.dependence_plot(
    ("returnsClosePrevRaw10_lag_3_mean", "returnsClosePrevRaw10_lag_3_mean"),
    shap_interaction_values, X_interaction)
# Interaction values dependence plot capturing interaction effects:

shap.dependence_plot(
    ("returnsClosePrevRaw10_lag_3_mean", "returnsOpenPrevMktres10"),
    shap_interaction_values, X_interaction)
shap_sum = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame([X_importance.columns.tolist(), shap_sum.tolist()]).T
importance_df.columns = ['column_name', 'shap_importance']
importance_df = importance_df.sort_values('shap_importance', ascending=False)
importance_df
