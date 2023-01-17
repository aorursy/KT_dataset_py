import os
import gc
import datetime

import pandas as pd #Analysis 
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis 
from scipy.stats import norm #Analysis 
from sklearn.preprocessing import StandardScaler #Analysis 
from scipy import stats #Analysis 
%matplotlib inline

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from scipy.optimize import minimize

import lightgbm as lgb
import catboost as cb
import xgboost as xgb

import warnings 
warnings.filterwarnings('ignore')

print(os.listdir('../input'))
train = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
print("shape of train : ",train.shape)

test = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])
print("shape of test : ",test.shape)
def missing_impute(df):
    for i in df.columns:
        if df[i].dtype == "object":
            df[i] = df[i].fillna("other")
        elif (df[i].dtype == "int64" or df[i].dtype == "float64"):
            df[i] = df[i].fillna(df[i].mean())
        else:
            pass
    return df


def datetime_extract(df, dt_col='first_active_month'):
    df['date'] = df[dt_col].dt.date 
    df['day'] = df[dt_col].dt.day 
    df['dayofweek'] = df[dt_col].dt.dayofweek
    df['dayofyear'] = df[dt_col].dt.dayofyear
    df['days_in_month'] = df[dt_col].dt.days_in_month
    df['daysinmonth'] = df[dt_col].dt.daysinmonth 
    df['month'] = df[dt_col].dt.month
    df['week'] = df[dt_col].dt.week 
    df['weekday'] = df[dt_col].dt.weekday
    df['weekofyear'] = df[dt_col].dt.weekofyear
    # df['year'] = train[dt_col].dt.year
    
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['date']).dt.days

    return df
# Do impute missing values for train & test
for df in [train, test]:
    missing_impute(df)
    
# Do extract datetime values for train & test
train = datetime_extract(train, dt_col='first_active_month')
test = datetime_extract(test, dt_col='first_active_month')
train.shape, test.shape
train.head()
ht = pd.read_csv("../input/historical_transactions.csv")
print("shape of historical_transactions: ", ht.shape)
ht['authorized_flag'] = ht['authorized_flag'].map({'Y':1, 'N':0})
# Do impute missing values for history
ht = missing_impute(ht)
def aggregate_historical_transactions(history):
    
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).\
                                      astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'city_id': ['nunique'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std', 'mean', 'size'],
        'installments': ['sum', 'median', 'max', 'min', 'std', 'mean', 'size'],
        'purchase_date': [np.ptp],
        'month_lag': ['min', 'max']
        }
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['hist_' + '_'.join(col).strip() 
                           for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='hist_transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history

history = aggregate_historical_transactions(ht)
del ht
gc.collect()
history.head()
train.shape, test.shape
train = pd.merge(train, history, on='card_id', how='left')
test = pd.merge(test, history, on='card_id', how='left')
merchant = pd.read_csv("../input/merchants.csv")
print("shape of merchant: ", merchant.shape)

new_merchant = pd.read_csv("../input/new_merchant_transactions.csv")
print("shape of new_merchant_transactions: ", new_merchant.shape)
new_merchant['authorized_flag'] = new_merchant['authorized_flag'].map({'Y':1, 'N':0})
# Do impute missing values for merchant and new_merchant
for df in [merchant, new_merchant]:
    missing_impute(df)
def aggregate_new_transactions(new_trans):    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'city_id': ['nunique'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std', 'mean', 'size'],
        'installments': ['sum', 'median', 'max', 'min', 'std', 'mean', 'size'],
        'month_lag': ['min', 'max']
        }
    agg_new_trans = new_trans.groupby(['card_id']).agg(agg_func)
    agg_new_trans.columns = ['new_' + '_'.join(col).strip() 
                           for col in agg_new_trans.columns.values]
    agg_new_trans.reset_index(inplace=True)
    
    df = (new_trans.groupby('card_id')
          .size()
          .reset_index(name='new_transactions_count'))
    
    agg_new_trans = pd.merge(df, agg_new_trans, on='card_id', how='left')
    
    return agg_new_trans

new_trans = aggregate_new_transactions(new_merchant)
train = pd.merge(train, new_trans, on='card_id', how='left')
test = pd.merge(test, new_trans, on='card_id', how='left')
# The target
target = train['target']
excluded_features = ['first_active_month', 'card_id', 'target', 'date']
use_cols = [col for col in train.columns if col not in excluded_features]

train = train[use_cols]
test = test[use_cols]

features = list(train[use_cols].columns)
categorical_feats = [col for col in features if 'feature_' in col]
from sklearn.preprocessing import LabelEncoder
for col in categorical_feats:
    print(col)
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))
df_all = pd.concat([train, test])
df_all = pd.get_dummies(df_all, columns=categorical_feats)

len_train = train.shape[0]

train = df_all[:len_train]
test = df_all[len_train:]
# Check missing again
train.isnull().sum()
train.info()
# Final fill missing values
for col in train.columns:
    for df in [train, test]:
        if df[col].dtype == "float64":
            print(col)
            df[col] = df[col].fillna(df[col].mean())
lgb_params = {"objective" : "regression", "metric" : "rmse", 
               "max_depth": 9, "min_child_samples": 20, 
               "reg_alpha": 1, "reg_lambda": 1,
               "num_leaves" : 64, "learning_rate" : 0.005, 
               "subsample" : 0.8, "colsample_bytree" : 0.8, 
               "verbosity": -1}

FOLDs = KFold(n_splits=10, shuffle=True, random_state=100)

oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))

features_lgb = list(train.columns)
feature_importance_df_lgb = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(train)):
    trn_data = lgb.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx], label=target.iloc[val_idx])

    print("LGB " + str(fold_) + "-" * 50)
    num_round = 10000
    clf = lgb.train(lgb_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 50)
    oof_lgb[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)

    fold_importance_df_lgb = pd.DataFrame()
    fold_importance_df_lgb["feature"] = features_lgb
    fold_importance_df_lgb["importance"] = clf.feature_importance()
    fold_importance_df_lgb["fold"] = fold_ + 1
    feature_importance_df_lgb = pd.concat([feature_importance_df_lgb, fold_importance_df_lgb], axis=0)
    predictions_lgb += clf.predict(test, num_iteration=clf.best_iteration) / FOLDs.n_splits
    
print(np.sqrt(mean_squared_error(oof_lgb, target)))
cols = (feature_importance_df_lgb[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df_lgb.loc[feature_importance_df_lgb.feature.isin(cols)]

plt.figure(figsize=(14,14))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')
FOLDs = KFold(n_splits=10, shuffle=True, random_state=100)
X = train
y = target

oof_cb = np.zeros(len(train))
predictions_cb = np.zeros(len(test))

for n_fold, (trn_idx, val_idx) in enumerate(FOLDs.split(X, y)):
    X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
    X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]
    
    # CatBoost Regressor estimator
    model = cb.CatBoostRegressor(
        learning_rate = 0.03,
        iterations = 1000,
        eval_metric = 'RMSE',
        allow_writing_files = False,
        od_type = 'Iter',
        bagging_temperature = 0.2,
        depth = 10,
        od_wait = 20,
        silent = True
    )
    
    # Fit
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=None,
        early_stopping_rounds=100
    )
    
    print("CB " + str(n_fold) + "-" * 50)
    
    oof_cb[val_idx] = model.predict(X_valid)
    test_preds = model.predict(test)
    predictions_cb += test_preds / FOLDs.n_splits

print(np.sqrt(mean_squared_error(oof_cb, target)))
xgb_params = {'eta': 0.005, 'max_depth': 9, 'subsample': 0.8, 'colsample_bytree': 0.8, 
          'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True}

FOLDs = KFold(n_splits=10, shuffle=True, random_state=100)

oof_xgb = np.zeros(len(train))
predictions_xgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(train)):
    trn_data = xgb.DMatrix(data=train.iloc[trn_idx], label=target.iloc[trn_idx])
    val_data = xgb.DMatrix(data=train.iloc[val_idx], label=target.iloc[val_idx])
    watchlist = [(trn_data, 'train'), (val_data, 'valid')]
    print("XGB " + str(fold_) + "-" * 50)
    num_round = 10000
    xgb_model = xgb.train(xgb_params, trn_data, num_round, watchlist, early_stopping_rounds=50, verbose_eval=1000)
    oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(train.iloc[val_idx]), ntree_limit=xgb_model.best_ntree_limit+50)

    predictions_xgb += xgb_model.predict(xgb.DMatrix(test), ntree_limit=xgb_model.best_ntree_limit+50) / FOLDs.n_splits

print(np.sqrt(mean_squared_error(oof_xgb, target)))
def find_best_weight(preds, target):
    def _validate_func(weights):
        ''' scipy minimize will pass the weights as a numpy array '''
        final_prediction = 0
        for weight, prediction in zip(weights, preds):
                final_prediction += weight * prediction
        return np.sqrt(mean_squared_error(final_prediction, target))

    #the algorithms need a starting value, right not we chose 0.5 for all weights
    #its better to choose many random starting points and run minimize a few times
    starting_values = [0.5]*len(preds)

    #adding constraints and a different solver as suggested by user 16universe
    #https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    #our weights are bound between 0 and 1
    bounds = [(0, 1)] * len(preds)
    
    res = minimize(_validate_func, starting_values, method='Nelder-Mead', bounds=bounds, constraints=cons)
    
    print('Ensemble Score: {best_score}'.format(best_score=(1-res['fun'])))
    print('Best Weights: {weights}'.format(weights=res['x']))
    
    return res
print('lgb', np.sqrt(mean_squared_error(oof_lgb, target)))
print('xgb', np.sqrt(mean_squared_error(oof_xgb, target)))
print('cb', np.sqrt(mean_squared_error(oof_cb, target)))
res = find_best_weight([oof_lgb, oof_cb, oof_xgb], target)
total_sum = 0.35864667 * oof_lgb + 0.59360413 * oof_cb + 0.14343413 * oof_xgb
print("CV score: {:<8.5f}".format(np.sqrt(mean_squared_error(total_sum, target))))
sub_df = pd.read_csv('../input/sample_submission.csv')
sub_df["target"] = 0.35864667 * predictions_lgb + 0.59360413 * predictions_cb + 0.14343413 * predictions_xgb
sub_df.to_csv("submission_ensemble.csv", index=False)