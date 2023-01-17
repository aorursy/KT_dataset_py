import gc

import os

import time

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import xgboost as xgb

import lightgbm as lgb

from scipy import stats

from scipy.signal import hann

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from scipy.signal import hilbert

from scipy.signal import convolve

from sklearn.svm import NuSVR, SVR

from catboost import CatBoostRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.model_selection import KFold,StratifiedKFold, RepeatedKFold

from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone





warnings.filterwarnings("ignore")
IS_LOCAL = False

if(IS_LOCAL):

    PATH="../input/LANL/"

else:

    PATH="../input/"

os.listdir(PATH)
print("There are {} files in test folder".format(len(os.listdir(os.path.join(PATH, 'test' )))))
%%time

train_df = pd.read_csv(os.path.join(PATH,'train.csv'), 

                       dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

train_df_save = train_df.copy
mean_acoustic = np.mean(train_df.acoustic_data)

train_df.acoustic_data = train_df.acoustic_data - mean_acoustic

print(mean_acoustic)

print (round(np.mean(train_df.acoustic_data),2))
print("Train: rows:{} cols:{}".format(train_df.shape[0], train_df.shape[1]))
pd.options.display.precision = 15

train_df.head(10)
train_ad_sample_df = train_df['acoustic_data'].values[::50]

train_ttf_sample_df = train_df['time_to_failure'].values[::50]



def plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title="Acoustic data and time to failure: 1% sampled data"):

    fig, ax1 = plt.subplots(figsize=(12, 8))

    plt.title(title)

    plt.plot(train_ad_sample_df, color='r')

    ax1.set_ylabel('acoustic data', color='r')

    plt.legend(['acoustic data'], loc=(0.01, 0.95))

    ax2 = ax1.twinx()

    plt.plot(train_ttf_sample_df, color='b')

    ax2.set_ylabel('time to failure', color='b')

    plt.legend(['time to failure'], loc=(0.01, 0.9))

    plt.grid(True)



plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df)

del train_ad_sample_df

del train_ttf_sample_df
gc.collect()

#train_ad_sample_df = train_df['acoustic_data'].values[:6291455]

#train_ttf_sample_df = (train_df['time_to_failure'].values[:6291455])

train_ad_sample_df = train_df['acoustic_data'].values[:50580000]

train_ttf_sample_df = (train_df['time_to_failure'].values[:50580000])



plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title="Acoustic data and time to failure: 1st 2 quakes")

del train_ad_sample_df

del train_ttf_sample_df
rows = 150000

segments = int(np.floor(train_df.shape[0] / rows))

print("Number of segments: ", segments)
def add_trend_feature(arr, abs_values=False):

    idx = np.array(range(len(arr)))

    if abs_values:

        arr = np.abs(arr)

    lr = LinearRegression()

    lr.fit(idx.reshape(-1, 1), arr)

    return lr.coef_[0]



def classic_sta_lta(x, length_sta, length_lta):

    sta = np.cumsum(x ** 2)

    # Convert to float

    sta = np.require(sta, dtype=np.float)

    # Copy for LTA

    lta = sta.copy()

    # Compute the STA and the LTA

    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]

    sta /= length_sta

    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]

    lta /= length_lta

    # Pad zeros

    sta[:length_lta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float

    dtiny = np.finfo(0.0).tiny

    idx = lta < dtiny

    lta[idx] = dtiny

    return sta / lta
gc.collect()
train_X = pd.DataFrame(index=range(segments), dtype=np.float64)



train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])

# These may be needed later

need_aggregated_features = False

if need_aggregated_features:

    total_mean = train_df['acoustic_data'].mean()

    total_std = train_df['acoustic_data'].std()

    total_max = train_df['acoustic_data'].max()

    total_min = train_df['acoustic_data'].min()

    total_sum = train_df['acoustic_data'].sum()

    total_abs_sum = np.abs(train_df['acoustic_data']).sum()
train_X.shape, train_y.shape
def create_features(seg_id, seg, X):

    xc = pd.Series(seg['acoustic_data'].values)

    zc = np.fft.fft(xc)

    X.loc[seg_id, 'mean'] = xc.mean()

    X.loc[seg_id, 'std'] = xc.std()

    X.loc[seg_id, 'max'] = xc.max()

    

    #FFT transform values

    realFFT = np.real(zc)

    imagFFT = np.imag(zc)

    

    X.loc[seg_id, 'Rmean'] = realFFT.mean()

    X.loc[seg_id, 'Rstd'] = realFFT.std()

    X.loc[seg_id, 'Rmax'] = realFFT.max()

    X.loc[seg_id, 'Rmin'] = realFFT.min()

    X.loc[seg_id, 'Imean'] = imagFFT.mean()

    X.loc[seg_id, 'Istd'] = imagFFT.std()

    X.loc[seg_id, 'Imax'] = imagFFT.max()

    X.loc[seg_id, 'Imin'] = imagFFT.min()

    X.loc[seg_id, 'Rmean_last_5000'] = realFFT[-5000:].mean()

    X.loc[seg_id, 'Rstd__last_5000'] = realFFT[-5000:].std()

    X.loc[seg_id, 'Rmax_last_5000'] = realFFT[-5000:].max()

    X.loc[seg_id, 'Rmin_last_5000'] = realFFT[-5000:].min()

    X.loc[seg_id, 'Rmean_last_15000'] = realFFT[-15000:].mean()

    X.loc[seg_id, 'Rstd_last_15000'] = realFFT[-15000:].std()

    X.loc[seg_id, 'Rmax_last_15000'] = realFFT[-15000:].max()

    X.loc[seg_id, 'Rmin_last_15000'] = realFFT[-15000:].min()

    X.loc[seg_id, 'mean_change_abs'] = np.mean(np.diff(xc))

    X.loc[seg_id, 'mean_change_rate'] = np.mean(np.nonzero((np.diff(xc) / xc[:-1]))[0])

    

    X.loc[seg_id, 'abs_max'] = np.abs(xc).max()

    X.loc[seg_id, 'abs_min'] = np.abs(xc).min()

    X.loc[seg_id, 'std_first_50000'] = xc[:50000].std()

    X.loc[seg_id, 'std_last_50000'] = xc[-50000:].std()

    X.loc[seg_id, 'std_first_10000'] = xc[:10000].std()

    X.loc[seg_id, 'std_last_10000'] = xc[-10000:].std()



    X.loc[seg_id, 'avg_first_50000'] = xc[:50000].mean()

    X.loc[seg_id, 'avg_last_50000'] = xc[-50000:].mean()

    X.loc[seg_id, 'avg_first_10000'] = xc[:10000].mean()

    X.loc[seg_id, 'avg_last_10000'] = xc[-10000:].mean()

   

    X.loc[seg_id, 'min_first_50000'] = xc[:50000].min()

    X.loc[seg_id, 'min_last_50000'] = xc[-50000:].min()

    X.loc[seg_id, 'min_first_10000'] = xc[:10000].min()

    X.loc[seg_id, 'min_last_10000'] = xc[-10000:].min()

    

    X.loc[seg_id, 'max_first_50000'] = xc[:50000].max()

    X.loc[seg_id, 'max_last_50000'] = xc[-50000:].max()

    X.loc[seg_id, 'max_first_10000'] = xc[:10000].max()

    X.loc[seg_id, 'max_last_10000'] = xc[-10000:].max()



    #X.loc[seg_id, 'max_to_min'] = xc.max() / np.abs(xc.min())

    X.loc[seg_id, 'max_to_min_diff'] = xc.max() - np.abs(xc.min())

    

    X.loc[seg_id, 'count_big'] = len(xc[np.abs(xc) > 500])

    X.loc[seg_id, 'sum'] = xc.sum()

    

    X.loc[seg_id, 'mean_change_rate_first_50000'] = np.mean(np.nonzero((np.diff(xc[:50000]) / xc[:50000][:-1]))[0])

    X.loc[seg_id, 'mean_change_rate_last_50000'] = np.mean(np.nonzero((np.diff(xc[-50000:]) / xc[-50000:][:-1]))[0])

    X.loc[seg_id, 'mean_change_rate_first_10000'] = np.mean(np.nonzero((np.diff(xc[:10000]) / xc[:10000][:-1]))[0])

    X.loc[seg_id, 'mean_change_rate_last_10000'] = np.mean(np.nonzero((np.diff(xc[-10000:]) / xc[-10000:][:-1]))[0])

    

    X.loc[seg_id, 'q95'] = np.quantile(xc, 0.95)

    X.loc[seg_id, 'q99'] = np.quantile(xc, 0.99)

    X.loc[seg_id, 'q05'] = np.quantile(xc, 0.05)

    X.loc[seg_id, 'q01'] = np.quantile(xc, 0.01)



    X.loc[seg_id, 'abs_q95'] = np.quantile(np.abs(xc), 0.95)

    X.loc[seg_id, 'abs_q99'] = np.quantile(np.abs(xc), 0.99)

    X.loc[seg_id, 'abs_q05'] = np.quantile(np.abs(xc), 0.05)

    X.loc[seg_id, 'abs_q01'] = np.quantile(np.abs(xc), 0.01)

    

    X.loc[seg_id, 'trend'] = add_trend_feature(xc)

    X.loc[seg_id, 'abs_trend'] = add_trend_feature(xc, abs_values=True)

    X.loc[seg_id, 'abs_mean'] = np.abs(xc).mean()

    X.loc[seg_id, 'abs_std'] = np.abs(xc).std()

    

    X.loc[seg_id, 'mad'] = xc.mad()

    

    X.loc[seg_id, 'Moving_average_700_mean'] = xc.rolling(window=700).mean().mean(skipna=True)

    X.loc[seg_id, 'Moving_average_1500_mean'] = xc.rolling(window=1500).mean().mean(skipna=True)

    X.loc[seg_id, 'Moving_average_3000_mean'] = xc.rolling(window=3000).mean().mean(skipna=True)

    X.loc[seg_id, 'Moving_average_6000_mean'] = xc.rolling(window=6000).mean().mean(skipna=True)

    ewma = pd.Series.ewm

    X.loc[seg_id, 'exp_Moving_average_300_mean'] = (ewma(xc, span=300).mean()).mean(skipna=True)

    X.loc[seg_id, 'exp_Moving_average_3000_mean'] = ewma(xc, span=3000).mean().mean(skipna=True)

    X.loc[seg_id, 'exp_Moving_average_30000_mean'] = ewma(xc, span=6000).mean().mean(skipna=True)

    no_of_std = 2

    X.loc[seg_id, 'MA_700MA_std_mean'] = xc.rolling(window=700).std().mean()

    X.loc[seg_id,'MA_700MA_BB_high_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X.loc[seg_id, 'MA_700MA_std_mean']).mean()

    X.loc[seg_id,'MA_700MA_BB_low_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X.loc[seg_id, 'MA_700MA_std_mean']).mean()

    X.loc[seg_id, 'MA_400MA_std_mean'] = xc.rolling(window=400).std().mean()

    X.loc[seg_id,'MA_400MA_BB_high_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X.loc[seg_id, 'MA_400MA_std_mean']).mean()

    X.loc[seg_id,'MA_400MA_BB_low_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X.loc[seg_id, 'MA_400MA_std_mean']).mean()

    X.loc[seg_id, 'MA_1000MA_std_mean'] = xc.rolling(window=1000).std().mean()

    

    X.loc[seg_id, 'iqr'] = np.subtract(*np.percentile(xc, [75, 25]))

    X.loc[seg_id, 'q999'] = np.quantile(xc,0.999)

    X.loc[seg_id, 'q001'] = np.quantile(xc,0.001)

    X.loc[seg_id, 'ave10'] = stats.trim_mean(xc, 0.1)



    for windows in [5, 10, 50, 100, 500, 1000, 5000, 10000]:

    

        x_roll_std = xc.rolling(windows).std().dropna().values

        x_roll_mean = xc.rolling(windows).mean().dropna().values

        X.loc[seg_id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()

        X.loc[seg_id, 'std_roll_std_' + str(windows)] = x_roll_std.std()

        X.loc[seg_id, 'max_roll_std_' + str(windows)] = x_roll_std.max()

        X.loc[seg_id, 'min_roll_std_' + str(windows)] = x_roll_std.min()

        X.loc[seg_id, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)

        X.loc[seg_id, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)

        X.loc[seg_id, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)

        X.loc[seg_id, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)

        X.loc[seg_id, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))

        X.loc[seg_id, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])

        X.loc[seg_id, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        

        X.loc[seg_id, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()

        X.loc[seg_id, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()

        X.loc[seg_id, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()

        X.loc[seg_id, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()

        X.loc[seg_id, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)

        X.loc[seg_id, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)

        X.loc[seg_id, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)

        X.loc[seg_id, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)

        X.loc[seg_id, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))

        X.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])

        X.loc[seg_id, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()  

    
# iterate over all segments

for seg_id in tqdm_notebook(range(segments)):

    seg = train_df.iloc[seg_id*rows:seg_id*rows+rows]

    create_features(seg_id, seg, train_X)

    # the y value is the last entry in the time to failure in the segment

    train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]
train_X_save = train_X.copy

train_y_save = train_y.copy

train_y.head(5)

# We will not train on the segments with a quake, because there are likely outliers

train_y_quake = np.nonzero(np.diff(train_y.time_to_failure) > 0)[0] + 1

print(len(train_y_quake))

print (len(train_y))



for idx in train_y_quake: 

    train_y.drop([idx],inplace=True)

    train_X.drop([idx],inplace = True)

#np.abs(train_X.corrwith(train_y)).sort_values(ascending=False).head(12)

train_X.to_csv('train_features.csv', index=False)

train_y.to_csv('train_y.csv', index=False)

train_X.shape, train_y.shape

train_X.head(), train_y.head()
scaler = StandardScaler()

scaler.fit(train_X)

scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)

#scaled_train_X = train_X
scaled_train_X.head(10)
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

test_X = pd.DataFrame(columns=train_X.columns, dtype=np.float64, index=submission.index)
submission.shape, test_X.shape
for seg_id in tqdm_notebook(test_X.index):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    # convert to mean 0 of the training dataset 

    # seg_mean = np.mean(seg.acoustic_data)

    seg.acoustic_data = seg.acoustic_data - mean_acoustic

    create_features(seg_id, seg, test_X)
# save before scaling

test_X.to_csv('test_features.csv', index=False)
scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)

#scaled_test_X = test_X

scaled_test_X.values[1117]
scaled_test_X.shape
scaled_test_X.tail(10)
n_fold = 5

def mae_cv (model):

    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42).get_n_splits(scaled_train_X.values)

    mae = -cross_val_score (model, scaled_train_X.values, train_y, scoring="neg_mean_absolute_error",

                           verbose=0,

                           cv=folds)

    return mae

%%time



lgb_params = {'num_leaves': 51,

         'min_data_in_leaf': 10, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.001,

         "boosting": "gbdt",

         "feature_fraction": 0.91,

         "bagging_freq": 1,

         "bagging_fraction": 0.91,

         "bagging_seed": 42,

         "metric": 'mae',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": -1,

         "random_state": 42}





lgb_model = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.01, n_estimators=720,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11, n_jobs = -1)



score = mae_cv(lgb_model)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

lgb_model
lgb_gamma_model = lgb.LGBMRegressor(objective='gamma',num_leaves=5,

                              learning_rate=0.01, n_estimators=720,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11, n_jobs = -1)



score = mae_cv(lgb_gamma_model)

print("LGBM - gamma score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

lgb_gamma_model
%%time 

xgb_params = {'eta': 0.03,

              'max_depth': 9,

              'subsample': 0.85,

              'objective': 'reg:linear',

              'eval_metric': 'mae',

              'silent': True,

              'nthread': 4}

    

xgb_model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1, eval_metric = 'mae',)



score = mae_cv(xgb_model)

print("XGB score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

xgb_model



#    xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, 

#                          verbose_eval=500, params=xgb_params)
%%time

rf_model = RandomForestRegressor(n_estimators=120, n_jobs=-1, min_samples_leaf=1, 

                           max_features = "auto",max_depth=15, )

score = mae_cv(rf_model)

print("Random Forest score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

rf_model
%%time

params = {'loss_function':'MAE',}

cat_model = CatBoostRegressor(iterations=1000,  eval_metric='MAE', verbose=False, **params)



score = mae_cv(cat_model)

print("Cat Boost score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

cat_model
%%time

KRR_model = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

score = mae_cv(KRR_model)

print("Kernel Ridge score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

print (score)

KRR_model
%%time

#ENet_model = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3,max_iter=5000))

ENet_model = ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3,max_iter=5000)

score = mae_cv(ENet_model)

print("Elastic Net score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

ENet_model
%%time

lasso_model = Lasso(alpha =0.0005, random_state=1)

score = mae_cv(lasso_model)

print("Lasso score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

lasso_model
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)   
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # We again fit the data on clones of the original models

    def fit(self, X, y):

        print (type(X))

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        print (KFold)

        # Train cloned base models then create out-of-fold predictions

        # that are needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X.iloc[train_index], y.iloc[train_index])

                y_pred = instance.predict(X.iloc[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

   

    #Do the predictions of all base models on the test data and use the averaged predictions as 

    #meta-features for the final prediction which is done by the meta-model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)
class StackingCVRegressorRetrained(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, regressors, meta_regressor, n_folds=5, use_features_in_secondary=False):

        self.regressors = regressors

        self.meta_regressor = meta_regressor

        self.n_folds = n_folds

        self.use_features_in_secondary = use_features_in_secondary



    def fit(self, X, y):

        self.regr_ = [clone(x) for x in self.regressors]

        self.meta_regr_ = clone(self.meta_regressor)



        kfold = KFold(n_splits=self.n_folds, shuffle=True)



        out_of_fold_predictions = np.zeros((X.shape[0], len(self.regressors)))



        # Create out-of-fold predictions for training meta-model

        for i, regr in enumerate(self.regr_):

            for train_idx, holdout_idx in kfold.split(X, y):

                instance = clone(regr)

                instance.fit(X[train_idx], y[train_idx])

                out_of_fold_predictions[holdout_idx, i] = instance.predict(X[holdout_idx])



        # Train meta-model

        if self.use_features_in_secondary:

            self.meta_regr_.fit(np.hstack((X, out_of_fold_predictions)), y)

        else:

            self.meta_regr_.fit(out_of_fold_predictions, y)

        

        # Retrain base models on all data

        for regr in self.regr_:

            regr.fit(X, y)



        return self



    def predict(self, X):

        meta_features = np.column_stack([

            regr.predict(X) for regr in self.regr_

        ])



        if self.use_features_in_secondary:

            return self.meta_regr_.predict(np.hstack((X, meta_features)))

        else:

            return self.meta_regr_.predict(meta_features)

%%time

#averaged_models = AveragingModels(models = (rf_model, xgb_model, KRR_model, lgb_model, ENet_model, cat_model, lasso_model))

#averaged_models = AveragingModels(models = (rf_model, lgb_model,  cat_model, lasso_model))

averaged_models = AveragingModels(models = (rf_model,cat_model))



score =mae_cv(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#averaged_models.fit (scaled_train_X.values, train_y)
averaged_models.fit (scaled_train_X.values, train_y)

averaged2_train_predict = averaged_models.predict(scaled_train_X.values)

print(mean_absolute_error(train_y, averaged2_train_predict))





averaged_prediction = np.zeros(len(scaled_test_X))

averaged_prediction += averaged_models.predict(scaled_test_X.values)

averaged_prediction
%%time

stacked_predict = StackingAveragedModels(base_models =(rf_model, xgb_model, lgb_model, cat_model,ENet_model), 

                                          meta_model =lasso_model) 

stacked_predict.fit(scaled_train_X, train_y)

stacked_train_pred = stacked_predict.predict(scaled_train_X)



print(mean_absolute_error(train_y, stacked_train_pred))



stacked_prediction = np.zeros(len(scaled_test_X))

stacked_prediction += stacked_predict.predict(scaled_test_X)**1.0

stacked_prediction[0:4]

submission.time_to_failure = averaged_prediction

submission.to_csv('submissionV30_averaged_cat_rf.csv',index=True)

submission.time_to_failure = stacked_prediction

submission.to_csv('submissionV30_stacked.csv',index=True)