# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/2019-2nd-ml-month-with-kakr"))

print(os.listdir("../input/2019-2nd-ml-month-with-kakr-neighbor-stat"))

print(os.listdir("../input/11th-solution-data-public-98316-private-99336"))



# Any results you write to the current directory are saved as output.
from IPython.display import Image

Image('../input/11th-solution-data-public-98316-private-99336/stacking_in_practice_1.png')
Image('../input/11th-solution-data-public-98316-private-99336/stacking_in_practice_2.png')
import sys

import os

import re

import warnings

import time

import gc

import random as rn



from timeit import default_timer as timer

from datetime import date, datetime, timedelta

from functools import wraps



import pandas as pd

import numpy as np



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler, RobustScaler

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor

from sklearn.linear_model import ElasticNet, Lasso, Ridge

from sklearn.svm import SVR

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans



import xgboost as xgb

import lightgbm as lgb

import catboost as cat



import keras

from keras.models import Sequential, Model

from keras.layers import Dense, Embedding, Reshape, Concatenate, Input, Flatten

from keras import optimizers

from keras import initializers

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras import backend as K

import tensorflow as tf

from tensorflow import set_random_seed



import scipy

from scipy import stats

from scipy.stats import norm, skew #for some statistics

from scipy.cluster import hierarchy as hc

from scipy.special import boxcox1p



from tqdm import tqdm_notebook as tqdm



#RANDOM_SEED = 42

os.environ['PYTHONHASHSEED'] = '0'

RANDOM_SEED = 0

rn.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

set_random_seed(RANDOM_SEED)



session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,

                              inter_op_parallelism_threads=1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

K.set_session(sess)



def plot_numeric_for_regression(df, field, target_field='price'):

    df = df[df[field].notnull()]



    fig = plt.figure(figsize = (16, 7))

    ax1 = plt.subplot(121)

    

    sns.distplot(df[df['data'] == 'train'][field], label='Train', hist_kws={'alpha': 0.5}, ax=ax1)

    sns.distplot(df[df['data'] == 'test'][field], label='Test', hist_kws={'alpha': 0.5}, ax=ax1)



    plt.xlabel(field)

    plt.ylabel('Density')

    plt.legend()

    

    ax2 = plt.subplot(122)

    

    df_copy = df[df['data'] == 'train'].copy()



    sns.scatterplot(x=field, y=target_field, data=df_copy, ax=ax2)

    

    plt.show()

    

def plot_categorical_for_regression(df, field, target_field='price', show_missing=True, missing_value='NA'):

    df_copy = df.copy()

    if show_missing: df_copy[field] = df_copy[field].fillna(missing_value)

    df_copy = df_copy[df_copy[field].notnull()]



    ax1_param = 121

    ax2_param = 122

    fig_size = (16, 7)

    if df_copy[field].nunique() > 30:

        ax1_param = 211

        ax2_param = 212

        fig_size = (16, 10)

    

    fig = plt.figure(figsize = fig_size)

    ax1 = plt.subplot(ax1_param)

    

    sns.countplot(x=field, hue='data', order=np.sort(df_copy[field].unique()), data=df_copy)

    plt.xticks(rotation=90, fontsize=11)

    

    ax2 = plt.subplot(ax2_param)

    

    df_copy = df_copy[df_copy['data'] == 'train']



    sns.boxplot(x=field, y=target_field, data=df_copy, order=np.sort(df_copy[field].unique()), ax=ax2)

    plt.xticks(rotation=90, fontsize=11)

    

    plt.show()



def get_prefix(group_col, target_col, prefix=None):

    if isinstance(group_col, list) is True:

        g = '_'.join(group_col)

    else:

        g = group_col

    if isinstance(target_col, list) is True:

        t = '_'.join(target_col)

    else:

        t = target_col

    if prefix is not None:

        return prefix + '_' + g + '_' + t

    return g + '_' + t

    

def groupby_helper(df, group_col, target_col, agg_method, prefix_param=None):

    try:

        prefix = get_prefix(group_col, target_col, prefix_param)

        #print(group_col, target_col, agg_method)

        group_df = df.groupby(group_col)[target_col].agg(agg_method)

        group_df.columns = ['{}_{}'.format(prefix, m) for m in agg_method]

    except BaseException as e:

        print(e)

    return group_df.reset_index()



def rmse(y_true, y_pred):

    return np.sqrt(mean_squared_error(y_true, y_pred))



def rmse_exp(y_true, y_pred):

    return np.sqrt(mean_squared_error(np.expm1(y_true), np.expm1(y_pred)))



def time_decorator(func): 

    @wraps(func)

    def wrapper(*args, **kwargs):

        print("\nStartTime: ", datetime.now() + timedelta(hours=9))

        start_time = time.time()

        

        df = func(*args, **kwargs)

        

        print("EndTime: ", datetime.now() + timedelta(hours=9))  

        print("TotalTime: ", time.time() - start_time)

        return df

        

    return wrapper



class SklearnWrapper(object):

    def __init__(self, clf, params=None, **kwargs):

        #if isinstance(SVR) is False:

        #    params['random_state'] = kwargs.get('seed', 0)

        self.clf = clf(**params)

        self.is_classification_problem = True

        self.use_avg_oof = kwargs.get('use_avg_oof', False)

    @time_decorator

    def train(self, x_train, y_train, x_cross=None, y_cross=None):

        if len(np.unique(y_train)) > 30:

            self.is_classification_problem = False

            

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        if self.is_classification_problem is True:

            return self.clf.predict_proba(x)[:,1]

        else:

            return self.clf.predict(x)

    

class XgbWrapper(object):

    def __init__(self, params=None, **kwargs):

        self.param = params

        self.use_avg_oof = kwargs.get('use_avg_oof', False)

        self.num_rounds = kwargs.get('num_rounds', 1000)

        self.early_stopping = kwargs.get('ealry_stopping', 100)



        self.eval_function = kwargs.get('eval_function', None)

        self.verbose_eval = kwargs.get('verbose_eval', 100)

        self.best_round = 0

    

    @time_decorator

    def train(self, x_train, y_train, x_cross=None, y_cross=None):

        need_cross_validation = True

       

        if isinstance(y_train, pd.DataFrame) is True:

            y_train = y_train[y_train.columns[0]]

            if y_cross is not None:

                y_cross = y_cross[y_cross.columns[0]]



        if x_cross is None:

            dtrain = xgb.DMatrix(x_train, label=y_train, silent= True)

            train_round = self.best_round

            if self.best_round == 0:

                train_round = self.num_rounds

            

            self.clf = xgb.train(self.param, dtrain, train_round)

            del dtrain

        else:

            dtrain = xgb.DMatrix(x_train, label=y_train, silent=True)

            dvalid = xgb.DMatrix(x_cross, label=y_cross, silent=True)

            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]



            self.clf = xgb.train(self.param, dtrain, self.num_rounds, watchlist, feval=self.eval_function,

                                 early_stopping_rounds=self.early_stopping,

                                 verbose_eval=self.verbose_eval)

            self.best_round = max(self.best_round, self.clf.best_iteration)



    def predict(self, x):

        return self.clf.predict(xgb.DMatrix(x), ntree_limit=self.best_round)



    def get_params(self):

        return self.param    

    

class LgbmWrapper(object):

    def __init__(self, params=None, **kwargs):

        self.param = params

        self.use_avg_oof = kwargs.get('use_avg_oof', False)

        self.num_rounds = kwargs.get('num_rounds', 1000)

        self.early_stopping = kwargs.get('ealry_stopping', 100)



        self.eval_function = kwargs.get('eval_function', None)

        self.verbose_eval = kwargs.get('verbose_eval', 100)

        self.best_round = 0

        

    @time_decorator

    def train(self, x_train, y_train, x_cross=None, y_cross=None):

        """

        x_cross or y_cross is None

        -> model train limted num_rounds

        

        x_cross and y_cross is Not None

        -> model train using validation set

        """

        if isinstance(y_train, pd.DataFrame) is True:

            y_train = y_train[y_train.columns[0]]

            if y_cross is not None:

                y_cross = y_cross[y_cross.columns[0]]



        if x_cross is None:

            dtrain = lgb.Dataset(x_train, label=y_train, silent= True)

            train_round = self.best_round

            if self.best_round == 0:

                train_round = self.num_rounds

                

            self.clf = lgb.train(self.param, train_set=dtrain, num_boost_round=train_round)

            del dtrain   

        else:

            dtrain = lgb.Dataset(x_train, label=y_train, silent=True)

            dvalid = lgb.Dataset(x_cross, label=y_cross, silent=True)

            self.clf = lgb.train(self.param, train_set=dtrain, num_boost_round=self.num_rounds, valid_sets=[dtrain, dvalid],

                                  feval=self.eval_function, early_stopping_rounds=self.early_stopping,

                                  verbose_eval=self.verbose_eval)

            self.best_round = max(self.best_round, self.clf.best_iteration)

            del dtrain, dvalid

            

        gc.collect()

    

    def predict(self, x):

        return self.clf.predict(x, num_iteration=self.clf.best_iteration)

    

    def plot_importance(self, importance_type='gain', max_num_features=20):

        lgb.plot_importance(self.clf, importance_type=importance_type, max_num_features=max_num_features, height=0.7, figsize=(10,30))

        plt.show()

        

    def get_params(self):

        return self.param



class CatWrapper(object):

    def __init__(self, params=None, **kwargs):

        self.param = params

        self.use_avg_oof = kwargs.get('use_avg_oof', False)

        self.num_rounds = kwargs.get('num_rounds', 1000)

        self.param['iterations'] = kwargs.get('num_rounds', 1000)

        self.early_stopping = kwargs.get('ealry_stopping', 100)



        self.eval_function = kwargs.get('eval_function', None)

        self.verbose_eval = kwargs.get('verbose_eval', 100)

        self.best_round = 0

        

    @time_decorator

    def train(self, x_train, y_train, x_cross=None, y_cross=None, cat_features=None):

        """

        x_cross or y_cross is None

        -> model train limted num_rounds

        

        x_cross and y_cross is Not None

        -> model train using validation set

        """

        if isinstance(y_train, pd.DataFrame) is True:

            y_train = y_train[y_train.columns[0]]

            if y_cross is not None:

                y_cross = y_cross[y_cross.columns[0]]



        if x_cross is None:

            dtrain = cat.Pool(x_train, y_train, cat_features=cat_features)

            train_round = self.best_round

            if self.best_round == 0:

                train_round = self.num_rounds

                

            self.clf = cat.CatBoost(params=self.param)

            self.clf.fit(dtrain, verbose_eval=self.verbose_eval)

            del dtrain   

        else:

            dtrain = cat.Pool(x_train, y_train, cat_features=cat_features)

            dvalid = cat.Pool(x_cross, y_cross, cat_features=cat_features)

            

            self.clf = cat.CatBoost(params=self.param)

            self.clf.fit(dtrain, eval_set=[dvalid], early_stopping_rounds=self.early_stopping, verbose_eval=self.verbose_eval)

            self.best_round = max(self.best_round, self.clf.best_iteration_)

            del dtrain, dvalid

            

        gc.collect()

    

    def predict(self, x):

        if self.clf.best_iteration_ is None: return self.clf.predict(x, ntree_end=self.best_round)

        else: return self.clf.predict(x, ntree_end=self.clf.best_iteration_)

        

    def get_params(self):

        return self.param

    

class KerasWrapper(object):

    def __init__(self, model_func, params=None, **kwargs):

        self.model_func = model_func

        self.param = params

        self.use_avg_oof = kwargs.get('use_avg_oof', False)

        self.epochs = kwargs.get('epochs', 20)

        self.batch_size = kwargs.get('batch_size', 16)

        self.callbacks = kwargs.get('callbacks', None)

        self.shuffle = kwargs.get('shuffle', True)

        self.best_epochs = 0

    @time_decorator

    def train(self, x_train, y_train, x_cross=None, y_cross=None):

        self.model = self.model_func(x_train.shape[1])

        if x_cross is None:

            train_epochs = self.best_epochs

            if self.best_epochs == 0:

                train_epochs = self.epochs

                

            self.model.fit(x_train, y_train, epochs=train_epochs, batch_size=self.batch_size,

                           shuffle=self.shuffle, callbacks=self.callbacks, verbose=0)

        else:

            hist = self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,

                                  shuffle=self.shuffle, validation_data=(x_cross, y_cross),

                                  callbacks=self.callbacks, verbose=0)

            self.best_epochs = max(self.best_epochs, len(hist.history['val_loss']))



    def predict(self, x):

        if isinstance(x, pd.DataFrame):

            return self.model.predict(x.values).ravel()

        else:

            return self.model.predict(x).ravel()

    

    def get_params(self):

        return self.param



class KerasEmbeddingWrapper(object):

    def __init__(self, model_func, params=None, **kwargs):

        self.model_func = model_func

        self.param = params

        self.use_avg_oof = kwargs.get('use_avg_oof', False)

        self.epochs = kwargs.get('epochs', 20)

        self.batch_size = kwargs.get('batch_size', 16)

        self.callbacks = kwargs.get('callbacks', None)

        self.embedding_cols = kwargs.get('embedding_cols', None)

        self.shuffle = kwargs.get('shuffle', True)

        self.best_epochs = 0

    @time_decorator

    def train(self, x_train, y_train, x_cross=None, y_cross=None):

        non_embedding_cols = [col for col in x_train.columns if col not in self.embedding_cols]

        self.model = self.model_func(x_train, self.embedding_cols)

        if x_cross is None:

            train_epochs = self.best_epochs

            if self.best_epochs == 0:

                train_epochs = self.epochs

                

            x_tr_list = []

            x_tr_list.append(x_train[non_embedding_cols])

            for col in self.embedding_cols:

                x_tr_list.append(x_train[col])

            self.model.fit(x_tr_list, y_train, epochs=train_epochs, batch_size=self.batch_size,

                           shuffle=self.shuffle, callbacks=self.callbacks, verbose=0)

        else:

            x_tr_list = []

            x_tr_list.append(x_train[non_embedding_cols])

            for col in self.embedding_cols:

                x_tr_list.append(x_train[col])



            x_cr_list = []

            x_cr_list.append(x_cross[non_embedding_cols])

            for col in self.embedding_cols:

                x_cr_list.append(x_cross[col])

            hist = self.model.fit(x_tr_list, y_train, epochs=self.epochs, batch_size=self.batch_size, shuffle=self.shuffle,

                                  validation_data=(x_cr_list, y_cross), callbacks=self.callbacks, verbose=0)

            self.best_epochs = max(self.best_epochs, len(hist.history['val_loss']))



    def predict(self, x):

        non_embedding_cols = [col for col in x.columns if col not in self.embedding_cols]

        x_list = []

        x_list.append(x[non_embedding_cols])

        for col in self.embedding_cols:

            x_list.append(x[col])

        return self.model.predict(x_list).ravel()

    

    def get_params(self):

        return self.param





@time_decorator

def get_oof(clf, x_train, y_train, x_test, eval_func, **kwargs):

    nfolds = kwargs.get('NFOLDS', 5)

    kfold_shuffle = kwargs.get('kfold_shuffle', True)

    kfold_random_state = kwargs.get('kfold_random_state', 0)

    stratified_kfold_ytrain = kwargs.get('stratifed_kfold_y_value', None)

    ntrain = x_train.shape[0]

    ntest = x_test.shape[0]

    

    kf_split = None

    if stratified_kfold_ytrain is None:

        kf = KFold(n_splits=nfolds, shuffle=kfold_shuffle, random_state=kfold_random_state)

        kf_split = kf.split(x_train)

    else:

        kf = StratifiedKFold(n_splits=nfolds, shuffle=kfold_shuffle, random_state=kfold_random_state)

        kf_split = kf.split(x_train, stratified_kfold_ytrain)

        

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))



    cv_sum = 0

    

    # before running model, print model param

    # lightgbm model and xgboost model use get_params()

    try:

        if clf.clf is not None:

            print(clf.clf)

    except:

        print(clf)

        print(clf.get_params())



    for i, (train_index, cross_index) in enumerate(kf_split):

        x_tr, x_cr = None, None

        y_tr, y_cr = None, None

        if isinstance(x_train, pd.DataFrame):

            x_tr, x_cr = x_train.iloc[train_index], x_train.iloc[cross_index]

            y_tr, y_cr = y_train.iloc[train_index], y_train.iloc[cross_index]

        else:

            x_tr, x_cr = x_train[train_index], x_train[cross_index]

            y_tr, y_cr = y_train[train_index], y_train[cross_index]



        clf.train(x_tr, y_tr, x_cr, y_cr)

        

        oof_train[cross_index] = clf.predict(x_cr)

        if hasattr(clf, 'use_avg_oof') and clf.use_avg_oof:

            oof_test += clf.predict(x_test)/nfolds



        cv_score = eval_func(y_cr, oof_train[cross_index])

        

        print('Fold %d / ' % (i+1), 'CV-Score: %.6f' % cv_score)

        cv_sum = cv_sum + cv_score

        

        del x_tr, x_cr, y_tr, y_cr

        

    gc.collect()

    

    score = cv_sum / nfolds

    print("Average CV-Score: ", score)



    # Using All Dataset, retrain

    if not hasattr(clf, 'use_avg_oof') or clf.use_avg_oof is False:

        clf.train(x_train, y_train)

        oof_test = clf.predict(x_test)



    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1), score



@time_decorator

def stacking(data_list, y_train, model_list, eval_func=None, nfolds=5, kfold_random_state=RANDOM_SEED):

    

    oof_train_list = []

    oof_test_list = []

    oof_cv_score_list = []

    

    for X_train, X_test in data_list:

        print(X_train.shape, X_test.shape, y_train.shape)

        for model in model_list:

            oof_train, oof_test, oof_cv_score = get_oof(model, X_train, y_train, X_test, eval_func,

                                                        NFOLDS=nfolds, kfold_random_state=kfold_random_state)

            oof_train_list.append(oof_train)

            oof_test_list.append(oof_test)

            oof_cv_score_list.append(oof_cv_score)

        

    X_train_next = pd.DataFrame(np.concatenate(oof_train_list, axis=1))

    X_test_next = pd.DataFrame(np.concatenate(oof_test_list, axis=1))

    

    print(X_train_next.shape, X_test_next.shape)

    

    return X_train_next, X_test_next, oof_cv_score_list





def load_data(nb_1km=True, nb_3km=True, nb_5km=True,

              n_5_nb=True, n_10_nb=True, n_20_nb=True,

              original=True, do_scale=False, fix_skew=False, do_ohe=True):

    train = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/train.csv')

    test = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/test.csv')



    train_copy = train.copy()

    train_copy['data'] = 'train'

    test_copy = test.copy()

    test_copy['data'] = 'test'

    test_copy['price'] = np.nan

    

    # remove outlier

    train_copy = train_copy[~((train_copy['sqft_living'] > 12000) & (train_copy['price'] < 3000000))].reset_index(drop=True)



    # concat train, test data to preprocess

    data = pd.concat([train_copy, test_copy]).reset_index(drop=True)

    data = data[train_copy.columns]

    

    # fix skew feature

    skew_columns = ['price']



    for c in skew_columns:

        data[c] = np.log1p(data[c])

    

    if original:

        # feature engineering

        data['date'] = pd.to_datetime(data['date'])

        data['yr_mo_sold'] = data['date'].dt.strftime('%Y-%m')

        data['yr_sold'] = data['date'].dt.year

        data['qt_sold'] = data['date'].dt.quarter

        data['week_sold'] = data['date'].dt.week

        data['dow_sold'] = data['date'].dt.dayofweek

        data['yr_sold - yr_built'] = data['yr_sold'] - data['yr_built']

        data['yr_sold - yr_renovated'] = data['yr_sold'] - data['yr_renovated']

        data['yr_renovated - yr_built'] = data['yr_renovated'] - data['yr_built']

        

        data['yr_sold'] = data['yr_sold'].astype(str)

        data['qt_sold'] = data['qt_sold'].astype(str)

        data['week_sold'] = data['week_sold'].astype(str)

        data['dow_sold'] = data['dow_sold'].astype(str)

        data.drop(['date'], axis=1, inplace=True)



        data['bedrooms + bathrooms'] = data['bedrooms'] + data['bathrooms']

        data['bathrooms / bedrooms'] = data['bathrooms'] / data['bedrooms']

        data.loc[np.isinf(data['bathrooms / bedrooms']), 'bathrooms / bedrooms'] = 0

        data['bathrooms / bedrooms'].fillna(0, inplace=True)

        data['sqft_living / bedrooms'] = data['sqft_living'] / data['bedrooms']

        data.loc[np.isinf(data['sqft_living / bedrooms']), 'sqft_living / bedrooms'] = 0

        data['sqft_living / bathrooms'] = data['sqft_living'] / data['bathrooms']

        data.loc[np.isinf(data['sqft_living / bathrooms']), 'sqft_living / bathrooms'] = 0

        data['sqft_living / floors'] = data['sqft_living'] / data['floors']

        data.loc[np.isinf(data['sqft_living / floors']), 'sqft_living / floors'] = 0

        data['sqft_lot / sqft_living'] = data['sqft_lot'] / data['sqft_living']

        data['sqft_basement / sqft_above'] = data['sqft_basement'] / data['sqft_above']

        data['sqft_lot15 / sqft_living15'] = data['sqft_lot15'] / data['sqft_living15']

        data['has_basement'] = data['sqft_basement'] > 0

        data['is_renovated'] = data['yr_renovated'] > 0

        data['sqft_living_changed'] = data['sqft_living'] != data['sqft_living15']

        data['sqft_lot_changed'] = data['sqft_lot'] != data['sqft_lot15']

        data['sqft_living * grade'] = data['sqft_living'] * data['grade']

        data['overall'] = data['grade'] + data['view'] + data['condition'] + data['waterfront'] + data['has_basement'] + data['is_renovated']

        data['sqft_living * overall'] = data['sqft_living'] * data['overall']



        data['zipcode'] = data['zipcode'].astype(str)

        data['zipcode-3'] = data['zipcode'].str[2:3]

        data['zipcode-4'] = data['zipcode'].str[3:4]

        data['zipcode-5'] = data['zipcode'].str[4:5]

        data['zipcode-34'] = data['zipcode'].str[2:4]

        data['zipcode-45'] = data['zipcode'].str[3:5]

        data['zipcode-35'] = data['zipcode-3'] + data['zipcode-5']



        # pca for lat, long

        coord = data[['lat','long']]

        pca = PCA(n_components=2)

        pca.fit(coord)



        coord_pca = pca.transform(coord)



        data['coord_pca1'] = coord_pca[:, 0]

        data['coord_pca2'] = coord_pca[:, 1]



        # kmeans for lat, long

        kmeans = KMeans(n_clusters=72, random_state=RANDOM_SEED).fit(coord)

        coord_cluster = kmeans.predict(coord)

        data['coord_cluster'] = coord_cluster

        data['coord_cluster'] = data['coord_cluster'].map(lambda x: 'cluster_' + str(x).rjust(2, '0'))



        lat_med = data['lat'].median()

        long_med = data['long'].median()



        lat2 = data['lat'].values

        long2 = data['long'].values



        bearing_arr = bearing_array(lat_med, long_med, lat2, long2)



        data['bearing_from_center'] = bearing_arr



        qcut_count = 10

        data['qcut_bearing'] = pd.qcut(data['bearing_from_center'], qcut_count, labels=range(qcut_count))

        data['qcut_bearing'] = data['qcut_bearing'].astype(str)



        # calculate grouped price

        group_cols = ['grade','bedrooms','bathrooms','view','condition','waterfront']



        for col in group_cols:

            group_df = groupby_helper(data[data['data'] == 'train'], col, 'price', ['mean']).fillna(0)

            data = data.merge(group_df, on=col, how='left').fillna(0)

            

        cat_cols = [

            'yr_mo_sold','yr_sold','qt_sold','week_sold','dow_sold','coord_cluster',

            'zipcode','zipcode-3','zipcode-4','zipcode-5','zipcode-34','zipcode-45','zipcode-35',

            'qcut_bearing'

        ]

            

        if do_ohe:

            for col in cat_cols:

                ohe_df = pd.get_dummies(data[[col]], prefix='ohe_'+col)

                data.drop(col, axis=1, inplace=True)

                data = pd.concat([data, ohe_df], axis=1)

        else:

            for col in cat_cols:

                le = LabelEncoder()

                data[col] = le.fit_transform(data[col])

    else:

        data = data[['id','price','data']]

        

    if nb_1km:

        neighbor_1km_stat = pd.read_csv('../input/2019-2nd-ml-month-with-kakr-neighbor-stat/neighbor_1km_stat.csv')

        data = data.merge(neighbor_1km_stat, on='id', how='left').fillna(0)

        

        if original:

            data['sqft_living - nb_1km_sqft_living_mean'] = data['sqft_living'] - data['nb_1km_sqft_living_mean']

            data['sqft_lot - nb_1km_sqft_lot_mean'] = data['sqft_lot'] - data['nb_1km_sqft_lot_mean']

            data['bedrooms - nb_1km_bedrooms_mean'] = data['bedrooms'] - data['nb_1km_bedrooms_mean']

            data['bathrooms - nb_1km_bathrooms_mean'] = data['bathrooms'] - data['nb_1km_bathrooms_mean']

            data['grade - nb_1km_grade_mean'] = data['grade'] - data['nb_1km_grade_mean']

            data['view - nb_1km_view_mean'] = data['view'] - data['nb_1km_view_mean']

            data['condition - nb_1km_condition_mean'] = data['condition'] - data['nb_1km_condition_mean']

    if nb_3km:

        neighbor_3km_stat = pd.read_csv('../input/2019-2nd-ml-month-with-kakr-neighbor-stat/neighbor_3km_stat.csv')

        data = data.merge(neighbor_3km_stat, on='id', how='left').fillna(0)

        

        if original:

            data['sqft_living - nb_3km_sqft_living_mean'] = data['sqft_living'] - data['nb_3km_sqft_living_mean']

            data['sqft_lot - nb_3km_sqft_lot_mean'] = data['sqft_lot'] - data['nb_3km_sqft_lot_mean']

            data['bedrooms - nb_3km_bedrooms_mean'] = data['bedrooms'] - data['nb_3km_bedrooms_mean']

            data['bathrooms - nb_3km_bathrooms_mean'] = data['bathrooms'] - data['nb_3km_bathrooms_mean']

            data['grade - nb_3km_grade_mean'] = data['grade'] - data['nb_3km_grade_mean']

            data['view - nb_3km_view_mean'] = data['view'] - data['nb_3km_view_mean']

            data['condition - nb_3km_condition_mean'] = data['condition'] - data['nb_3km_condition_mean']

    if nb_5km:

        neighbor_5km_stat = pd.read_csv('../input/2019-2nd-ml-month-with-kakr-neighbor-stat/neighbor_5km_stat.csv')

        data = data.merge(neighbor_5km_stat, on='id', how='left').fillna(0)

        

        if original:

            data['sqft_living - nb_5km_sqft_living_mean'] = data['sqft_living'] - data['nb_5km_sqft_living_mean']

            data['sqft_lot - nb_5km_sqft_lot_mean'] = data['sqft_lot'] - data['nb_5km_sqft_lot_mean']

            data['bedrooms - nb_5km_bedrooms_mean'] = data['bedrooms'] - data['nb_5km_bedrooms_mean']

            data['bathrooms - nb_5km_bathrooms_mean'] = data['bathrooms'] - data['nb_5km_bathrooms_mean']

            data['grade - nb_5km_grade_mean'] = data['grade'] - data['nb_5km_grade_mean']

            data['view - nb_5km_view_mean'] = data['view'] - data['nb_5km_view_mean']

            data['condition - nb_5km_condition_mean'] = data['condition'] - data['nb_5km_condition_mean']

    if n_5_nb:

        nearest_5_neighbor_stat = pd.read_csv('../input/2019-2nd-ml-month-with-kakr-neighbor-stat/nearest_5_neighbor_stat.csv')

        data = data.merge(nearest_5_neighbor_stat, on='id', how='left').fillna(0)

        

        if original:

            data['sqft_living - n_5_nb_sqft_living_mean'] = data['sqft_living'] - data['n_5_nb_sqft_living_mean']

            data['sqft_lot - n_5_nb_sqft_lot_mean'] = data['sqft_lot'] - data['n_5_nb_sqft_lot_mean']

            data['bedrooms - n_5_nb_bedrooms_mean'] = data['bedrooms'] - data['n_5_nb_bedrooms_mean']

            data['bathrooms - n_5_nb_bathrooms_mean'] = data['bathrooms'] - data['n_5_nb_bathrooms_mean']

            data['grade - n_5_nb_grade_mean'] = data['grade'] - data['n_5_nb_grade_mean']

            data['view - n_5_nb_view_mean'] = data['view'] - data['n_5_nb_view_mean']

            data['condition - n_5_nb_condition_mean'] = data['condition'] - data['n_5_nb_condition_mean']

    if n_10_nb:

        nearest_10_neighbor_stat = pd.read_csv('../input/2019-2nd-ml-month-with-kakr-neighbor-stat/nearest_10_neighbor_stat.csv')

        data = data.merge(nearest_10_neighbor_stat, on='id', how='left').fillna(0)

        

        if original:

            data['sqft_living - n_10_nb_sqft_living_mean'] = data['sqft_living'] - data['n_10_nb_sqft_living_mean']

            data['sqft_lot - n_10_nb_sqft_lot_mean'] = data['sqft_lot'] - data['n_10_nb_sqft_lot_mean']

            data['bedrooms - n_10_nb_bedrooms_mean'] = data['bedrooms'] - data['n_10_nb_bedrooms_mean']

            data['bathrooms - n_10_nb_bathrooms_mean'] = data['bathrooms'] - data['n_10_nb_bathrooms_mean']

            data['grade - n_10_nb_grade_mean'] = data['grade'] - data['n_10_nb_grade_mean']

            data['view - n_10_nb_view_mean'] = data['view'] - data['n_10_nb_view_mean']

            data['condition - n_10_nb_condition_mean'] = data['condition'] - data['n_10_nb_condition_mean']

    if n_20_nb:

        nearest_20_neighbor_stat = pd.read_csv('../input/2019-2nd-ml-month-with-kakr-neighbor-stat/nearest_20_neighbor_stat.csv')

        data = data.merge(nearest_20_neighbor_stat, on='id', how='left').fillna(0)

        

        if original:

            data['sqft_living - n_20_nb_sqft_living_mean'] = data['sqft_living'] - data['n_20_nb_sqft_living_mean']

            data['sqft_lot - n_20_nb_sqft_lot_mean'] = data['sqft_lot'] - data['n_20_nb_sqft_lot_mean']

            data['bedrooms - n_20_nb_bedrooms_mean'] = data['bedrooms'] - data['n_20_nb_bedrooms_mean']

            data['bathrooms - n_20_nb_bathrooms_mean'] = data['bathrooms'] - data['n_20_nb_bathrooms_mean']

            data['grade - n_20_nb_grade_mean'] = data['grade'] - data['n_20_nb_grade_mean']

            data['view - n_20_nb_view_mean'] = data['view'] - data['n_20_nb_view_mean']

            data['condition - n_20_nb_condition_mean'] = data['condition'] - data['n_20_nb_condition_mean']

                

    if fix_skew:

        ordinal_cols = [

            'id','price','data',

            'grade','overall','view','condition','waterfront','is_renovated','has_basement',

        ]

        if do_ohe:

            exclude_cols = [col for col in data.columns if 'ohe_' in col] + ordinal_cols

        else:

            exclude_cols = cat_cols + ordinal_cols

        numeric_feats = [col for col in data.columns if col not in exclude_cols]

        skewed_feats = data[numeric_feats].apply(lambda x : skew(x.dropna())).sort_values(ascending=False)

        skewness = pd.DataFrame({'skew' :skewed_feats})

        skewness = skewness[abs(skewness) > 0.75]

        skewed_features = skewness.index

        lam = 0.15

        for feat in skewed_features:

            data[feat] = boxcox1p(data[feat], lam)

            data[feat] = data[feat].fillna(0)

    

    df = data.drop(['id','price','data'], axis=1).copy()



    train_len = data[data['data'] == 'train'].shape[0]

    X_train = df.iloc[:train_len]

    X_test = df.iloc[train_len:]

    y_train = data[data['data'] == 'train']['price']

    

    if do_scale:

        if do_ohe:

            non_numeric_cols = [col for col in X_train.columns if 'ohe_' in col]

            numeric_cols = [col for col in X_train.columns if 'ohe_' not in col]

        else:

            non_numeric_cols = cat_cols

            numeric_cols = [col for col in X_train.columns if col not in cat_cols]

        X_train_rb = X_train[numeric_cols].copy()

        X_test_rb = X_test[numeric_cols].copy()



        rb = RobustScaler()

        X_train_rb = rb.fit_transform(X_train_rb)

        X_test_rb = rb.transform(X_test_rb)



        X_train_rb = pd.DataFrame(X_train_rb, index=X_train.index, columns=X_train[numeric_cols].columns)

        X_test_rb = pd.DataFrame(X_test_rb, index=X_test.index, columns=X_test[numeric_cols].columns)

        

        X_train_rb = pd.concat([X_train[non_numeric_cols], X_train_rb], axis=1)

        X_test_rb = pd.concat([X_test[non_numeric_cols], X_test_rb], axis=1)

        

        print(X_train_rb.shape, X_test_rb.shape, y_train.shape)

        

        return X_train_rb, X_test_rb, y_train

    else :

        print(X_train.shape, X_test.shape, y_train.shape)

        return X_train, X_test, y_train



def haversine_array(lat1, lng1, lat2, lng2): 

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2)) 

    AVG_EARTH_RADIUS = 6371 # in km 

    lat = lat2 - lat1 

    lng = lng2 - lng1 

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2 

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d)) 

    return h



def bearing_array(lat1, lng1, lat2, lng2): 

    AVG_EARTH_RADIUS = 6371 # in km 

    lng_delta_rad = np.radians(lng2 - lng1) 

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2)) 

    y = np.sin(lng_delta_rad) * np.cos(lat2) 

    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad) 

    return np.degrees(np.arctan2(y, x))

warnings.filterwarnings('ignore')



# Set a few plotting defaults

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (12, 8)

plt.rcParams['font.size'] = 12



pd.options.display.max_rows = 10000

pd.options.display.max_columns = 10000

pd.options.display.max_colwidth = 1000
data_list = []



X_train_1km, X_test_1km, y_train = load_data(nb_1km=True, nb_3km=False, nb_5km=False,

                                             n_5_nb=False, n_10_nb=False, n_20_nb=False,

                                             original=False, do_scale=False, do_ohe=True)

data_list.append((X_train_1km, X_test_1km))



X_train_3km, X_test_3km, y_train = load_data(nb_1km=False, nb_3km=True, nb_5km=False,

                                             n_5_nb=False, n_10_nb=False, n_20_nb=False,

                                             original=False, do_scale=False, do_ohe=True)

data_list.append((X_train_3km, X_test_3km))



X_train_5km, X_test_5km, y_train = load_data(nb_1km=False, nb_3km=False, nb_5km=True,

                                             n_5_nb=False, n_10_nb=False, n_20_nb=False,

                                             original=False, do_scale=False, do_ohe=True)

data_list.append((X_train_5km, X_test_5km))



X_train_5_nn, X_test_5_nn, y_train = load_data(nb_1km=False, nb_3km=False, nb_5km=False,

                                               n_5_nb=True, n_10_nb=False, n_20_nb=False,

                                               original=False, do_scale=False, do_ohe=True)

data_list.append((X_train_5_nn, X_test_5_nn))



X_train_10_nn, X_test_10_nn, y_train = load_data(nb_1km=False, nb_3km=False, nb_5km=False,

                                                 n_5_nb=False, n_10_nb=True, n_20_nb=False,

                                                 original=False, do_scale=False, do_ohe=True)

data_list.append((X_train_10_nn, X_test_10_nn))



X_train_20_nn, X_test_20_nn, y_train = load_data(nb_1km=False, nb_3km=False, nb_5km=False,

                                                 n_5_nb=False, n_10_nb=False, n_20_nb=True,

                                                 original=False, do_scale=False, do_ohe=True)

data_list.append((X_train_20_nn, X_test_20_nn))



X_train_ori, X_test_ori, y_train = load_data(nb_1km=False, nb_3km=False, nb_5km=False,

                                             n_5_nb=False, n_10_nb=False, n_20_nb=False,

                                             original=True, do_scale=False, do_ohe=True)

data_list.append((X_train_ori, X_test_ori))



X_train_full, X_test_full, y_train = load_data(nb_1km=True, nb_3km=True, nb_5km=True,

                                               n_5_nb=True, n_10_nb=True, n_20_nb=True,

                                               original=True, do_scale=False, do_ohe=True)

data_list.append((X_train_full, X_test_full))
run_flag = False



if run_flag:

    model_list = []



    lgb_param = {

        'objective': 'regression',

        'learning_rate': 0.01,

        'max_depth': 20,

        'num_leaves': 63,

        'min_data_in_leaf': 30,

        'bagging_fraction': 0.7,

        'bagging_freq': 1,

        'feature_fraction': 0.2,

        'seed': RANDOM_SEED,

        'metric': ['rmse'],

    }

    lgb_model = LgbmWrapper(params=lgb_param, num_rounds=100000, ealry_stopping=200, verbose_eval=False)

    model_list.append(lgb_model)



    lgb_param2 = {

        'objective': 'regression',

        'learning_rate': 0.01,

        'max_depth': 10,

        'num_leaves': 31,

        'min_data_in_leaf': 30,

        'bagging_fraction': 0.7,

        'bagging_freq': 1,

        'feature_fraction': 0.2,

        'seed': RANDOM_SEED,

        'metric': ['rmse'],

    }

    lgb_model2 = LgbmWrapper(params=lgb_param2, num_rounds=100000, ealry_stopping=200, verbose_eval=False)

    model_list.append(lgb_model2)



    lgb_param3 = {

        'objective': 'regression',

        'learning_rate': 0.01,

        'max_depth': 3,

        'num_leaves': 7,

        'min_data_in_leaf': 30,

        'bagging_fraction': 0.7,

        'bagging_freq': 1,

        'feature_fraction': 0.2,

        'seed': RANDOM_SEED,

        'metric': ['rmse'],

    }

    lgb_model3 = LgbmWrapper(params=lgb_param3, num_rounds=100000, ealry_stopping=200, verbose_eval=False)

    model_list.append(lgb_model3)



    xgb_param = {

        'eval_metric': 'rmse',

        'seed': RANDOM_SEED,

        'eta': 0.01,

        'max_depth': 20,

        'subsample': 0.7,

        'colsample_bytree': 0.5,

        'silent': 1,

    }

    xgb_model = XgbWrapper(params=xgb_param, num_rounds=100000, ealry_stopping=200, verbose_eval=False)

    model_list.append(xgb_model)



    xgb_param2 = {

        'eval_metric': 'rmse',

        'seed': RANDOM_SEED,

        'eta': 0.01,

        'max_depth': 10,

        'subsample': 0.7,

        'colsample_bytree': 0.5,

        'silent': 1,

    }

    xgb_model2 = XgbWrapper(params=xgb_param2, num_rounds=100000, ealry_stopping=200, verbose_eval=False)

    model_list.append(xgb_model2)



    xgb_param3 = {

        'eval_metric': 'rmse',

        'seed': RANDOM_SEED,

        'eta': 0.01,

        'max_depth': 3,

        'subsample': 0.7,

        'colsample_bytree': 0.5,

        'silent': 1,

    }

    xgb_model3 = XgbWrapper(params=xgb_param3, num_rounds=100000, ealry_stopping=200, verbose_eval=False)

    model_list.append(xgb_model3)



    rf_param = {

        'n_estimators': 1000,

        'max_depth': 20,

        'max_features': 0.6,

        'n_jobs': -1,

        'random_state': RANDOM_SEED

    }

    rf_model = SklearnWrapper(RandomForestRegressor, params=rf_param)

    model_list.append(rf_model)



    rf_param2 = {

        'n_estimators': 1000,

        'max_depth': 3,

        'max_features': 0.6,

        'n_jobs': -1,

        'random_state': RANDOM_SEED

    }

    rf_model2 = SklearnWrapper(RandomForestRegressor, params=rf_param2)

    model_list.append(rf_model2)



    ridge_param = {'alpha': 1e-3, 'normalize': True, 'max_iter': 1e7, 'random_state': RANDOM_SEED}

    ridge_model = SklearnWrapper(Ridge, params=ridge_param)

    model_list.append(ridge_model)



    X_train_single, X_test_single, cv_score_single = stacking(data_list, y_train, model_list, eval_func=rmse_exp)
X_train_single = pd.read_csv('../input/11th-solution-data-public-98316-private-99336/x_train_single.csv')

X_test_single = pd.read_csv('../input/11th-solution-data-public-98316-private-99336/x_test_single.csv')

cv_score_single_df = pd.read_csv('../input/11th-solution-data-public-98316-private-99336/cv_score_single.csv')
cv_score_single_df
cv_score_single_df.set_index('name').plot.bar(figsize=(16,8))

plt.ylabel('Single Model CV Score');
X_train_single.shape, X_test_single.shape
X_train_stage1 = pd.concat([

    X_train_single,

    X_train_full[['nb_1km_price_mean','nb_3km_price_mean','nb_5km_price_mean',

                  'n_5_nb_price_mean','n_10_nb_price_mean','n_20_nb_price_mean']]

], axis=1)



X_test_stage1 = pd.concat([

    X_test_single,

    X_test_full[['nb_1km_price_mean','nb_3km_price_mean','nb_5km_price_mean',

                 'n_5_nb_price_mean','n_10_nb_price_mean','n_20_nb_price_mean']].reset_index(drop=True)

], axis=1)



X_train_stage1.shape, X_test_stage1.shape
def create_model(input_dim):

    model = Sequential()

    model.add(Dense(64, activation='selu', input_dim=input_dim,

                    kernel_initializer=initializers.he_normal(seed=RANDOM_SEED),

                    bias_initializer=initializers.Constant(0.01)))

    model.add(Dense(32, activation='selu', 

                    kernel_initializer=initializers.he_normal(seed=RANDOM_SEED),

                    bias_initializer=initializers.Constant(0.01)))

    model.add(Dense(16, activation='selu',

                    kernel_initializer=initializers.he_normal(seed=RANDOM_SEED),

                    bias_initializer=initializers.Constant(0.01)))

    model.add(Dense(8, activation='selu',

                    kernel_initializer=initializers.he_normal(seed=RANDOM_SEED),

                    bias_initializer=initializers.Constant(0.01)))

    model.add(Dense(1,

                    kernel_initializer=initializers.he_normal(seed=RANDOM_SEED),

                    bias_initializer=initializers.Constant(0.01)))





    optimizer = optimizers.RMSprop(lr=0.001)

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])

    

    return model



patient = 200



callbacks = [

    EarlyStopping(monitor='val_loss', patience=patient, mode='min', verbose=1),

    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patient/2, min_lr=0.00001, verbose=1, mode='min')

]
if run_flag:

    model_list_second = []



    keras_model = KerasWrapper(create_model, epochs=100000, batch_size=512, callbacks=callbacks, use_avg_oof=True)

    model_list_second.append(keras_model)



    et_param = {

        'n_estimators': 1000,

        'max_depth': 20,

        'max_features': 0.6,

        'n_jobs': -1,

        'random_state': RANDOM_SEED

    }

    et_model = SklearnWrapper(ExtraTreesRegressor, params=et_param)

    model_list_second.append(et_model)



    et_param2 = {

        'n_estimators': 1000,

        'max_depth': 3,

        'max_features': 0.6,

        'n_jobs': -1,

        'random_state': RANDOM_SEED

    }

    et_model2 = SklearnWrapper(ExtraTreesRegressor, params=et_param2)

    model_list_second.append(et_model2)



    lgb_param = {

        'objective': 'regression',

        'learning_rate': 0.01,

        'max_depth': 20,

        'num_leaves': 63,

        'min_data_in_leaf': 30,

        'bagging_fraction': 0.7,

        'bagging_freq': 1,

        'feature_fraction': 0.2,

        'seed': RANDOM_SEED,

        'metric': ['rmse'],

    }

    lgb_model = LgbmWrapper(params=lgb_param, num_rounds=100000, ealry_stopping=200, verbose_eval=False)

    model_list_second.append(lgb_model)



    lgb_param2 = {

        'objective': 'regression',

        'learning_rate': 0.01,

        'max_depth': 10,

        'num_leaves': 31,

        'min_data_in_leaf': 30,

        'bagging_fraction': 0.7,

        'bagging_freq': 1,

        'feature_fraction': 0.2,

        'seed': RANDOM_SEED,

        'metric': ['rmse'],

    }

    lgb_model2 = LgbmWrapper(params=lgb_param2, num_rounds=100000, ealry_stopping=200, verbose_eval=False)

    model_list_second.append(lgb_model2)



    lgb_param3 = {

        'objective': 'regression',

        'learning_rate': 0.01,

        'max_depth': 3,

        'num_leaves': 7,

        'min_data_in_leaf': 30,

        'bagging_fraction': 0.7,

        'bagging_freq': 1,

        'feature_fraction': 0.2,

        'seed': RANDOM_SEED,

        'metric': ['rmse'],

    }

    lgb_model3 = LgbmWrapper(params=lgb_param3, num_rounds=100000, ealry_stopping=200, verbose_eval=False)

    model_list_second.append(lgb_model3)



    xgb_param = {

        'eval_metric': 'rmse',

        'seed': RANDOM_SEED,

        'eta': 0.01,

        'max_depth': 20,

        'subsample': 0.7,

        'colsample_bytree': 0.5,

        'silent': 1,

    }

    xgb_model = XgbWrapper(params=xgb_param, num_rounds=100000, ealry_stopping=200, verbose_eval=False)

    model_list_second.append(xgb_model)



    xgb_param2 = {

        'eval_metric': 'rmse',

        'seed': RANDOM_SEED,

        'eta': 0.01,

        'max_depth': 10,

        'subsample': 0.7,

        'colsample_bytree': 0.5,

        'silent': 1,

    }

    xgb_model2 = XgbWrapper(params=xgb_param2, num_rounds=100000, ealry_stopping=200, verbose_eval=False)

    model_list_second.append(xgb_model2)



    xgb_param3 = {

        'eval_metric': 'rmse',

        'seed': RANDOM_SEED,

        'eta': 0.01,

        'max_depth': 3,

        'subsample': 0.7,

        'colsample_bytree': 0.5,

        'silent': 1,

    }

    xgb_model3 = XgbWrapper(params=xgb_param3, num_rounds=100000, ealry_stopping=200, verbose_eval=False)

    model_list_second.append(xgb_model3)



    rf_param = {

        'n_estimators': 1000,

        'max_depth': 20,

        'max_features': 0.6,

        'n_jobs': -1,

        'random_state': RANDOM_SEED

    }

    rf_model = SklearnWrapper(RandomForestRegressor, params=rf_param)

    model_list_second.append(rf_model)



    rf_param2 = {

        'n_estimators': 1000,

        'max_depth': 3,

        'max_features': 0.6,

        'n_jobs': -1,

        'random_state': RANDOM_SEED

    }

    rf_model2 = SklearnWrapper(RandomForestRegressor, params=rf_param2)

    model_list_second.append(rf_model2)



    ridge_param = {'alpha': 1e-10, 'normalize': True, 'max_iter': 1e7, 'random_state': RANDOM_SEED}

    ridge_model = SklearnWrapper(Ridge, params=ridge_param)

    model_list_second.append(ridge_model)



    gbr_param = {

        'n_estimators': 1000,

        'learning_rate':0.1,

        'max_depth': 20,

        'subsample': 0.7,

        'max_features': 0.6,

        'random_state': RANDOM_SEED

    }

    gbr_model = SklearnWrapper(GradientBoostingRegressor, params=gbr_param)

    model_list_second.append(gbr_model)



    gbr_param2 = {

        'n_estimators': 1000,

        'learning_rate':0.1,

        'max_depth': 10,

        'subsample': 0.7,

        'max_features': 0.6,

        'random_state': RANDOM_SEED

    }

    gbr_model2 = SklearnWrapper(GradientBoostingRegressor, params=gbr_param2)

    model_list_second.append(gbr_model2)



    gbr_param3 = {

        'n_estimators': 1000,

        'learning_rate':0.1,

        'max_depth': 2,

        'subsample': 0.7,

        'max_features': 0.6,

        'random_state': RANDOM_SEED

    }

    gbr_model3 = SklearnWrapper(GradientBoostingRegressor, params=gbr_param3)

    model_list_second.append(gbr_model3)



    lasso_param = {'alpha':1e-6, 'normalize':True, 'max_iter':1e7, 'random_state':RANDOM_SEED}

    lasso_model = SklearnWrapper(Lasso, params=lasso_param)

    model_list_second.append(lasso_model)



    elastic_param = {'alpha':1e-6, 'normalize':True, 'max_iter':1e5, 'random_state':RANDOM_SEED, 'l1_ratio':0.8}

    elastic_model = SklearnWrapper(ElasticNet, params=elastic_param)

    model_list_second.append(elastic_model)



    svr_param = {'C':1e3, 'epsilon':0.001, 'gamma':1e-4}

    svr_model = SklearnWrapper(SVR, params=svr_param)

    model_list_second.append(svr_model)
if run_flag:

    data_list_stage1 = [(X_train_stage1, X_test_stage1)]

    X_train_stage2, X_test_stage2, cv_score_stage1 = stacking(data_list_stage1, y_train, model_list_second,

                                                              eval_func=rmse_exp)
X_train_stage2 = pd.read_csv('../input/11th-solution-data-public-98316-private-99336/x_train_stage2.csv')

X_test_stage2 = pd.read_csv('../input/11th-solution-data-public-98316-private-99336/x_test_stage2.csv')

cv_score_stage1_df = pd.read_csv('../input/11th-solution-data-public-98316-private-99336/cv_score_stage1.csv')
cv_score_stage1_df
cv_score_stage1_df.set_index('name').plot.bar(figsize=(16,8))

plt.ylabel('Stage 1 CV Score');
ridge_l3_param = {

    'alpha': 1e-10,

    'normalize': True,

    'max_iter': 1e7,

    'random_state': RANDOM_SEED

}



ridge_l3_model = SklearnWrapper(Ridge, params=ridge_l3_param, use_avg_oof=True)



ridge_l3_train, ridge_l3_test, ridge_l3_cv_score = get_oof(ridge_l3_model, X_train_stage2, y_train, X_test_stage2,

                                                           rmse_exp, NFOLDS=5, kfold_random_state=RANDOM_SEED)
lasso_l3_param = {'alpha':1e-7, 'normalize':True, 'max_iter':1e7, 'random_state':RANDOM_SEED}



lasso_l3_model = SklearnWrapper(Lasso, params=lasso_l3_param, use_avg_oof=True)

lasso_l3_train, lasso_l3_test, lasso_l3_cv_score = get_oof(lasso_l3_model, X_train_stage2, y_train, X_test_stage2,

                                                           rmse_exp, NFOLDS=5, kfold_random_state=RANDOM_SEED)
elastic_l3_param = {'alpha':1e-8, 'normalize':True, 'max_iter':1e6, 'random_state':RANDOM_SEED, 'l1_ratio':0.8}



elastic_l3_model = SklearnWrapper(ElasticNet, params=elastic_l3_param, use_avg_oof=True)

elastic_l3_train, elastic_l3_test, elastic_l3_cv_score = get_oof(elastic_l3_model, X_train_stage2, y_train, X_test_stage2,

                                                                 rmse_exp, NFOLDS=5, kfold_random_state=RANDOM_SEED)
svr_l3_param = {'C':1e3, 'epsilon':0.001, 'gamma':1e-4}



svr_l3_model = SklearnWrapper(SVR, params=svr_l3_param, use_avg_oof=True)

svr_l3_train, svr_l3_test, svr_l3_cv_score = get_oof(svr_l3_model, X_train_stage2, y_train, X_test_stage2,

                                                     rmse_exp, NFOLDS=5, kfold_random_state=RANDOM_SEED)
cv_score_stage2_df = pd.read_csv('../input/11th-solution-data-public-98316-private-99336/cv_score_stage2.csv')

cv_score_stage2_df
cv_score_stage2_df.set_index('name').plot.bar(figsize=(16,8))

plt.ylabel('Stage 2 CV Score');
# public score: 98316.65734, private score: 99336.33652



avg_pred = elastic_l3_test



test = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/test.csv')



output = f'stacking_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'

print(output)



submission = pd.DataFrame({'id': test['id'], 'price': np.expm1(avg_pred.ravel())})

submission.to_csv(output, index=False)