import os

import sys

import gc

import time

import json

import re

import random

from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL

from logging import getLogger, StreamHandler, FileHandler, Formatter

from contextlib import contextmanager

from collections import Counter, defaultdict, OrderedDict

from functools import partial

from joblib import Parallel, delayed



from tqdm import tqdm



import numpy as np

import pandas as pd

from scipy.stats import kurtosis, skew



import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.utils.multiclass import unique_labels



from sklearn.utils import check_random_state

from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from gensim.models import Word2Vec

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer, SimpleImputer



from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, train_test_split

from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, log_loss

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import confusion_matrix, roc_curve

from scipy.optimize import minimize

 

import lightgbm as lgb

import xgboost as xgb

import catboost as cb



import tensorflow as tf

import tensorflow.keras as keras

import torch





import warnings

warnings.filterwarnings("ignore")



%matplotlib inline

plt.style.use('ggplot')

pd.set_option('max_rows', 1000)

pd.set_option('max_columns', 1000)
def seed_everything(seed=42):

    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    

    

def get_logger(

    filename='log',

    disable_stream_handler=False,

    disable_file_handler=False,

    level=20, # INFO

    log_format="%(message)s",

):

    logger = getLogger(__name__)

    logger.setLevel(level)

    

    if not disable_stream_handler:

        handler1 = StreamHandler()

        handler1.setFormatter(Formatter(log_format))

        logger.addHandler(handler1)

    

    if not disable_file_handler:

        handler2 = FileHandler(filename=f"{filename}.log")

        handler2.setFormatter(Formatter(log_format))

        logger.addHandler(handler2)

    

    return logger





def printl(msg, level=20):

    # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # (10, 20, 30, 40, 50)

    try:

        if level == 10:

            logger.debug(msg)

        elif level == 20:

            logger.info(msg)

        elif level == 30:

            logger.warning(msg)

        elif level == 40:

            logger.error(msg)

        elif level == 50:

            logger.critical(msg)

        else:

            raise ValueError

    except NameError:

        print(msg)

        

        

@contextmanager

def timer(name):

    t0 = time.time()

    yield

    printl(f"[{name}] done in {time.time() - t0:.0f} s")

    

    

def load_df(path):

    basename = os.path.basename(path)

    ext = path.split('.')[-1]

    

    if ext == 'csv':

        df = pd.read_csv(path)

    elif ext == 'pkl':

        df = pd.read_pickle(path)

    else:

        raise IOError(f'Not Accessable Format .{ext}')



    printl(f"{basename} shape / {df.shape}")

    

    return df
def oof_target_encoding(

    train,

    target,

    test,

    cv,

    cols,

    handle_missing='value',

    handle_unknown='value',

    min_samples_leaf=1,

    smoothing=1.0,

    agg='mean'

):

    encoded_cols = []

    for c in cols:

        ecol = f'{agg}_TE_{c}'

        train[ecol] = np.nan

        encoded_cols.append(ecol)

    

    # TE for oof

    for fold, (dev_idx, val_idx) in enumerate(cv):

        te = TargetEncoder(cols=cols,

                           handle_missing=handle_missing,

                           handle_unknown=handle_unknown,

                           min_samples_leaf=min_samples_leaf,

                           smoothing=smoothing)

        te.fit(train.loc[dev_idx, cols], target[dev_idx], agg=agg)

        train.loc[val_idx, encoded_cols] = te.transform(train.loc[val_idx, cols]).values

    

    # TE for test

    te = TargetEncoder(cols=cols,

                       handle_missing=handle_missing,

                       handle_unknown=handle_unknown,

                       min_samples_leaf=min_samples_leaf,

                       smoothing=smoothing)

    te.fit(train[cols], target)

    test[encoded_cols] = te.transform(test[cols])

    

    return train, test





class NNPreprocessor:

    def __init__(

        self,

        categorical_columns='auto',

        minmax_columns='auto',

        gaussian_columns='auto',

        ignore_columns=[],

        n_quantiles=1000,

        max_iter=10,

        random_state=42,

    ):

        self.categorical_columns = categorical_columns

        self.minmax_columns = minmax_columns

        self.gaussian_columns = gaussian_columns

        self.ignore_columns = ignore_columns



        self.dummied_columns = []

        self.has_nan_columns = []

        self.nan_flag_columns = []

        

        self.oe = OneHotEncoder(cols=categorical_columns,

                                handle_missing='indicator',

                                handle_unknown='error',

                                use_cat_names=True)

        self.mme = MinMaxScaler()

        self.qt = QuantileTransformer(n_quantiles=n_quantiles,

                                      output_distribution='normal',

                                      random_state=random_state)

        self.imp = IterativeImputer(sample_posterior=True,

                                    max_iter=max_iter,

                                    random_state=random_state)

    

    def _set_columns(self, X):

        if self.categorical_columns == 'auto':

            columns = [c for c in X.columns if X.dtypes[c] == 'object']

            columns = [c for c in columns if c not in self.ignore_columns]

            self.categorical_columns = columns

        

        numerical_columns = [c for c in X.columns if c not in self.categorical_columns + self.ignore_columns]

        

        if self.minmax_columns == 'auto':

            nunique = X[numerical_columns].nunique()

            columns = nunique[nunique == 2].index.to_list()

            self.minmax_columns = columns

        

        if self.gaussian_columns == 'auto':

            columns = [c for c in numerical_columns if c not in self.minmax_columns]

            self.gaussian_columns = columns

        

        nan_count = X[numerical_columns].isnull().sum()

        columns = nan_count[nan_count > 0].index.to_list()

        self.has_nan_columns = columns

        self.nan_flag_columns = [f'{c}_nanflag' for c in self.has_nan_columns]

        

    def get_feature_columns(self):

        columns = []

        columns += self.dummied_columns

        columns += self.minmax_columns

        columns += self.gaussian_columns

        columns += self.nan_flag_columns

        return columns



    def fit(self, X):

        self._set_columns(X)

        

        # categorical encoding

        if len(self.categorical_columns) > 0:

            self.oe.fit(X[self.categorical_columns])

            self.dummied_columns += self.oe.get_feature_names()

        

        # minmax encoding

        if len(self.minmax_columns) > 0:

            self.mme.fit(X[self.minmax_columns])

        

        # (rank-)gaussian encoding

        if len(self.gaussian_columns) > 0:

            self.qt.fit(X[self.gaussian_columns])

            

        # multiple imputation

        if len(self.has_nan_columns) > 0:

            columns = self.minmax_columns + self.gaussian_columns

            self.imp.fit(X[columns])

    

    def transform(self, X):

        # categorical encoding

        if len(self.categorical_columns) > 0:

            X = pd.concat([X, self.oe.transform(X[self.categorical_columns])], axis=1)

            X = X.drop(self.categorical_columns, axis=1)

        

        # minmax encoding

        if len(self.minmax_columns) > 0:

            X[self.minmax_columns] = self.mme.transform(X[self.minmax_columns])

        

        # (rank-)gaussian encoding

        if len(self.gaussian_columns) > 0:

            X[self.gaussian_columns] = self.qt.transform(X[self.gaussian_columns])

        

        # make nan flag dataframe

        if len(self.has_nan_columns) > 0:

            X_nan = X[self.has_nan_columns].copy()

            X_nan = X_nan.isnull().astype(np.float32)

            X_nan.columns = self.nan_flag_columns

        

            # multiple imputation

            columns = self.minmax_columns + self.gaussian_columns

            X[columns] = self.imp.transform(X[columns])

            

            # add nan dataframe

            X = pd.concat([X, X_nan], axis=1)

        

        return X
class MyGroupKFold:

    def __init__(self, n_splits=5, shuffle=True, random_state=None):

        self.n_splits = n_splits

        self.shuffle = shuffle

        self.random_state = random_state

    

    def split(self, X, y=None, groups=None):

        groups = pd.Series(groups)

        unique_groups = np.unique(groups)

        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

        for tr_group_idx, va_group_idx in kf.split(unique_groups):

            tr_groups, va_groups = unique_groups[tr_group_idx], unique_groups[va_group_idx]

            tr_indices = groups[groups.isin(tr_groups)].index.to_list()

            va_indices = groups[groups.isin(va_groups)].index.to_list()

            yield tr_indices, va_indices

            

            

class StratifiedGroupKFold:

    def __init__(self, n_splits=5, shuffle=True, random_state=None):

        self.n_splits = n_splits

        self.shuffle = shuffle

        self.random_state = random_state

        

    # Implementation based on this kaggle kernel:

    #    https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation

    def split(self, X, y=None, groups=None):

        k = self.n_splits

        rnd = check_random_state(self.random_state)

            

        # labels_num: zero-origin number of label

        # ex) unique = [0,1,2,3] -> labels_num = 4

        labels_num = np.max(y) + 1

        

        # y_counts_per_group: in-group label distribution

        # y_distr: whole label distribution

        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))

        y_distr = Counter()

        for label, g in zip(y, groups):

            y_counts_per_group[g][label] += 1

            y_distr[label] += 1



        # y_counts_per_fold: in-fold label distribution

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))

        groups_per_fold = defaultdict(set)

        

        # return mean std of per label counts when y_counts is in fold

        def eval_y_counts_per_fold(y_counts, fold):

            y_counts_per_fold[fold] += y_counts

            std_per_label = []

            for label in range(labels_num):

                label_std = np.std(

                    [y_counts_per_fold[i][label] / y_distr[label] for i in range(k)]

                )

                std_per_label.append(label_std)

            y_counts_per_fold[fold] -= y_counts

            return np.mean(std_per_label)

        

        # list of [group, y_counts]

        # if shuffle: fold changes in same np.std(y_counts)

        # ascending groups by degree of label variance

        groups_and_y_counts = list(y_counts_per_group.items())

        if self.shuffle:

            rnd.shuffle(groups_and_y_counts)

        groups_and_y_counts = sorted(groups_and_y_counts, key=lambda x: -np.std(x[1]))



        # set fold for each group such that label distirbution will be uniform

        for g, y_counts in groups_and_y_counts:

            best_fold = None

            min_eval = None

            for i in range(k):

                fold_eval = eval_y_counts_per_fold(y_counts, i)

                if min_eval is None or fold_eval < min_eval:

                    min_eval = fold_eval

                    best_fold = i

            y_counts_per_fold[best_fold] += y_counts

            groups_per_fold[best_fold].add(g)



        all_groups = set(groups)

        for i in range(k):

            train_groups = all_groups - groups_per_fold[i]

            test_groups = groups_per_fold[i]



            train_indices = [i for i, g in enumerate(groups) if g in train_groups]

            test_indices = [i for i, g in enumerate(groups) if g in test_groups]



            yield train_indices, test_indices

            

            

def build_cv_spliter(

    X,

    y,

    group=None,

    strategy='stratified',

    n_splits=5,

    shuffle=True,

    random_seed=42,

    return_indices=False,

):

    if strategy == 'kfold':

        kf = KFold(n_splits=n_splits, random_state=random_seed, shuffle=shuffle)

        cv = kf.split(X)

    elif strategy == 'stratified':

        kf = StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=shuffle)

        cv = kf.split(X, y)

    elif strategy == 'group':

        kf = MyGroupKFold(n_splits=n_splits, random_state=random_seed, shuffle=shuffle)

        cv = kf.split(X, y, group)

    elif strategy == 'stratified-group':

        kf = StratifiedGroupKFold(n_splits=n_splits, random_state=random_seed, shuffle=shuffle)

        cv = kf.split(X, y, group)

    else:

        raise NotImplementedError(f'strategy {strategy} not implemented.')

    

    if not return_indices:

        cv_spliter = []

        for dev_idx, val_idx in cv:

            cv_spliter.append([dev_idx, val_idx])

        return cv_spliter

    else:

        fold_indices = np.zeros(len(X), dtype=np.int64)

        for fold, (_, val_idx) in enumerate(cv):

            fold_indices[val_idx] = int(fold)

        return fold_indices

    

    

def cvt_cv2indices(cv):

    # get total length

    length = 0

    for _, val_idx in cv:

        length += len(val_idx)

    # cvt cv -> indices

    fold_indices = np.zeros(length, dtype=np.int64)

    for fold, (_, val_idx) in enumerate(cv):

        fold_indices[val_idx] = int(fold)

    return fold_indices





def cvt_indices2cv(indices):

    cv_spliter = []

    for fold in set(indices):

        dev_idx = np.where(indices != fold)[0]

        val_idx = np.where(indices == fold)[0]

        cv_spliter.append([dev_idx, val_idx])

    return cv_spliter
def extract_high_corr_columns(df, threshold=0.99, verbose=1):

    df_corr = abs(df.corr())

    delete_columns = []

    

    # diagonal values filled by zero

    for i in range(0, len(df_corr.columns)):

        df_corr.iloc[i, i] = 0

    

    # loop as removing high-correlated columns in df_corr

    while True:

        df_max_column_value = df_corr.max()

        max_corr = df_max_column_value.max()

        query_column = df_max_column_value.idxmax()

        target_column = df_corr[query_column].idxmax()

        

        if max_corr < threshold:

            break

        else:

            # drop feature which is highly correlated with others 

            if sum(df_corr[query_column]) <= sum(df_corr[target_column]):

                delete_column = target_column

                saved_column = query_column

            else:

                delete_column = query_column

                saved_column = target_column

            

            df_corr.drop([delete_column], axis=0, inplace=True)

            df_corr.drop([delete_column], axis=1, inplace=True)

            delete_columns.append(delete_column)

            

            if verbose > 0:

                printl('{}: Drop: {} <- Query: {}, Corr: {:.5f}'.format(

                    len(delete_columns), delete_column, saved_column, max_corr

                ))



    return delete_columns





def _get_lgb_fimp(

    params,

    X_train,

    y_train,

    features,

    shuffle,

    seed=42,

    categorical=[]

):

    # Shuffle target if required

    y = y_train.copy()

    if shuffle:

        random.seed(seed)

        np.random.seed(seed)

        y = y_train.copy().sample(frac=1.0)

    

    arg_categorical = categorical if len(categorical) > 0 else 'auto'

    dtrain = lgb.Dataset(X_train[features],

                         label=y.values,

                         categorical_feature=arg_categorical)

    

    # Fit the model

    clf = lgb.train(params, dtrain)



    # Get feature importances

    imp_df = pd.DataFrame()

    imp_df['feature'] = features

    imp_df['split'] = clf.feature_importance(importance_type='split')

    imp_df['gain'] = clf.feature_importance(importance_type='gain')

    

    return imp_df





def null_importance_selection(

    params,

    X_train,

    y_train,

    features,

    seed=42,

    categorical=[],

    num_actual_run=1,

    num_null_run=40,

    eps=1e-10,

    valid_percentile=75,

):

    actual_imp_df = pd.DataFrame()

    

    np.random.seed(seed)

    for i in tqdm(range(num_actual_run)):

        seed = np.random.randint(1000)

        imp_df = _get_lgb_fimp(params,

                               X_train,

                               y_train,

                               features,

                               shuffle=False,

                               seed=seed,

                               categorical=categorical)

        imp_df['run'] = i

        actual_imp_df = pd.concat([actual_imp_df, imp_df], axis=0)

    

    null_imp_df = pd.DataFrame()

    

    np.random.seed(seed)

    for i in tqdm(range(num_null_run)):

        seed = np.random.randint(1000)

        imp_df = _get_lgb_fimp(params,

                               X_train,

                               y_train,

                               features,

                               shuffle=True,

                               seed=seed,

                               categorical=categorical)

        imp_df['run'] = i

        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)

    

    feature_scores = []

    

    for _f in actual_imp_df['feature'].unique():

        # importance gain of gain

        act_fimp_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'split'].mean()

        null_fimp_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'split'].values

        split_score = np.log(eps + act_fimp_split / (1 + np.percentile(null_fimp_split, valid_percentile)))

        

        # importance gain of gain

        act_fimp_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'gain'].mean()

        null_fimp_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'gain'].values

        gain_score = np.log(eps + act_fimp_gain / (1 + np.percentile(null_fimp_gain, valid_percentile)))



        feature_scores.append((_f, split_score, gain_score))

    

    scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])

    return scores_df
def plot_feature_importance(

    feature_importance_df,

    feature_name='feature',

    importance_name=['split', 'gain'],

    top_k=50,

    fig_width=16,

    fig_height=8,

    fontsize=14,

):

    if isinstance(importance_name, str):

        importance_name = [importance_name]

    

    num_importance = len(importance_name)

    plt.figure(figsize=(fig_width, fig_height*num_importance))

    gs = gridspec.GridSpec(1, num_importance)

    

    def _fetch_best_features(df, fimp='gain'):

        cols = (df[[feature_name, fimp]]

                .groupby(feature_name)

                .mean()

                .sort_values(by=fimp, ascending=False)

                .index

                .values[:top_k])

        return cols, df.loc[df[feature_name].isin(cols)]

    

    for i, fimp in enumerate(importance_name):

        cols, best_features = _fetch_best_features(feature_importance_df, fimp)

        ax = plt.subplot(gs[0, i])

        sns.barplot(x=fimp, y=feature_name, data=best_features, order=cols, ax=ax)

        title = f'Features {fimp} importance (averaged/folds)'

        plt.title(title, fontweight='bold', fontsize=fontsize)

    

    plt.tight_layout()
def run_kfold_lightgbm(

    params,

    X_train,

    y_train,

    X_test,

    cv,

    features,

    metrics,

    categorical=[],

    verbose_eval=100,

):

    oof = np.zeros(len(X_train))

    predictions = np.zeros(len(X_test))

    feature_importance_df = pd.DataFrame()

    

    n_splits = len(cv)

    printl(f"k={n_splits} folds lightgbm running...")

    printl(f"train data/feature shape: {X_train[features].shape}")

    

    for fold, (dev_idx, val_idx) in enumerate(cv):

        arg_categorical = categorical if len(categorical) > 0 else 'auto'

        dev_data = lgb.Dataset(X_train.loc[dev_idx, features],

                               label=y_train[dev_idx],

                               categorical_feature=arg_categorical)

        val_data = lgb.Dataset(X_train.loc[val_idx, features],

                               label=y_train[val_idx],

                               categorical_feature=arg_categorical)

        

        clf = lgb.train(params, dev_data, valid_sets=[dev_data, val_data], verbose_eval=verbose_eval)

        time.sleep(1)

        

        oof[val_idx] = clf.predict(X_train.loc[val_idx, features], num_iteration=clf.best_iteration)

        predictions += clf.predict(X_test[features], num_iteration=clf.best_iteration) / n_splits

        

        msg = f'fold: {fold}'

        for name, func in metrics.items():

            score = func(y_train[val_idx], oof[val_idx])

            msg += f' - {name}: {score:.5f}'

        printl(msg)



        fold_importance_df = pd.DataFrame()

        fold_importance_df['feature'] = features

        fold_importance_df['split'] = clf.feature_importance(importance_type='split')

        fold_importance_df['gain'] = clf.feature_importance(importance_type='gain')

        fold_importance_df['fold'] = fold

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    msg = f'CV score'

    for name, func in metrics.items():

        score = func(y_train, oof)

        msg += f' - {name}: {score:.5f}'

    printl(msg)



    return oof, predictions, feature_importance_df





def advarsarial_validation_lightgbm(

    params,

    X_train,

    X_test,

    features,

    categorical=[],

    n_splits=5,

    shuffle=True,

    seed=42,

):

    X_train_adv = X_train.copy()

    X_test_adv = X_test.copy()

    

    X_train_adv['test'] = 0

    X_test_adv['test'] = 1

    

    X_train_adv = pd.concat([X_train_adv, X_test_adv], axis=0).reset_index(drop=True)

    y_train_adv = X_train_adv['test']

    X_train_adv = X_train_adv.drop('test', axis=1)

    

    printl(f'{X_train_adv.shape}, {y_train_adv.shape}, {len(features)}')

    

    cv = build_cv_spliter(X_train_adv,

                          y_train_adv,

                          strategy='stratified',

                          n_splits=n_splits,

                          shuffle=shuffle,

                          random_seed=seed)

    

    adv_metrics = {'AUC': roc_auc_score}

    _, adv, feature_importance_df = run_kfold_lightgbm(params,

                                                       X_train_adv,

                                                       y_train_adv,

                                                       X_train,

                                                       cv,

                                                       features,

                                                       adv_metrics,

                                                       categorical=cat_features)

    

    return adv, feature_importance_df





def run_kfold_catboost_clf(

    params,

    X_train,

    y_train,

    X_test,

    cv,

    features,

    metrics,

    categorical=[],

):

    oof = np.zeros(len(X_train))

    predictions = np.zeros(len(X_test))

    feature_importance_df = pd.DataFrame()

    

    n_splits = len(cv)

    printl(f"k={n_splits} folds catboost running...")

    printl(f"train data/feature shape: {X_train[features].shape}")

    

    for fold, (dev_idx, val_idx) in enumerate(cv):

        arg_categorical = categorical if len(categorical) > 0 else 'auto'

        dev_data = cb.Pool(X_train.loc[dev_idx, features],

                           label=y_train[dev_idx],

                           cat_features=arg_categorical)

        val_data = cb.Pool(X_train.loc[val_idx, features],

                           label=y_train[val_idx],

                           cat_features=arg_categorical)

        

        clf = cb.CatBoostClassifier(**params)

        clf.fit(dev_data, eval_set=val_data)

        time.sleep(1)

        

        oof[val_idx] = clf.predict_proba(X_train.loc[val_idx, features])[:,1]

        predictions += clf.predict_proba(X_test[features])[:,1] / n_splits

        

        msg = f'fold: {fold}'

        for name, func in metrics.items():

            score = func(y_train[val_idx], oof[val_idx])

            msg += f' - {name}: {score:.5f}'

        printl(msg)

        

        fold_importance_df = pd.DataFrame()

        fold_importance_df['feature'] = features

        fold_importance_df['gain'] = clf.get_feature_importance()

        fold_importance_df['fold'] = fold

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    msg = f'CV score'

    for name, func in metrics.items():

        score = func(y_train, oof)

        msg += f' - {name}: {score:.5f}'

    printl(msg)



    return oof, predictions, feature_importance_df





def run_kfold_catboost_regr(

    params,

    X_train,

    y_train,

    X_test,

    cv,

    features,

    metrics,

    categorical=[],

):

    oof = np.zeros(len(X_train))

    predictions = np.zeros(len(X_test))

    feature_importance_df = pd.DataFrame()

    

    n_splits = len(cv)

    printl(f"k={n_splits} folds catboost running...")

    printl(f"train data/feature shape: {X_train[features].shape}")

    

    for fold, (dev_idx, val_idx) in enumerate(cv):

        arg_categorical = categorical if len(categorical) > 0 else 'auto'

        dev_data = cb.Pool(X_train.loc[dev_idx, features],

                           label=y_train[dev_idx],

                           cat_features=arg_categorical)

        val_data = cb.Pool(X_train.loc[val_idx, features],

                           label=y_train[val_idx],

                           cat_features=arg_categorical)

        

        clf = cb.CatBoostRegressor(**params)

        clf.fit(dev_data, eval_set=val_data)

        time.sleep(1)

        

        oof[val_idx] = clf.predict(X_train.loc[val_idx, features])

        predictions += clf.predict(X_test[features]) / n_splits

        

        msg = f'fold: {fold}'

        for name, func in metrics.items():

            score = func(y_train[val_idx], oof[val_idx])

            msg += f' - {name}: {score:.5f}'

        printl(msg)

        

        fold_importance_df = pd.DataFrame()

        fold_importance_df['feature'] = features

        fold_importance_df['gain'] = clf.get_feature_importance()

        fold_importance_df['fold'] = fold

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    msg = f'CV score'

    for name, func in metrics.items():

        score = func(y_train, oof)

        msg += f' - {name}: {score:.5f}'

    printl(msg)



    return oof, predictions, feature_importance_df





def build_neuralnet(

    recipe,

    loss='mse',

    optimizer='adam',

    lr=1e-3,

    monitor='val_loss',

    es_patience=-1,

    restore_best_weights=True,

    lr_scheduler='none',

    lr_factor=0.1,

    lr_patience=5,

    seed=42,

    **_,

):

    tf.random.set_seed(seed)

    model = keras.models.model_from_json(recipe)

    

    if loss == 'mse':

        loss = keras.losses.mean_squared_error

    elif loss == 'bce':

        loss = keras.losses.binary_crossentropy

    else:

        raise NotImplementedError

    

    if optimizer == 'adam':

        optimizer = keras.optimizers.Adam(lr)

    else:

        raise NotImplementedError

    

    model.compile(optimizer=optimizer, loss=loss)

    

    callbacks = []

    

    if es_patience >= 0:

        es = keras.callbacks.EarlyStopping(monitor=monitor,

                                           patience=es_patience,

                                           restore_best_weights=restore_best_weights,

                                           verbose=1)

        callbacks.append(es)

    

    if lr_scheduler == 'none':

        pass

    elif lr_scheduler == 'reduce_on_plateau':

        lr_sche = keras.callbacks.ReduceLROnPlateau(monitor=monitor,

                                                    factor=lr_factor,

                                                    patience=lr_patience,

                                                    verbose=1)

        callbacks.append(lr_sche)

    else:

        raise NotImplementedError

    

    return model, callbacks





def train_neuralnet(

    params,

    X_train,

    y_train,

    validation_data=None,

):

    model, callbacks = build_neuralnet(**params)

    model.fit(X_train,

              y_train,

              validation_data=validation_data,

              batch_size=params['batch_size'],

              epochs=params['epochs'],

              callbacks=callbacks)

    return model





def run_kfold_neuralnet(

    params,

    X_train,

    y_train,

    X_test,

    cv,

    features,

    metrics,

):

    oof = np.zeros(len(X_train))

    predictions = np.zeros(len(X_test))

    

    n_splits = len(cv)

    printl(f"k={n_splits} folds neuralnet running...")

    printl(f"train data/feature shape: {X_train[features].shape}")

    

    for fold, (dev_idx, val_idx) in enumerate(cv):

        validation_data = [X_train.loc[val_idx, features], y_train[val_idx]]

        model = train_neuralnet(params,

                                X_train.loc[dev_idx, features],

                                y_train[dev_idx],

                                validation_data=validation_data)

        

        oof[val_idx] = model.predict(X_train.loc[val_idx, features].values)[:,0]

        predictions += model.predict(X_test[features].values)[:,0] / n_splits

        

        msg = f'fold: {fold}'

        for name, func in metrics.items():

            score = func(y_train[val_idx], oof[val_idx])

            msg += f' - {name}: {score:.5f}'

        printl(msg)

    

    msg = f'CV score'

    for name, func in metrics.items():

        score = func(y_train, oof)

        msg += f' - {name}: {score:.5f}'

    printl(msg)



    return oof, predictions
def search_averaging_weight(

    oof_df,

    predictions_df,

    target,

    metric,

    solver='L-BFGS-B',

    trials=1000,

    seed=42,

):

    predictions = []

    

    for c in oof_df.columns:

        predictions.append(np.array(oof_df.loc[:, c]))

        

    best_score = 0

    np.random.seed(seed)

    for i in tqdm(range(trials)):

        dice = np.random.rand(len(predictions))

        weights = dice / dice.sum()

        

        blended = np.zeros(len(predictions[0]))

        for weight, pred in zip(weights, predictions):

            blended += weight * pred



        score = metric(target, blended)

        if score > best_score:

            best_score = score

            best_weights = weights

    

    printl('CV Score: {best_loss:.7f}'.format(best_loss=best_score))

    printl('Best Weights: {weights:}'.format(weights=best_weights))

    

    return best_score, best_weights
logger = get_logger()
INPUT_DIR = '../input/ailab-ml-training-2/'



df_path_dict = {

    'train': os.path.join(INPUT_DIR, 'train.csv'),

    'test': os.path.join(INPUT_DIR, 'test.csv'),

    'sample_submission': os.path.join(INPUT_DIR, 'sample_submission.csv'),

}



metrics = {

    'RMSLE': mean_squared_log_error

}



ID = 'test_id' # train_id --> test_id

TARGET = 'price'

N_SPLITS = 5

SEED = 42



seed_everything(seed=SEED)
with timer('Data Loading'):

    train_df = load_df(df_path_dict['train'])

    test_df = load_df(df_path_dict['test'])

    sample_submission = load_df(df_path_dict['sample_submission'])
train_df = train_df.rename({'train_id': ID}, axis=1)
train_df.head()
test_df.head()
print('shape')

print(train_df.shape, test_df.shape)
print('item_condition_id')



print('train unique:', train_df['item_condition_id'].unique())

print('test unique:', test_df['item_condition_id'].unique())



print('train nan:', train_df['item_condition_id'].isnull().mean())

print('test nan:', test_df['item_condition_id'].isnull().mean())



sns.distplot(train_df['item_condition_id'], kde=False, label='train')

sns.distplot(test_df['item_condition_id'], kde=False, label='test')

plt.legend()

plt.show()
print('brand_name')



print('train unique:', train_df['brand_name'].unique(), train_df['brand_name'].nunique())

print('test unique:', test_df['brand_name'].unique(), test_df['brand_name'].nunique())



print('train nan:', train_df['brand_name'].isnull().mean())

print('test nan:', test_df['brand_name'].isnull().mean())



train_set = set(train_df['brand_name'].unique())

test_set = set(test_df['brand_name'].unique())

print('union: ', len(train_set.union(test_set)))

print('intersection: ', len(train_set.intersection(test_set)))

print('train only: ', len(train_set.difference(test_set)))

print('test only: ', len(test_set.difference(train_set)))



sns.distplot(train_df['brand_name'].value_counts(), kde=False, label='train')

sns.distplot(test_df['brand_name'].value_counts(), kde=False, label='test')

plt.legend()

plt.yscale('log')

plt.show()
print('shipping')



sns.distplot(train_df['shipping'], kde=False, label='train')

sns.distplot(test_df['shipping'], kde=False, label='test')

plt.legend()

plt.show()
# target encoding 'brand_name'



cv = build_cv_spliter(

    X=train_df,

    y=train_df[TARGET],

    group=None,

    strategy='kfold',

    n_splits=N_SPLITS,

    shuffle=True,

    random_seed=SEED,

    return_indices=False,

)



train_df, test_df = oof_target_encoding(

    train=train_df,

    target=train_df[TARGET],

    test=test_df,

    cv=cv,

    cols=['brand_name'],

    handle_missing='return_nan',

    handle_unknown='return_nan',

    min_samples_leaf=1,

    smoothing=1.0,

    agg='mean',

)
train_df.head()
sns.distplot(train_df['mean_TE_brand_name'], kde=False, label='train')

sns.distplot(test_df['mean_TE_brand_name'], kde=False, label='test')

plt.legend()

plt.title('histgram of target encoded brand_name')

plt.show()
train_df.head()
merge_cols = [ID]

base_feature_cols = [

    'item_condition_id',

    'shipping',

    'mean_TE_brand_name',

]



Xy_train = pd.DataFrame(train_df[merge_cols]).reset_index(drop=True)

Xy_train[TARGET] = train_df[TARGET].values

Xy_train = pd.merge(Xy_train, train_df[merge_cols + base_feature_cols], on=merge_cols, how='left')



X_test = pd.DataFrame(test_df[merge_cols]).reset_index(drop=True)

X_test[TARGET] = np.nan

X_test = pd.merge(X_test, test_df[merge_cols + base_feature_cols], on=merge_cols, how='left')
X_train = Xy_train.drop([TARGET], axis=1)

y_train = Xy_train[TARGET]



X_test = X_test[X_test[TARGET].isnull()].reset_index(drop=True)

X_test = X_test.drop(TARGET, axis=1)
X_train.head()
cat_features = [

    'item_condition_id',

    'shipping',

]



oe = OrdinalEncoder(cols=cat_features, handle_unknown='impute')

X_train = oe.fit_transform(X_train)

X_test = oe.transform(X_test)
X_train.head()
num_features = [col for col in X_test.columns if X_test.dtypes[col] != 'object']

excludes = [ID, TARGET]

features = [col for col in set(num_features + cat_features) if col not in excludes]

features = sorted(features)
print('shape')

print(X_train.shape, y_train.shape, X_test.shape, len(features))
cv = build_cv_spliter(

    X=X_train,

    y=y_train,

    group=None,

    strategy='kfold',

    n_splits=N_SPLITS,

    shuffle=True,

    random_seed=SEED,

    return_indices=False,

)
params = {

    'boosting_type': 'gbdt',

    'objective': 'rmse',

    'metric': 'rmse',

    'num_iterations': 10000,

    'early_stopping_round': 200,

    'num_leaves': 32,

    'learning_rate': 0.1,

    'bagging_fraction': 0.8,

    'bagging_freq': 5,

    'feature_fraction': 1.0,

    'min_data_in_leaf': 20,

    'lambda_l1': 1.0,

    'lambda_l2': 1.0,

    'seed': SEED,

    'device': 'cpu',

    'verbosity': -1

}



with timer('kfold lightgbm'):

    oof_lgb, predictions_lgb, feature_importance_df = run_kfold_lightgbm(

        params,

        X_train,

        y_train,

        X_test,

        cv,

        features,

        metrics,

        categorical=cat_features,

        verbose_eval=100,

    )
plot_feature_importance(feature_importance_df, fig_width=16, fig_height=2)
oof = oof_lgb

predictions = predictions_lgb
print('price')



print('GT   : mean - {:8.3f}, std - {:8.3f}, max - {:8.3f}, min - {:8.3f}'.format(

    y_train.mean(),

    y_train.std(),

    y_train.max(),

    y_train.min(),

))

print('OOF  : mean - {:8.3f}, std - {:8.3f}, max - {:8.3f}, min - {:8.3f}'.format(

    oof.mean(),

    oof.std(),

    oof.max(),

    oof.min()

))

print('Pred : mean - {:8.3f}, std - {:8.3f}, max - {:8.3f}, min - {:8.3f}'.format(

    predictions.mean(),

    predictions.std(),

    predictions.max(),

    predictions.min()

))



fig, axes = plt.subplots(1, 3)

fig.set_size_inches(3 * 6, 4)



sns.distplot(y_train, kde=False, label='GT', ax=axes[0])

sns.distplot(oof, kde=False, label='oof', ax=axes[0])

axes[0].legend()

axes[0].set_title('GT v.s. oof')



sns.distplot(y_train - oof, kde=False, label='GT', ax=axes[1])

axes[1].set_title('difference')



sns.distplot(oof, kde=False, label='OOF', ax=axes[2])

sns.distplot(predictions, kde=False, label='Pred', ax=axes[2])

axes[2].legend()

axes[2].set_title('oof v.s. pred')



plt.show()
sub = pd.DataFrame()

sub[ID] = test_df[ID].values

sub[TARGET] = predictions

sub.to_csv('submission.csv', index=False)
sub.head()