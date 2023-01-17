# ログに出すキーワード

description_of_this_commit='logistic regression stacking l2'



seed = 71

rounds = 100
import numpy as np

import pandas as pd

import os

from pathlib import Path

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)

import matplotlib.pyplot as plt

%matplotlib inline

import logging

import time

import scipy as sp

import itertools 

import optuna
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_validate

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

import time

from tqdm import tqdm_notebook as tqdm

import lightgbm as lgb

from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder



from keras.layers import Input, Dense, Dropout, BatchNormalization

from keras.optimizers import Adam

from keras.models import Model

from keras.callbacks import EarlyStopping

from sklearn.linear_model import LogisticRegression



from sklearn.dummy import DummyClassifier
## 環境ごとのパスの違いを吸収

import os

if 'KAGGLE_URL_BASE' in os.environ:

    print('running in kaggle kernel')

    data_dir = Path('../input')

    log_dir = Path('./')

else:

    print('running in other environment')

    data_dir = Path('../data/raw')

    log_dir = Path('../log')

data_dir
logger = None
def getLogger():

    global logger

    if logger is not None:

        return logger

    logfile = log_dir / 'all.log'

    logger = logging.getLogger(description_of_this_commit)

    formatter = logging.Formatter('%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')

    fh = logging.FileHandler(logfile, 'a+')

    fh.setFormatter(formatter)

    #fh.setLevel(logging.DEBUG)

    logger.addHandler(fh)

    logger.addHandler(logging.StreamHandler())

    logger.setLevel(logging.DEBUG)

    return logger
logger = getLogger()
df_all_cache = None
# 一番基本的なデータロードが終わったところでキャッシュを保存しておき、二回目以降はキャッシュのコピーを返して高速化

def load_data():

    # load data from file or return cache

    global df_all_cache

    if df_all_cache is not None:

        return df_all_cache.copy()

    df_train = pd.read_csv(data_dir / 'train.csv', index_col=0)

    df_test = pd.read_csv(data_dir / 'test.csv', index_col=0)

    def lookup(s):    

        #via: https://stackoverflow.com/questions/29882573/pandas-slow-date-conversion

        # 行ごとに時刻をparseしないことで高速化

        dates = {date:pd.to_datetime(date) for date in s.unique()}

        return s.map(dates)

    df_test['loan_condition'] = 0.0

    df_train['is_train'] = True

    df_test['is_train'] = False



    df_all_cache = pd.concat([df_train, df_test], axis=0)

    df_all_cache['issue_d'] = lookup(df_all_cache['issue_d'])

    df_all_cache['earliest_cr_line'] = lookup(df_all_cache['earliest_cr_line'])

    return df_all_cache.copy()
def encode_missing_pattern(df_all):

    # 欠損値の存在する列のフラグを並べ一つの2進数としてエンコード

    m = df_all.isnull().sum()

    cols_with_missing = list(m[m != 0].index)



    df_all['missing_pattern'] = 0

    for col in cols_with_missing:

        df_all['missing_pattern'] *= 2

        df_all.loc[df_all[col].isnull(), 'missing_pattern'] += 1

    

    return df_all



def count_missing(df_all):

    # 欠損値の数を返す

    df_all['missing_count'] = df_all.isnull().sum(axis=1)

    return df_all
def missing_value_impute(df_all):

    # カラムごとに決めたルールに従い欠損値補完

    numeric_cols = []

    for col in df_all.columns:

        if df_all[col].dtype in ['int64', 'float64']:

            numeric_cols.append(col)

    numeric_cols.remove('loan_condition')



    #df_all[numeric_cols].isnull().sum()



    imputation_rules = {

     'acc_now_delinq': 0,

     'annual_inc': 'median',

     'collections_12_mths_ex_med': 0,

     'delinq_2yrs': 0,

     'dti': 'median',

     'emp_length_num': 0,

     'inq_last_6mths': 0,

     'mths_since_last_delinq': 'median',

     'mths_since_last_major_derog': 9999,

     'mths_since_last_record': 'median',

     'open_acc': 0,

     'pub_rec': 0,

     'revol_util': 'median',

     'total_acc': 'median',

     'tot_coll_amt': 0,

     'tot_cur_bal': 'median',

    }





    import numbers

    for col, v in df_all[numeric_cols].isnull().sum().iteritems():

        if v == 0:

            continue

        if col not in imputation_rules:

            print('rule not found!!')

        col_missing = f'{col}_missing'

        df_all[col_missing] = 0

        df_all.loc[df_all[col].isnull(), col_missing] = 1

        imputer = imputation_rules[col]

        if isinstance(imputer, numbers.Number):

            df_all.loc[df_all[col].isnull(), col] = imputer

        elif imputer == 'median':

            df_all.loc[df_all[col].isnull(), col] = df_all[col].median()



    # テスト

    col = 'annual_inc'

    col_missing = f'{col}_missing'

    if col_missing in df_all.columns:

        print((df_all[df_all[col_missing] == 1][col] == df_all[col].median()).all())



    col = 'acc_now_delinq'

    col_missing = f'{col}_missing'

    if col_missing in df_all.columns:

        print((df_all[df_all[col_missing] == 1][col] == 0).all())



    col = 'mths_since_last_major_derog'

    col_missing = f'{col}_missing'

    if col_missing in df_all.columns:

        print((df_all[df_all[col_missing] == 1][col] == 9999).all())

    return df_all
def add_ratios(df_all):

    df_all = add_installment_ratio(df_all)

    df_all = add_income_loan_ratio(df_all)

    df_all = add_curbal_income_ratio(df_all)

    df_all = add_install_dti_ratio(df_all)

    return df_all
def add_spi(df_all):

    df_spi = load_spi_data()

    df_all = df_all.reset_index().merge(df_spi, on='issue_d', how='left').set_index('ID')

    return df_all

def add_states(df_all):

    ## state

    df_state = load_state_data()

    df_all = df_all.reset_index().merge(df_state, on='addr_state', how='left').set_index('ID')

    return df_all
def add_other_features(df_all):

    df_all = add_ratios(df_all)

    

    df_all = add_spi(df_all)

    df_all = add_states(df_all)

    df_all = add_time_features(df_all)    

    return df_all
def add_annual_inc_is_clean(df_all):

    # 年収がキリのいい数かどうか

    df_all['annual_inc_is_clean100'] = df_all['annual_inc'].apply(lambda x: 1 if x%100 ==0 else 0)

    df_all['annual_inc_is_clean1000'] = df_all['annual_inc'].apply(lambda x: 1 if x%1000 ==0 else 0)

    df_all['annual_inc_is_clean10000'] = df_all['annual_inc'].apply(lambda x: 1 if x%10000 ==0 else 0)

    

    return df_all
def predict_annual_inc(df_all):

    # 自己申告年収がキリのよくない人を教師データとして他の人の年収を予測し、予実の乖離を特徴量化する

    df_all_inc_pred = df_all.copy()

    df_all_inc_pred.loc[df_all_inc_pred['annual_inc'] > 1000000, 'annual_inc'] = 1000000

    X_train = df_all_inc_pred[df_all_inc_pred['is_train'] & ~df_all_inc_pred['annual_inc_is_clean1000']].drop(columns=['annual_inc'])

    y_train = df_all_inc_pred[df_all_inc_pred['is_train'] & ~df_all_inc_pred['annual_inc_is_clean1000']]['annual_inc']



    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

    clf = LGBMRegressor(n_estimators=9999)

    clf.fit(X_t, y_t, early_stopping_rounds=100,  eval_set=[(X_v, y_v)])



    pred = clf.predict(df_all.drop(columns=['annual_inc']))



    df_all['annual_inc_predicted'] = pred

    df_all.loc[df_all['annual_inc_predicted'] <= 0, 'annual_inc_predicted'] = 100

    df_all['annual_inc_diff_pred_actual'] = df_all['annual_inc'] - df_all['annual_inc_predicted']

    df_all['annual_inc_ratio_pred_actual'] = df_all['annual_inc'] / df_all['annual_inc_predicted']

    return df_all
def add_installment_ratio(df_all):

    #月収に対する月々の支払いの割合。月収(=年収/12)が0の場合は1.0

    df_all.loc[df_all['annual_inc'] != 0, 'installment_ratio'] = df_all[df_all['annual_inc'] != 0].apply(lambda r: r['installment'] / (r['annual_inc'] / 12 ), axis=1)

    df_all.loc[df_all['annual_inc'] == 0, 'installment_ratio'] = 1.0

    return df_all
def add_income_loan_ratio(df_all):

    df_all.loc[df_all['annual_inc'] != 0, 'income_loan_ratio'] = df_all[df_all['annual_inc'] != 0].apply(lambda r: r['loan_amnt'] / (r['annual_inc']), axis=1)

    df_all.loc[df_all['annual_inc'] == 0, 'income_loan_ratio'] = 1.0

    return df_all
def add_curbal_income_ratio(df_all):

    # 預金が少なくても年収が高い人は返済できる、などの組合せ的性質があるかも

    df_all.loc[df_all['annual_inc'] != 0, 'curbal_income_ratio'] = df_all[df_all['annual_inc'] != 0].apply(lambda r: r['tot_cur_bal'] / (r['annual_inc']), axis=1)

    df_all.loc[df_all['annual_inc'] == 0, 'curbal_income_ratio'] = 1.0

    return df_all    
def add_install_dti_ratio(df_all):

    # dtiの定義よりこれが1になるはず？ とりあえず計算

    df_all.loc[df_all['annual_inc']*df_all['dti'] != 0,'install_dti_ratio'] = df_all['installment'] / (df_all['dti'] * df_all['annual_inc'] / 12 )

    df_all.loc[df_all['annual_inc']*df_all['dti'] == 0,'install_dti_ratio'] = 1.0

    

    return df_all
def load_spi_data():

    df_spi = pd.read_csv(data_dir / 'spi.csv', parse_dates=['date'])

    df_spi['year'] = df_spi['date'].dt.year

    df_spi['month'] = df_spi['date'].dt.month

    df_spi = df_spi.groupby(['year', 'month'])['close'].mean().reset_index()

    for window_size in range(2,13):

        col = f'close_roll_{window_size}'

        df_spi[col] = df_spi['close'].rolling(window_size).mean()

        df_spi.loc[df_spi[col].isnull(), col] = 208

    df_spi['day'] = 1

    df_spi['date'] = pd.to_datetime(df_spi[['year', 'month', 'day']])

    df_spi = df_spi.drop(columns = ['year', 'month', 'day'])

    df_spi = df_spi.rename(columns={'date': 'issue_d'})



    return df_spi
def load_state_data():

    df_state = pd.read_csv(data_dir / 'statelatlong.csv')

    df_state = df_state.rename(columns={'State': 'addr_state'})

    df_gdp = pd.read_csv(data_dir / 'US_GDP_by_State.csv')

    df_gdp = df_gdp.rename(columns={'State': 'City', 'State & Local Spending': 'state_spending', 'Gross State Product': 'state_product', 'Real State Growth %': 'state_growth', 'Population (million)': 'state_population'})

    df_state = df_state.merge(df_gdp, on='City')

    df_state = df_state.drop(columns = ['City'])

    df_state = df_state.groupby(['addr_state']).mean().reset_index().drop(columns=['year'])

    return df_state
def feature_combinations(df_all, columns):

    n = len(columns)

    for i in range(n-1):

        for j in range(i+1, n):

            coli = columns[i]

            colj = columns[j]

            new_col = f'mul_{coli}_{colj}'

            df_all[new_col] = df_all[coli] * df_all[colj]

    return df_all
def parse_emp_length(df_all):

    emp_length_map = {'10+ years': 100, '7 years': 7, '5 years': 5, '2 years':2, '8 years':8, '3 years':3,

       '< 1 year':0.5, '1 year':1, '6 years':6, '4 years':4, '9 years':9}

    df_all['emp_length_num'] = df_all.emp_length.map(emp_length_map)

    return df_all
def segmentate(df_all):

    ## annual_inct, loan_amntをセグメント化する

    segs = [0] + list(np.linspace(10000, 90000, 9)) + list(np.linspace(100000, 1000000, 10))

    df_all['annual_inc_seg'] = f'over {int(segs[-1]) // 1000}k'

    for i in range(1, len(segs)-1):

        lower = segs[i-1]

        upper = segs[i]

        df_all.loc[(lower <= df_all['annual_inc']) & (df_all['annual_inc'] < upper), 'annual_inc_seg'] = f'{int(lower//1000):03}k to {int(upper//1000):03}k'

    segs = np.linspace(0, df_all['loan_amnt'].max(), 21)



    df_all['loan_amnt_seg'] = f'over {int(segs[-1]) // 1000}k'

    for i in range(1, len(segs)-1):

        lower = segs[i-1]

        upper = segs[i]

        df_all.loc[(lower <= df_all['loan_amnt']) & (df_all['loan_amnt'] < upper), 'loan_amnt_seg'] = f'{int(lower//1000):03}k to {int(upper//1000):03}k'

    return df_all
def _ordinal_encode_cat_pair(df_all, col1, col2):

    map1 = {x: i for i, x in enumerate(sorted(df_all[col1].unique()))}

    map2 = {x: i for i, x in enumerate(sorted(df_all[col2].unique()))}

    return df_all[col1].map(map1) *1000 + df_all[col2].map(map2)



def ordinal_encode_cat_comb(df_all, col_combs=None):

    if col_combs is None:

        col_combs = []

        cols = ['loan_amnt_seg', 'annual_inc_seg', 'grade', 'sub_grade']

        for i in range(len(cols)):

            for j in range(len(cols)):

                if i == j:

                    continue

                col1, col2 = cols[i], cols[j]

                if 'grade' in col1 and 'grade' in col2:

                    continue

                col_combs.append([col1, col2])

    for col1, col2 in col_combs:

        new_col = f'pair_enc_{col1}__{col2}'

        df_all[new_col] = _ordinal_encode_cat_pair(df_all, col1, col2)

    return df_all
def remove_time_features(df_all):

    cols_time_features = []

    for col in df_all.columns:

        if df_all[col].dtype == 'datetime64[ns]':

            cols_time_features.append(col)

    

    if cols_time_features:

        df_all = df_all.drop(columns=['issue_d', 'earliest_cr_line'])

    return df_all
def add_time_features(df_all):

    meantime_till_issue = (df_all['issue_d'] -  df_all['earliest_cr_line']).mean()

    

    df_all['earliest_cr_line_missing'] = 0

    df_all.loc[df_all['earliest_cr_line'].isnull(), 'earliest_cr_line_missing'] = 1

    

    # impute earliest_cr_line

    df_all.loc[df_all['earliest_cr_line'].isnull(), 'earliest_cr_line'] = df_all['issue_d'] - meantime_till_issue



    # add days since earliest_cr_line

    df_all['days_since_earliest_cr_line'] = (df_all['issue_d'] - df_all['earliest_cr_line']).dt.days

    

    df_all['issue_month'] = df_all['issue_d'].dt.month

    return df_all
def encode_categorical_features(df_all):

    missing_marker = '__MISSING_VALUE__'

    categorical_cols = []

    for col in df_all.columns:

        if df_all[col].dtype in ['object', 'datetime64[ns]']:

            if col not in ['issue_d', 'grade', 'sub_grade']:

                categorical_cols.append(col)



    ## missing values

    cols_with_none = ['emp_title', 'emp_length', 'title']

    for col in cols_with_none:

        col_missing = f'{col}_missing'

        # add flag

        df_all[col_missing] = 0

        df_all.loc[df_all[col].isnull(), col_missing] = 1

        # impute

        df_all.loc[df_all[col].isnull(), col] = missing_marker



        

    for col in categorical_cols:

        # replace cols with count encoded values

        new_col = f'{col}_count'

        df_all[new_col] = df_all[col].map(df_all[col].value_counts())

        

    # grade / sub_gradeは特別扱いしアルファベット順に連番を振る

    for f in ['grade', 'sub_grade']:

        gs = sorted(df_all[f].unique())

        df_all[f] = df_all[f].map({g: i for (g, i) in zip(gs, range(len(gs)))})

    return df_all
from numpy.random import normal

## via https://www.kaggle.com/vprokopev/mean-likelihood-encodings-a-comprehensive-study

def mean_encode(train_data, test_data, columns, target_col, reg_method=None,

                alpha=5, add_random='k_fold', rmean=0, rstd=0.1, folds=5):

    '''Returns a DataFrame with encoded columns'''

    encoded_cols = []

    target_mean_global = train_data[target_col].mean()

    for col in columns:

        # Getting means for test data

        nrows_cat = train_data.groupby(col)[target_col].count()

        target_means_cats = train_data.groupby(col)[target_col].mean()

        target_means_cats_adj = (target_means_cats*nrows_cat + 

                                 target_mean_global*alpha)/(nrows_cat+alpha)

        # Mapping means to test data

        encoded_col_test = test_data[col].map(target_means_cats_adj)

        # Getting a train encodings

        if reg_method == 'expanding_mean':

            train_data_shuffled = train_data.sample(frac=1, random_state=1)

            cumsum = train_data_shuffled.groupby(col)[target_col].cumsum() - train_data_shuffled[target_col]

            cumcnt = train_data_shuffled.groupby(col).cumcount()

            encoded_col_train = cumsum/(cumcnt)

            encoded_col_train.fillna(target_mean_global, inplace=True)

            if add_random:

                encoded_col_train = encoded_col_train + normal(loc=rmean, scale=rstd, 

                                                               size=(encoded_col_train.shape[0]))

        elif (reg_method == 'k_fold') and (folds > 1):

            kfold = StratifiedKFold(train_data[target_col].values, folds, shuffle=True, random_state=1)

            parts = []

            for tr_in, val_ind in kfold:

                # divide data

                df_for_estimation, df_estimated = train_data.iloc[tr_in], train_data.iloc[val_ind]

                # getting means on data for estimation (all folds except estimated)

                nrows_cat = df_for_estimation.groupby(col)[target_col].count()

                target_means_cats = df_for_estimation.groupby(col)[target_col].mean()

                target_means_cats_adj = (target_means_cats*nrows_cat + 

                                         target_mean_global*alpha)/(nrows_cat+alpha)

                # Mapping means to estimated fold

                encoded_col_train_part = df_estimated[col].map(target_means_cats_adj)

                if add_random:

                    encoded_col_train_part = encoded_col_train_part + normal(loc=rmean, scale=rstd, 

                                                                             size=(encoded_col_train_part.shape[0]))

                # Saving estimated encodings for a fold

                parts.append(encoded_col_train_part)

            encoded_col_train = pd.concat(parts, axis=0)

            encoded_col_train.fillna(target_mean_global, inplace=True)

        else:

            encoded_col_train = train_data[col].map(target_means_cats_adj)

            if add_random:

                encoded_col_train = encoded_col_train + normal(loc=rmean, scale=rstd, 

                                                               size=(encoded_col_train.shape[0]))



        # Saving the column with means

        encoded_col = pd.concat([encoded_col_train, encoded_col_test], axis=0)

        encoded_col[encoded_col.isnull()] = target_mean_global

        encoded_cols.append(pd.DataFrame({'mean_'+target_col+'_'+col:encoded_col}))

    all_encoded = pd.concat(encoded_cols, axis=1)

    return (all_encoded.loc[train_data.index,:], 

            all_encoded.loc[test_data.index,:])

def remove_categorical_features(df_all):

    categorical_features = []

    text_features = ['emp_title', 'title']

    for col in df_all.columns:

        if df_all[col].dtype == 'object' and col not in text_features:

            categorical_features.append(col)

    df_all = df_all.drop(columns=text_features)

    df_all = df_all.drop(columns=categorical_features)

    return df_all
def remove_old_records(df_all):

    df_all = df_all[(df_all['issue_d'] > pd.to_datetime('2014-01-01 00:00:00'))]

    # rows reduced (1321847, 33) => (1075503, 33)

    return df_all
def check_score_time_split(df_al):

    X_t, y_t, X_v, y_v = time_split(df_all)

    clf = LGBMClassifier(n_estimators=9999)

    clf.fit(X_t, y_t, early_stopping_rounds=30, eval_metric='auc', eval_set=[(X_v, y_v)])

    pred = clf.predict_proba(X_v)

    auc = roc_auc_score(y_v, pred[:,1])

    return auc
def get_score_on_text_feature(df_all, text_col, max_features=1000):

    ## text_colのカラムだけを使って予測モデルを作り、得られたスコアを特徴量として返す

    missing_marker = '__MISSING_VALUE__'

    df_all.loc[df_all[text_col].isnull(), text_col] = missing_marker

    vec = TfidfVectorizer(stop_words='english', max_features=max_features)

    term_doc = vec.fit_transform(df_all[text_col])

    X_train = term_doc[np.where(df_all['is_train'])]

    y_train = df_all[df_all['is_train']]['loan_condition']

    X_test = term_doc[np.where(~ df_all['is_train'])]

    

    stacking_scores, scores, pred = cv_score_lgb(X_train, y_train, n_splits=5, rounds=30, X_test=X_test)

    

    return np.append(stacking_scores, pred), scores
def cv_score_lgb(X_train, y_train, params={}, n_splits=5, rounds=30, X_test=None, categorical_feature=None):

    logger.info(f'cv_score_lgb rounds={rounds}')

    # calc cv averaging when X_test is not None

    

    ## cv score (for stacking)

    stacking_scores = pd.DataFrame({'score': np.zeros(X_train.shape[0])})

    scores = []

    predictions = []

    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)

    for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

        # sparseな場合

        if type(X_train) == sp.sparse.csr.csr_matrix:

            X_t = X_train[train_ix]

            y_t = y_train.iloc[train_ix]

            X_v = X_train[test_ix]

            y_v = y_train.iloc[test_ix]

        else:

            X_t, y_t = X_train.iloc[train_ix], y_train.iloc[train_ix]

            X_v, y_v = X_train.iloc[test_ix], y_train.iloc[test_ix]

        clf = LGBMClassifier(n_estimators=9999, random_state=seed, **params)

        logger.info(f'cv_score_lgb classifier={clf}')

        if categorical_feature:

            clf.fit(X_t, y_t, early_stopping_rounds=rounds, eval_metric='auc', eval_set=[(X_v, y_v)], categorical_feature=categorical_feature, verbose=50)

        else:

            clf.fit(X_t, y_t, early_stopping_rounds=rounds, eval_metric='auc', eval_set=[(X_v, y_v)], verbose=50)

        #clf = DummyClassifier()

        #clf.fit(X_t, y_t)

        y_pred  = clf.predict_proba(X_v)[:,1]

        # test_ixはインデックスではなく行番号のリストなのでilocでアクセス

        stacking_scores.iloc[test_ix, stacking_scores.columns.get_loc('score')]= y_pred

        score = roc_auc_score(y_v, y_pred)

        scores.append(score)

        if X_test is not None:

            y_pred_test  = clf.predict_proba(X_test)[:,1]

            predictions.append(y_pred_test)

    mean = sum(scores) / n_splits

    logger.info(f'cv_score_lgb  scores={scores}, mean={mean}')

    

    stacking_scores = stacking_scores['score'].values

    if X_test is not None:

        pred = sum(predictions) / n_splits

        return stacking_scores, scores, pred

    else:

        return stacking_scores, scores
def create_nn_model(input_len):

    inp = Input(shape=(input_len,))#, sparse=True) # 疎行列を入れる

    x = Dense(194, activation='relu')(inp)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Dense(64, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Dense(64, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    outp = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model





def cv_score_nn(X_train, y_train, params={}, n_splits=5, epochs=99, X_test=None):

    logger.info(f'cv_score_nn starts')

    # calc cv averaging when X_test is not None

    

    ## cv score (for stacking)

    stacking_scores = pd.DataFrame({'score': np.zeros(X_train.shape[0])})

    scores = []

    predictions = []

    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)

    for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

        X_t, y_t = X_train.iloc[train_ix], y_train.iloc[train_ix]

        X_v, y_v = X_train.iloc[test_ix], y_train.iloc[test_ix]

        model = create_nn_model(X_t.shape[1])

        logger.info(f'cv_score_nn classifier={model}')

        es = EarlyStopping(monitor='val_loss', patience=0)

        model.fit(X_t, y_t, batch_size=512, epochs=epochs, validation_data=(X_v, y_v), callbacks=[es])

        y_pred = model.predict(X_v) # predict_proba[:,1]でない点に注意

        y_pred = (y_pred.T)[0]

        stacking_scores.iloc[test_ix, stacking_scores.columns.get_loc('score')]= y_pred

        score = roc_auc_score(y_v, y_pred)

        scores.append(score)

        if X_test is not None:

            y_pred_test = model.predict(X_test) 

            y_pred_test = (y_pred_test.T)[0]

            predictions.append(y_pred_test)

    mean = sum(scores) / n_splits

    logger.info(f'cv_score_lgb  scores={scores}, mean={mean}')

    

    stacking_scores = stacking_scores['score'].values

    if X_test is not None:

        pred = sum(predictions) / n_splits

        return stacking_scores, scores, pred

    else:

        return stacking_scores, scores
def cv_score_logreg(X_train, y_train, params={}, n_splits=5, X_test=None):

    logger.info(f'cv_score_logreg')

    # calc cv averaging when X_test is not None

    

    ## cv score (for stacking)

    stacking_scores = pd.DataFrame({'score': np.zeros(X_train.shape[0])})

    scores = []

    predictions = []

    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)

    for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

        X_t, y_t = X_train.iloc[train_ix], y_train.iloc[train_ix]

        X_v, y_v = X_train.iloc[test_ix], y_train.iloc[test_ix]

        clf = LogisticRegression(**params)

        logger.info(f'cv_score_logreg classifier={clf}')

        clf.fit(X_t, y_t)

        y_pred  = clf.predict_proba(X_v)[:,1]



        stacking_scores.iloc[test_ix, stacking_scores.columns.get_loc('score')]= y_pred

        score = roc_auc_score(y_v, y_pred)

        scores.append(score)

        if X_test is not None:

            y_pred_test  = clf.predict_proba(X_test)[:,1]

            predictions.append(y_pred_test)

    mean = sum(scores) / n_splits

    logger.info(f'cv_score_logreg  scores={scores}, mean={mean}')

    

    stacking_scores = stacking_scores['score'].values

    if X_test is not None:

        pred = sum(predictions) / n_splits

        return stacking_scores, scores, pred

    else:

        return stacking_scores, scores
def importance(gb, X_test):

    feature_importance = gb.feature_importances_

    sorted_idx = np.argsort(feature_importance)

    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.figure(figsize=(30, 16))

    plt.subplot(1, 2, 2)

    plt.barh(pos, feature_importance[sorted_idx], align='center')

    plt.yticks(pos, list(X_test.columns[sorted_idx]))

    plt.xlabel('Relative Importance')

    plt.title('Variable Importance')

    plt.show()

    return X_test.columns[sorted_idx]

def check_score_time_split_lgb(df_all, th='2014-05-01 00:00:00'):

    X_train, y_train, X_val, y_val = time_split(df_all, th)

    clf = LGBMClassifier(n_estimators=9999, random_state=seed)

    clf.fit(X_train, y_train, early_stopping_rounds=30, eval_metric='auc', eval_set=[(X_val, y_val)])

    y_pred  = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    logger.info(f'check_score_time_split_lgb  score={score}')

    return score
def add_timesplit_flag(df_all, th='2014-05-01 00:00:00'):

    df_all['is_recent'] = True

    df_all.loc[(df_all.issue_d <= pd.to_datetime(th)), 'is_recent'] = False

    return df_all
def my_gridsearch_lgb(X_train, y_train, X_val, y_val, grid_params):

    ## early_stoppingありでGridSearchCVを使う方法がわからないので再実装

    func_name = 'my_gridsearch_lgb'

    gridsearch_results = []

    values = grid_params.values()

    for values in itertools.product(*values):

        param = {k:v for k,v in zip(grid_params.keys(), values)}



        logger.info(f'{func_name} start validation for param={param}')

        clf = LGBMClassifier(n_estimators=9999, random_state=seed, **param)

        clf.fit(X_train, y_train, early_stopping_rounds=30, eval_metric='auc', eval_set=[(X_v, y_v)])

        #clf = DummyClassifier()

        #clf.fit(X_t, y_t)

        y_pred  = clf.predict_proba(X_val)[:,1]

        score = roc_auc_score(y_val, y_pred)

        logger.info(f'{func_name}  validation done for param={param}, score={score}')

        gridsearch_results.append({'param': param, 'score': score})

    

    best_score = max(gridsearch_results, key=lambda x: x['score'])['score']

    best_param = max(gridsearch_results, key=lambda x: x['score'])['param']

    return gridsearch_results, best_score, best_param    



def my_gridsearch_cv_lgb(X_train, y_train, grid_params, n_splits=5):

    ## early_stoppingありでGridSearchCVを使う方法がわからないので再実装

    func_name = 'my_cv_lgb'

    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)

    cv_results = []

    values = grid_params.values()

    for values in itertools.product(*values):

        param = {k:v for k,v in zip(grid_params.keys(), values)}

        scores = []



        logger.info(f'{func_name} start cv for param={param}')

        for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

            # sparseな場合

            if type(X_train) == sp.sparse.csr.csr_matrix:

                X_t = X_train[train_ix]

                y_t = y_train.iloc[train_ix]

                X_v = X_train[test_ix]

                y_v = y_train.iloc[test_ix]

            else:

                X_t, y_t = X_train.iloc[train_ix], y_train.iloc[train_ix]

                X_v, y_v = X_train.iloc[test_ix], y_train.iloc[test_ix]



            clf = LGBMClassifier(n_estimators=9999,  random_state=seed, **param)

            clf.fit(X_t, y_t, early_stopping_rounds=30, eval_metric='auc', eval_set=[(X_v, y_v)])

            #clf = DummyClassifier()

            #clf.fit(X_t, y_t)

            y_pred  = clf.predict_proba(X_v)[:,1]

            score = roc_auc_score(y_v, y_pred)

            scores.append(score)

        mean = sum(scores) / n_splits

        logger.info(f'{func_name}  cv done for param={param}, scores={scores}, mean={mean}')

        cv_results.append({'param': param, 'scores': scores, 'mean_score': mean})

    

    best_score = max(cv_results, key=lambda x: x['mean_score'])['mean_score']

    best_param = max(cv_results, key=lambda x: x['mean_score'])['param']

    return cv_results, best_score, best_param
def tune_lgb_cv(X_train, y_train, params, n_splits=5, n_trials=5, rounds=30, categorical_feature=None, metric='mean'):

    func_name = 'tune_lgb_cv'

    logger.info(f'{func_name} start hyperparameter search')



    def objective(trial):

        param = {}

        for p in params:

            kv = params[p]

            t = kv['type']

            if t == 'fixed':

                param[p] = kv['value']

            elif t == 'int':

                param[p] = trial.suggest_int(p, kv['lower'], kv['upper'])

            elif t == 'uniform':

                param[p] = trial.suggest_uniform(p, kv['lower'], kv['upper'])

            elif t == 'loguniform':

                param[p] = trial.suggest_loguniform(p, kv['lower'], kv['upper'])



        aucs = []

        skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)



        for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

            # sparseな場合

            if type(X_train) == sp.sparse.csr.csr_matrix:

                X_t = X_train[train_ix]

                y_t = y_train.iloc[train_ix]

                X_v = X_train[test_ix]

                y_v = y_train.iloc[test_ix]

            else:

                X_t, y_t = X_train.iloc[train_ix], y_train.iloc[train_ix]

                X_v, y_v = X_train.iloc[test_ix], y_train.iloc[test_ix]



            clf = LGBMClassifier(n_estimators=9999, random_state=seed,  **param)

            if categorical_feature:

                clf.fit(X_t, y_t, early_stopping_rounds=rounds, eval_metric='auc', eval_set=[(X_v, y_v)], categorical_feature=categorical_feature)

            else:

                clf.fit(X_t, y_t, early_stopping_rounds=rounds, eval_metric='auc', eval_set=[(X_v, y_v)])



            y_pred  = clf.predict_proba(X_v)[:,1]

            auc = roc_auc_score(y_v, y_pred)

            aucs.append(auc)

            logger.info(f'{func_name} a fold with param={param}, auc={auc}')



        if metric == 'mean':

            score = sum(aucs) / n_splits

        elif metric == 'min':

            score = min(aucs)

        logger.info(f'{func_name}  cv done with param={param}, aucs={aucs}, score={score}, metric={metric}')

  

        return -score

    study = optuna.create_study()

    study.optimize(objective, n_trials=n_trials)

    return study
def tune_lgb(X_train, y_train, X_val, y_val, params, n_trials=5):

    func_name = 'tune_lgb'

    logger.info(f'{func_name} start hyperparameter search')



    def objective(trial):

        param = {}

        for p in params:

            kv = params[p]

            t = kv['type']

            if t == 'fixed':

                param[p] = kv['value']

            elif t == 'int':

                param[p] = trial.suggest_int(p, kv['lower'], kv['upper'])

            elif t == 'uniform':

                param[p] = trial.suggest_uniform(p, kv['lower'], kv['upper'])

            elif t == 'loguniform':

                param[p] = trial.suggest_loguniform(p, kv['lower'], kv['upper'])

        clf = LGBMClassifier(n_estimators=9999,  random_state=seed, **param)

        clf.fit(X_train, y_train, early_stopping_rounds=300, eval_metric='auc', eval_set=[(X_val, y_val)])

        y_pred  = clf.predict_proba(X_val)[:,1]

        score = roc_auc_score(y_val, y_pred)

        logger.info(f'{func_name} validation done for param={param}, score={score}')

        return -score

    study = optuna.create_study()

    study.optimize(objective, n_trials=n_trials)

    logger.info(f'{func_name} finished. best_params={study.best_params}, best_values={study.best_value}')

    return study
df_all = load_data()
## テキストデータのTFIDFのみで予測したmeta featureを追加

for col in ['emp_title', 'title']:

    %time scores, auc = get_score_on_text_feature(df_all, col)

    new_col = f'lgbscore_{col}'

    df_all[new_col] = scores
df_all = add_timesplit_flag(df_all)

df_all = encode_missing_pattern(df_all)

df_all = count_missing(df_all)

#df_all = remove_old_records(df_all)

df_all = parse_emp_length(df_all)

df_all = missing_value_impute(df_all)



#df_all = segmentate(df_all)

#df_all = ordinal_encode_cat_comb(df_all)



df_all = add_annual_inc_is_clean(df_all)



df_all = add_ratios(df_all)

df_all = add_states(df_all)

#df_all = add_spi(df_all)

df_all = add_time_features(df_all)



df_all_back_for_lgbm_catencode = df_all.copy()
df_all = encode_categorical_features(df_all)

df_all = remove_categorical_features(df_all)

df_all = remove_time_features(df_all)

#df_all = predict_annual_inc(df_all)
df_all_back_for_numerical_model = df_all.copy()

df_all = df_all_back_for_numerical_model.copy()
X_train = df_all[df_all['is_train']].drop(columns=['loan_condition', 'is_train', 'is_recent'])

y_train = df_all[df_all['is_train']]['loan_condition']

X_test = df_all[~ (df_all['is_train'])].drop(columns=['is_train', 'loan_condition', 'is_recent'])

for col in X_train:

    scaler = StandardScaler()

    scaler.fit(X_train[[col]])

    X_train[[col]] = scaler.transform(X_train[[col]])

    X_test[[col]] = scaler.transform(X_test[[col]])
%time stacking_scores, scores, pred = cv_score_nn(X_train, y_train, n_splits=5, epochs=99, X_test=X_test)
metafeature_nn = np.append(stacking_scores, pred)
df_all = df_all_back_for_lgbm_catencode.copy()
df_all = segmentate(df_all)

df_all = ordinal_encode_cat_comb(df_all)
categorical_features = [col for col in df_all.columns if df_all[col].dtypes in ['object', 'datetime64[ns]']]
df_all.loc[df_all['emp_length'].isnull(), 'emp_length'] = '__MISSING_VALUE__'
for col in categorical_features:

    le = LabelEncoder()

    le.fit(df_all[col])

    df_all[col] = le.transform(df_all[col])
# 組合せで生成したカテゴリ変数も追加

categorical_features += [col for col in df_all.columns if 'pair_enc' in col]
X_train = df_all[df_all['is_train']].drop(columns=['loan_condition', 'is_train', 'is_recent'])

y_train = df_all[df_all['is_train']]['loan_condition']

X_test = df_all[~ (df_all['is_train'])].drop(columns=['is_train', 'loan_condition', 'is_recent'])
current_best_params_for_lgb2 = {'min_child_samples': 60, 'num_leaves': 15, 'min_child_weight': 3, 'subsample': 0.5451137318009219, 'colsample_bytree': 0.05220322801009669}



%time stacking_scores, scores, pred = cv_score_lgb(X_train, y_train,  params=current_best_params_for_lgb2, n_splits=5, rounds=rounds, X_test=X_test)

metafeature_lgbm = np.append(stacking_scores, pred)
df_all = df_all_back_for_numerical_model.copy()
X_train = df_all[df_all['is_train']].drop(columns=['loan_condition', 'is_train', 'is_recent'])

y_train = df_all[df_all['is_train']]['loan_condition']

X_test = df_all[~ (df_all['is_train'])].drop(columns=['is_train', 'loan_condition', 'is_recent'])



#current_best_params_for_lgb = {'min_child_samples': 74, 'num_leaves': 12, 'min_child_weight': 11, 'subsample': 0.6130339453753051, 'colsample_bytree': 0.23506396376975347}

current_best_params_for_lgb = {'min_child_samples': 120, 'num_leaves': 21, 'min_child_weight': 31, 'subsample': 0.8168082691950477, 'colsample_bytree': 0.46720503612089026}



#current_best_params_for_lgb = {'min_child_samples': 120, 'num_leaves': 78, 'min_child_weight': 2, 'subsample': 0.6175732431082002, 'colsample_bytree': 0.4890612005186778, 'learning_rate': 0.004327392867032571}

%time stacking_scores, scores, pred = cv_score_lgb(X_train, y_train,  params=current_best_params_for_lgb, n_splits=5, rounds=rounds, X_test=X_test)

metafeature_lgbm2 = np.append(stacking_scores, pred)
df_all = load_data()

df_all['lgbm'] = metafeature_lgbm

df_all['lgbm2'] = metafeature_lgbm2

df_all['nn'] = metafeature_nn

#df_all['logreg'] = metafeature_logreg



df_all = df_all[['lgbm', 'lgbm2', 'nn',  'loan_condition', 'is_train']]

X_train = df_all[df_all['is_train']].drop(columns=['loan_condition', 'is_train'])

y_train = df_all[df_all['is_train']]['loan_condition']

X_test = df_all[~ (df_all['is_train'])].drop(columns=['is_train', 'loan_condition'])



logreg_param = {'C': 0.001, 'penalty': 'l1'}

%time stacking_scores, scores, pred = cv_score_logreg(X_train, y_train,  params=logreg_param, n_splits=5, X_test=X_test)

import datetime

ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

submission = pd.read_csv(data_dir / 'sample_submission.csv', index_col=0).drop(columns=['loan_condition'])

submission['loan_condition'] = pred

submission.to_csv(f'./submission_{description_of_this_commit}_{ts}.csv')
# head and tail latest submission

!ls -tr | grep submission | tail -n 1 | xargs head

!echo

!ls -tr | grep submission | tail -n 1 | xargs tail