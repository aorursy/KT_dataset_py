import pandas as pd

import numpy as np

from pathlib import Path

import os

import re

from collections import defaultdict

import math
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
seed_averaging = True
seeds = [71, 7, 64, 65, 12345]
## 環境ごとのパスの違いを吸収

import os

if 'KAGGLE_URL_BASE' in os.environ:

    print('running in kaggle kernel')

    data_dir = Path('/kaggle/input')

else:

    print('running in other environment')

    data_dir = Path('./')

data_dir
train_file = data_dir / 'train.csv'

test_file = data_dir / 'test.csv'

country_file = data_dir / 'country_info.csv'
col_target = 'ConvertedSalary'
pd.set_option('display.max_columns', 120)

pd.set_option('display.max_rows', 150)
def remove_outliers(df_all):

    # 年収max値の2000000.0と回答している人が227人いる。怪しいので除去する

    df_all = df_all.loc[df_all[col_target] != 2000000.0]

    # 0も除去してしまうか？ 0の予測を外すとエラーが極めて大きいので0かどうか予測するモデルを作ったらいいのでは？

    df_all = df_all.loc[df_all[col_target] != 0]

    

    return df_all
def add_country_info(df_all):

    df_country = pd.read_csv(country_file)

    df_all = pd.merge(df_all, df_country, on='Country', how='left')

    return df_all
df_all_cache = None
# 一番基本的なデータロードが終わったところでキャッシュを保存しておき、二回目以降はキャッシュのコピーを返して高速化

def load_data():

    # load data from file or return cache

    global df_all_cache

    if df_all_cache is not None:

        return df_all_cache.copy()



    df_train = pd.read_csv(train_file)

    df_test = pd.read_csv(test_file)

    df_country = pd.read_csv(country_file)

    

    df_train['is_train'] = True

    df_test['is_train'] = False

    df_test[col_target] = -0.5

    df_test = df_test[df_train.columns]



    df_all_cache = pd.concat([df_train, df_test], axis=0)

    return df_all_cache.copy()
def encode_multiple_answers(df_all):

    # 複数回答カラムを数値にエンコード。

    # 1. 回答数を追加

    # 2. a;b;d → 8 + 4 + 1 のような感じでエンコード

    #  もっとも登場回数が大きいValueが上位ビットにエンコードされるようにする

    #  欠損値が埋められていることを前提とする

    multiple_cols = ["DevType", "CommunicationTools", "FrameworkWorkedWith"]

    

    

    for col in multiple_cols:

        # 回答数

        new_col = col + '_answercount'

        df_all[new_col] = df_all[col].apply(lambda x: len(re.split('\s*;\s*' , x)))

        

    for col in multiple_cols:

        # エンコード

        new_col = col + '_multiple_encoded'



        unique_values = defaultdict(int)

        for row in df_all[col]:

            vs = re.split('\s*;\s*', row)

            for v in vs:

                unique_values[v] += 1

        score_map = {}

        val = 1

        for k in sorted(unique_values.keys(), key=lambda k: unique_values[k]):

            score_map[k] = val

            val *= 2

        def to_score(record):

            vs = re.split('\s*;\s*', record)

            score = 0

            for v in vs:

                if v in score_map:

                    score += score_map[v]

            return score

        df_all[new_col] = df_all[col].apply(to_score)

    return df_all
def encode_missing_pattern(df_all):

    # 欠損値の存在する列のフラグを並べ一つの2進数としてエンコード

    m = df_all.isnull().sum()

    cols_with_missing = list(m[m != 0].index)



    df_all['missing_pattern'] = 0

    for col in cols_with_missing:

        df_all['missing_pattern'] *= 2

        df_all.loc[df_all[col].isnull(), 'missing_pattern'] += 1

    

    # ケタが大きくなりすぎるので小さくする

    df_all['missing_pattern'] *= 1e-16

    return df_all
def count_missing(df_all):

    # 欠損値の数を返す

    df_all['missing_count'] = df_all.isnull().sum(axis=1)

    return df_all
def missing_value_impute_numbers(df_all):

    # 数値は-1で埋める

    numeric_cols = []

    for col in df_all.columns:

        if df_all[col].dtype in ['int64', 'float64']:

            numeric_cols.append(col)

    numeric_cols.remove(col_target)



    for col, v in df_all[numeric_cols].isnull().sum().iteritems():

        if v == 0:

            continue

        # 埋めたことを表すフラグはつけない。欠損パターンに法則がありそうなのでパターンだけ別関数でもたせる

        #col_missing = f'{col}_missing'

        #df_all[col_missing] = 0

        #df_all.loc[df_all[col].isnull(), col_missing] = 1

        df_all.loc[df_all[col].isnull(), col] = -1



    return df_all
def missing_value_impute_categories(df_all):

    # カテゴリはマーカーで埋める

    missing_marker = '__MISSING_VALUE__'

    categorical_cols = []

    for col in df_all.columns:

        if df_all[col].dtype in ['object']:

            categorical_cols.append(col)



    for col, v in df_all[categorical_cols].isnull().sum().iteritems():

        if v == 0:

            continue

        df_all.loc[df_all[col].isnull(), col] = missing_marker



    return df_all
def encode_categorical_features(df_all):

    categorical_cols = []

    for col in df_all.columns:

        if df_all[col].dtype in ['object']:

            categorical_cols.append(col)

        

    for col in categorical_cols:

        # replace cols with count encoded values

        #new_col = f'{col}_count'

        df_all[col] = df_all[col].map(df_all[col].value_counts())

        

    # grade / sub_gradeは特別扱いしアルファベット順に連番を振る

    #for f in ['grade', 'sub_grade']:

    #    gs = sorted(df_all[f].unique())

    #   df_all[f] = df_all[f].map({g: i for (g, i) in zip(gs, range(len(gs)))})

    return df_all
def cv_score_lgb(X_train, y_train, params={}, n_splits=5, rounds=30, X_test=None, categorical_feature=None, seed=None, threashold=None):

    print(f'cv_score_lgb rounds={rounds}')

    # calc cv averaging when X_test is not None

    

    ## cv score (for stacking)

    stacking_scores = pd.DataFrame({'score': np.zeros(X_train.shape[0])})

    scores = []

    predictions = []

    kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)

    #skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)

    for i, (train_ix, test_ix) in enumerate(tqdm(kf.split(X_train, y_train))):

        X_t, y_t = X_train.iloc[train_ix], y_train.iloc[train_ix]

        X_v, y_v = X_train.iloc[test_ix], y_train.iloc[test_ix]

        clf = LGBMRegressor(n_estimators=9999, random_state=seed, **params)

        print(f'cv_score_lgb classifier={clf}')

        clf.fit(X_t, y_t, early_stopping_rounds=rounds, eval_metric='rmse', eval_set=[(X_v, y_v)], verbose=100)



        y_pred  = clf.predict(X_v)

        if threashold is not None:

            y_pred = np.clip(y_pred, threashold, None)

        # test_ixはインデックスではなく行番号のリストなのでilocでアクセス

        stacking_scores.iloc[test_ix, stacking_scores.columns.get_loc('score')]= y_pred

        # calculate RMSLE here

        score = math.sqrt(sum((y_v - y_pred)**2) / (len(y_pred)))

        scores.append(score)

        if X_test is not None:

            y_pred_test  = clf.predict(X_test)

            predictions.append(y_pred_test)

    mean = sum(scores) / n_splits

    print(f'cv_score_lgb  scores={scores}, mean={mean}')

    

    stacking_scores = stacking_scores['score'].values

    if X_test is not None:

        pred = sum(predictions) / n_splits

        return stacking_scores, scores, pred

    else:

        return stacking_scores, scores
df_all = load_data()
# 上がらない

# df_all = remove_outliers(df_all)
# 単純に入れたら下がったからパス

# df_all = add_country_info(df_all)
# 評価指標がRMSLEなのでターゲットをログ変換してからモデリングする

df_all[col_target] = np.log(df_all[col_target] + 1)
df_all = count_missing(df_all)

df_all = encode_missing_pattern(df_all)

df_all = missing_value_impute_numbers(df_all)

df_all = missing_value_impute_categories(df_all)
# 複数回答のエンコード

df_all = encode_multiple_answers(df_all)
df_all = encode_categorical_features(df_all)
X_train = df_all[df_all['is_train']].drop(columns=[col_target, 'is_train'])

y_train = df_all[df_all['is_train']][col_target]

X_test = df_all[~ (df_all['is_train'])].drop(columns=[col_target, 'is_train'])

if seed_averaging:

    print('performing seed averaging')

    preds = []

    for seed in seeds:

        stacking_scores, scores, pred = cv_score_lgb(X_train, y_train, rounds=100, X_test=X_test, seed=seed)

        preds.append(pred)

    pred = sum(preds) / len(seeds)

else:

    stacking_scores, scores, pred = cv_score_lgb(X_train, y_train, rounds=100, X_test=X_test, seed=seeds[0])
submission = X_test.copy()

submission = submission[['Respondent']]

submission[col_target] = pred
# logの逆変換

submission[col_target] = np.exp(submission[col_target]) - 1
# 丸め(繰り上げ)

submission[col_target] = submission[col_target].apply(math.ceil)

# 丸め(四捨五入)

#submission[col_target] = submission[col_target].apply(round)

# 極端に小さな予測値を保守的にする

threashold = math.e ** 0.2 

submission.loc[submission[col_target] < threashold, col_target] = threashold
submission.to_csv('submission.csv', index=False)