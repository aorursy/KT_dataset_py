# 開始時間
from datetime import datetime, timedelta, timezone
JST = timezone(timedelta(hours=+9), 'JST')
print(datetime.now(JST))
!pip install kaggle
!pip install datarobot
# 基本
import datetime as dt
import random
import glob
import cv2
import os
from os import path
import gc
import sys
import json

# データ加工
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import scipy as sp
import scipy.sparse as sps

# データ可視化
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker
plt.style.use('ggplot')
%matplotlib inline
import seaborn as sns

# 検定
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from tqdm import tqdm_notebook as tqdm
# 評価
from sklearn.metrics import roc_auc_score, r2_score

# 前処理
from sklearn.feature_extraction.text import TfidfVectorizer
from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import QuantileTransformer   # 数値列のRankGauss処理用

# モデリング
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

# DataRobot SDK
import datarobot as dr
from datarobot.enums import AUTOPILOT_MODE
from datarobot.enums import DIFFERENCING_METHOD
from datarobot.enums import TIME_UNITS

# 表示量
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 100)

# 乱数シード固定
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(2020)
def capping(series, min_threshold, max_threshold):
    series_filtered = series.copy()
    index_outlier_up = [series_filtered  >= max_threshold]
    index_outlier_low = [series_filtered <= min_threshold]
    series_filtered.iloc[index_outlier_up] = max_threshold
    series_filtered.iloc[index_outlier_low] = min_threshold
    return series_filtered
# コピペで使える。Kaggleでの実験を効率化する小技まとめ - 天色グラフィティ
#   https://amalog.hateblo.jp/entry/kaggle-snippets#DataFrame%E3%81%AE%E3%83%A1%E3%83%A2%E3%83%AA%E3%82%92%E7%AF%80%E7%B4%84%E3%81%99%E3%82%8B

# pandasのDataFrameはintならnp.int64に、floatならnp.float64がデフォルトで使われます。 
# しかし、ある程度データセットが大きくなってくると、DataFrameがメモリを圧迫して学習を思うように進めることができなかったりします。
# そこで、各列の値の範囲を参照し、適切な型に変換します。

import logging

def reduce_mem_usage(df, logger=None, level=logging.DEBUG):
    print_ = print if logger is None else lambda msg: logger.log(level, msg)
    start_mem = df.memory_usage().sum() / 1024**2
    print_('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != 'object' and col_type != 'datetime64[ns]':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)  # feather-format cannot accept float16
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print_('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print_('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df
# データサイズ(small, mid, full)
flg_datasize = 'full'

# CVするか？ → 意味なし
#flg_cv = True

# 特徴量を絞るか？
flg_cut_features = False

# 手動で特徴量を指定する？
feature_list = []
#feature_list = ['sub_grade', 'grade', 'dti', 'loan_amnt/installment', 'revol_bal-tot_cur_bal', 'zip_code', 'open_acc', 'issue_d-earliest_cr_line', 'installment', 'tot_cur_bal', 'revol_util', 'mths_since_last_record', 'annual_inc/installment', 'home_ownership', 'loan_amnt-revol_bal', 'annual_inc-tot_cur_bal', 'inq_last_6mths', 'annual_inc/Real State Growth %_mean', 'mths_since_last_major_derog', 'revol_bal', 'StateName', 'annual_inc', 'loan_amnt-tot_cur_bal', 'loan_amnt', 'earliest_cr_line', 'annual_inc-installment', 'StateCode', 'installment/loan_amnt', 'total_acc', 'addr_state', 'installment-tot_cur_bal', 'mths_since_last_delinq', 'revol_bal-tot_coll_amt', 'installment-loan_amnt', 'annual_inc/State & Local Spending_mean', 'emp_length', 'title', 'delinq_2yrs', 'annual_inc/Population (million)_mean', 'tot_coll_amt', 'Latitude', 'Real State Growth %_mean', 'annual_inc/Gross State Product_mean', 'purpose', 'deputy', 'emp_length_isna', 'ceo', 'emp_title_isna', 'recruiter', 'distribution']

# Topいくつまでの特量に絞るかを指定する
if len(feature_list) > 0:
    feature_imp_cnt = len(feature_list)
else:
    feature_imp_cnt = 100

# テキスト型を TFIDF かけるときの最大カラム数 
TFIDF_max_features = 2000

# plotなどinfomationを表示するか
flg_info = False
# 使用メモリを削減するために各列の型を明示的に指定 → issue_d を 2015年に限定したので使わなくなった
# https://qiita.com/nannoki/items/2a8934de31ad2258439d
# dtypes = {
#     'ID': 'int32',
#     'loan_amnt': 'int32',
#     'installment': 'float32',
#     'grade': 'object',
#     'sub_grade': 'object',
#     'emp_title': 'object',
#     'emp_length': 'object',
#     'home_ownership': 'object',
#     'annual_inc': 'float32',
#     'issue_d': 'object',
#     'purpose': 'object',
#     'title': 'object',
#     'zip_code': 'object',
#     'addr_state': 'object',
#     'dti': 'float32',
#     'delinq_2yrs': 'float32',
#     'earliest_cr_line': 'object',
#     'inq_last_6mths': 'float32',
#     'mths_since_last_delinq': 'float32',
#     'mths_since_last_record': 'float32',
#     'open_acc': 'float32',
#     'pub_rec': 'float32',
#     'revol_bal': 'int32',
#     'revol_util': 'float32',
#     'total_acc': 'float32',
#     'initial_list_status': 'object',
#     'collections_12_mths_ex_med': 'float32',
#     'mths_since_last_major_derog': 'float32',
#     'application_type': 'object',
#     'acc_now_delinq': 'float32',
#     'tot_coll_amt': 'float32',
#     'tot_cur_bal': 'float32',
#     'loan_condition': 'int8'
# }

# dtypes = {
#     'ID': 'int32',
#     'loan_amnt': 'int32',
#     'installment': 'float64',
#     'grade': 'object',
#     'sub_grade': 'object',
#     'emp_title': 'object',
#     'emp_length': 'object',
#     'home_ownership': 'object',
#     'annual_inc': 'float64',
#     'issue_d': 'object',
#     'purpose': 'object',
#     'title': 'object',
#     'zip_code': 'object',
#     'addr_state': 'object',
#     'dti': 'float64',
#     'delinq_2yrs': 'float64',
#     'earliest_cr_line': 'object',
#     'inq_last_6mths': 'float64',
#     'mths_since_last_delinq': 'float64',
#     'mths_since_last_record': 'float64',
#     'open_acc': 'float64',
#     'pub_rec': 'float64',
#     'revol_bal': 'int32',
#     'revol_util': 'float64',
#     'total_acc': 'float64',
#     'initial_list_status': 'object',
#     'collections_12_mths_ex_med': 'float64',
#     'mths_since_last_major_derog': 'float64',
#     'application_type': 'object',
#     'acc_now_delinq': 'float64',
#     'tot_coll_amt': 'float64',
#     'tot_cur_bal': 'float64',
#     'loan_condition': 'int8'
# }


if flg_datasize == 'small':
    df_train = pd.read_csv('../input/homework-for-students3/train.csv', index_col=0, parse_dates=['issue_d', 'earliest_cr_line'], skiprows=lambda x: x%98!=0)
    df_test = pd.read_csv('../input/homework-for-students3/test.csv', index_col=0, parse_dates=['issue_d', 'earliest_cr_line'], skiprows=lambda x: x%98!=0)
    submission = pd.read_csv('../input/homework-for-students3/sample_submission.csv', index_col=0, skiprows=lambda x: x%98!=0)
elif flg_datasize == 'mid':
    df_train = pd.read_csv('../input/homework-for-students3/train.csv', index_col=0, parse_dates=['issue_d', 'earliest_cr_line'], skiprows=lambda x: x%20!=0)
    df_test = pd.read_csv('../input/homework-for-students3/test.csv', index_col=0, parse_dates=['issue_d', 'earliest_cr_line'], skiprows=lambda x: x%20!=0)
    submission = pd.read_csv('../input/homework-for-students3/sample_submission.csv', index_col=0, skiprows=lambda x: x%20!=0)
elif flg_datasize == 'full':
    df_train = pd.read_csv('../input/homework-for-students3/train.csv', index_col=0, parse_dates=['issue_d', 'earliest_cr_line'])
    df_test = pd.read_csv('../input/homework-for-students3/test.csv', index_col=0, parse_dates=['issue_d', 'earliest_cr_line'])
    submission = pd.read_csv('../input/homework-for-students3/sample_submission.csv', index_col=0)

    
print(df_train.shape)
# 2015年のデータだけ残す

df_train = df_train[df_train['issue_d'] >= dt.datetime(2015,1,1)]
df_train['issue_d']
# 使用メモリを削減するために各列の型を明示的に指定
# https://qiita.com/nannoki/items/2a8934de31ad2258439d
# dtypes_df_US_GDP_by_State = {
#     'State': 'object',
#     'State & Local Spending': 'float32',
#     'Gross State Product': 'float32',
#     'Real State Growth %': 'float32',
#     'Population (million)': 'float32',
#     'year': 'int16'
# }

# アメリカの各州のGDP
df_US_GDP_by_State = pd.read_csv('../input/homework-for-students3/US_GDP_by_State.csv')

# アメリカの郵便番号データベース（使ってない）
#df_free_zipcode_database = pd.read_csv('../input/homework-for-students3/free-zipcode-database.csv', index_col=0)

# df_statelatlong = {
#     'State': 'object',
#     'Latitude': 'float32',
#     'Longitude': 'float32',
#     'City': 'object'
# }
# アメリカの各州の緯度経度
df_statelatlong = pd.read_csv('../input/homework-for-students3/statelatlong.csv')
# df_US_GDP_by_State を 'State' で集計
df_US_GDP_by_State = df_US_GDP_by_State.groupby('State').mean()[['State & Local Spending', 'Gross State Product', 'Real State Growth %', 'Population (million)']]

# MultiIndexになってしまっていて使いづらいので解除
col_names = []
for c1 in ['State & Local Spending', 'Gross State Product', 'Real State Growth %', 'Population (million)']:
#    for c2 in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
    for c2 in ['mean']:
        col_names.append(c1 + '_' + c2)

df_US_GDP_by_State.columns = col_names
df_US_GDP_by_State.reset_index(inplace=True)

# 'State' 列が 'All states combined'の行を削除
df_US_GDP_by_State = df_US_GDP_by_State[df_US_GDP_by_State['State'] != 'All states combined']

if flg_info:
    df_US_GDP_by_State
# 各DataFrameの列名を修正
df_US_GDP_by_State.rename(columns={'State': 'StateName'}, inplace=True)
df_statelatlong.rename(columns={'State': 'StateCode', 'City': 'StateName'}, inplace=True)

#print('#### df_US_GDP_by_State')
#print(df_US_GDP_by_State.head(1))
#print('#### df_statelatlong')
#print(df_statelatlong.head(1))
# 結合
df_train = pd.merge(left=df_train, right=pd.merge(df_US_GDP_by_State, df_statelatlong, on='StateName'), how='left', left_on='addr_state', right_on='StateCode')
df_test = pd.merge(left=df_test, right=pd.merge(df_US_GDP_by_State, df_statelatlong, on='StateName'), how='left', left_on='addr_state', right_on='StateCode')

# なんとなくcsvに出力
df_train.to_csv('train_with_geo-datas.csv')

if flg_info:
    df_train
## 全数値列のhistを見てみる
if flg_info:
    num_cols = df_train.select_dtypes(include='number').columns.values.tolist()
    num_cols.remove('loan_condition')
    print(num_cols)
    
    for f in num_cols:
        plt.figure(figsize=[5,5])
        df_train[f].hist(density=True, alpha=0.5, bins=20, color="blue")
        df_test[f].hist(density=True, alpha=0.5, bins=20, color="red")
        plt.xlabel(f)
        plt.ylabel('density')
        plt.show()
        # print(df_train[f].describe(), df_test[f].describe())
# 'dti' の 999の行を確認する
#df_train.query('dti > 100')
# pub_rec の 20以上
#df_train.query('pub_rec > 20')
# revol_util の 150以上
#df_train.query('revol_util > 150')
# acc_now_delinq の 6 以上
#df_train.query('acc_now_delinq > 6')
# tot_coll_amt の 200000 以上
#df_train.query('tot_coll_amt > 200000')
# xとyに分割
Y_train = df_train.loan_condition
X_train = df_train.drop(['loan_condition'], axis=1)
X_test = df_test

# 特徴量エンジニアリング前の、元のカラム名一覧を取っておく（欠損フラグを立てるところで使う）
base_columns = X_train.columns

# 同じユーザが何回も借りてきてないか疑ってみる
# 重複ユーザがいないかチェック
cols_dup_user_check = ['grade', 'sub_grade', 'earliest_cr_line', 'emp_title', 'emp_length', 'zip_code', 'addr_state', 'home_ownership']
df_train[df_train.duplicated(cols_dup_user_check, keep=False)].sort_values(by=cols_dup_user_check, ascending=True)
X_train['user_id'] = \
    X_train['grade'].fillna('NaN') + \
    X_train['sub_grade'].fillna('NaN') + \
    X_train['earliest_cr_line'].fillna('NaN').apply(lambda x: x.strftime('%Y/%m/%d')) + \
    X_train['emp_title'].fillna('NaN') + \
    X_train['emp_length'].fillna('NaN') + \
    X_train['zip_code'].fillna('NaN') + \
    X_train['addr_state'].fillna('NaN') + \
    X_train['home_ownership'].fillna('NaN')

X_test['user_id'] = \
    X_test['grade'].fillna('NaN') + \
    X_test['sub_grade'].fillna('NaN') + \
    X_test['earliest_cr_line'].fillna('NaN').apply(lambda x: x.strftime('%Y/%m/%d')) + \
    X_test['emp_title'].fillna('NaN') + \
    X_test['emp_length'].fillna('NaN') + \
    X_test['zip_code'].fillna('NaN') + \
    X_test['addr_state'].fillna('NaN') + \
    X_test['home_ownership'].fillna('NaN')

if flg_info:
    X_train
# user_id列の顧客IDを単位として分割することにする
user_id = X_train['user_id']
unique_user_ids = user_id.unique()

# クロスバリデーションのfoldを定義
#skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)
kf = KFold(n_splits=5, random_state=71, shuffle=True)

# パーティション用のuser_id列を削除
X_train.drop(['user_id'], axis=1, inplace=True)
X_test.drop(['user_id'], axis=1, inplace=True)

user_id
# メモリがもったいないので使わない df を削除
del df_train, df_test, df_US_GDP_by_State, df_statelatlong
gc.collect()
# 元のカラムのうち、欠損値を含む列に対して欠損フラグ列を追加する
for col in base_columns:
    if X_train[col].isnull().any():
        X_train[col + '_isna'] = X_train[col].isnull().replace({True : 1, False : 0})
        X_test[col + '_isna'] = X_test[col].isnull().replace({True : 1, False : 0})

X_train[['emp_title', 'emp_title_isna']]
# 'issue_d' - 'earliest_cr_line' 間の日数を新たなカラムとして追加
X_train['issue_d-earliest_cr_line'] = (X_train['issue_d'] - X_train['earliest_cr_line']).apply(lambda x: x.days)
X_test['issue_d-earliest_cr_line'] = (X_test['issue_d'] - X_test['earliest_cr_line']).apply(lambda x: x.days)

X_train['issue_d-earliest_cr_line']
# 列を削除
X_train.drop(['issue_d'], axis=1, inplace=True)
X_test.drop(['issue_d'], axis=1, inplace=True)

# 現在はTarget Encodingしてしまっている
# X_train.drop(['earliest_cr_line'], axis=1, inplace=True)
# X_test.drop(['earliest_cr_line'], axis=1, inplace=True)
colormap = plt.cm.RdBu
plt.figure(figsize=(30,30))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(X_train.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
# array(['10+ years', '2 years', '1 year', '3 years', nan, '7 years',
#       '5 years', '6 years', '8 years', '< 1 year', '4 years', '9 years'],
#      dtype=object)

dict = {np.nan:0, '< 1 year':0.5, '1 year':1, '2 years':2, '3 years':3, '4 years':4, '5 years':5, '6 years':6, '7 years':7, '8 years':8, '9 years':9, '10+ years':10}
X_train['emp_length'] = X_train['emp_length'].map(dict)
X_test['emp_length'] = X_test['emp_length'].map(dict)

X_train[X_train.emp_length < 0.5]
# 金額系の列を使って特徴量生成
#num_cols = ['annual_inc', 'installment', 'loan_amnt', 'revol_bal', 'tot_coll_amt', 'tot_cur_bal']

# 全数値列に対して網羅的に特徴量生成
num_cols_for_auto_create = X_train.select_dtypes(include='number').columns.values.tolist()

print(num_cols_for_auto_create)

for i in range(0, len(num_cols_for_auto_create)-1):
    for j in (i+1, len(num_cols_for_auto_create)-1):
        c1 = num_cols_for_auto_create[i]
        c2 = num_cols_for_auto_create[j]
        X_train[c1 + 'X' + c2] = X_train[c1] * X_train[c2]
        X_test[c1 + 'X' + c2]  = X_test[c1]  * X_test[c2]

        # c2に0が含まれてないなら割り算カラムも追加
        if (X_train[c2] == 0).sum() == 0 & (X_test[c2] == 0).sum() == 0:
            X_train[c1 + '/' + c2] = X_train[c1] / X_train[c2]
            X_test[c1 + '/' + c2]  = X_test[c1]  / X_test[c2]

# 無限大をNanに変換（後で欠損値として補完される）
X_train=X_train.replace([np.inf, -np.inf], np.nan)
X_test=X_test.replace([np.inf, -np.inf], np.nan)

print(X_train.columns)
# ローン支払い回数
X_train['loan_amnt/installment'] = X_train['loan_amnt'] / X_train['installment']
X_test['loan_amnt/installment']  = X_test['loan_amnt']  / X_test['installment']

X_train[['installment', 'loan_amnt', 'loan_amnt/installment']]
# 年収と州の各値との割合
X_train['annual_inc/State & Local Spending_mean'] = X_train['annual_inc'] / X_train['State & Local Spending_mean']
X_train['annual_inc/Gross State Product_mean']    = X_train['annual_inc'] / X_train['Gross State Product_mean']
X_train['annual_inc/Real State Growth %_mean']    = X_train['annual_inc'] / X_train['Real State Growth %_mean']
X_train['annual_inc/Population (million)_mean']   = X_train['annual_inc'] / X_train['Population (million)_mean']

X_test['annual_inc/State & Local Spending_mean'] = X_test['annual_inc'] / X_test['State & Local Spending_mean']
X_test['annual_inc/Gross State Product_mean']    = X_test['annual_inc'] / X_test['Gross State Product_mean']
X_test['annual_inc/Real State Growth %_mean']    = X_test['annual_inc'] / X_test['Real State Growth %_mean']
X_test['annual_inc/Population (million)_mean']   = X_test['annual_inc'] / X_test['Population (million)_mean']
# 各列の変換方針
# loan_amnt                   Standard
# installment                 Standard
# annual_inc                  log
# dti                         log
# delinq_2yrs                 log
# inq_last_6mths              log
# mths_since_last_delinq      log
# mths_since_last_record      Standard
# open_acc                    log
# pub_rec                     log
# revol_bal                   log
# revol_util                  log
# total_acc                   log
# collections_12_mths_ex_med  log
# mths_since_last_major_derog log
# acc_now_delinq              log
# tot_coll_amt                log
# tot_cur_bal                 log
# State & Local Spending_mean log
# Gross State Product_mean    log
# Real State Growth %_mean    Standard
# Population (million)_mean   log
# Latitude                    Standard
# Longitude                   Standard
# issue_d-earliest_cr_line    Standard
# # StandardScalerで標準化
# num_cols_Standard = ['loan_amnt', 'installment', 'mths_since_last_record', 'Real State Growth %_mean', 'Latitude', 'Longitude', 'issue_d-earliest_cr_line']
# 
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# 
# for c in num_cols_Standard:
#     # 学習データに基づいて複数カラムの標準化を定義
#     scaler.fit(X_train.iloc[:, X_train.columns == c])
#     # 変換後データの列を作成
#     X_train[c + '_STD'] = scaler.transform(X_train.iloc[:, X_train.columns == c])
#     X_test[c + '_STD'] = scaler.transform(X_test.iloc[:, X_test.columns == c])
# 
#     plt.figure(figsize=[10,5])
#     plt.subplot(1,2,1)
#     X_train[c].hist(density=True, alpha=0.5, bins=20, color="blue")
#     X_test[c].hist(density=True, alpha=0.5, bins=20, color="red")
#     plt.xlabel(c)
#     plt.ylabel('density')
#     plt.subplot(1,2,2)
#     X_train[c + '_STD'].hist(density=True, alpha=0.5, bins=20, color="blue")
#     X_test[c + '_STD'].hist(density=True, alpha=0.5, bins=20, color="red")
#     plt.xlabel(c + '_STD')
#     plt.ylabel('density')
#     plt.show()
# 
#     # 変換後データで置換して不要な列を削除
#     X_train[c] = X_train[c + '_STD']
#     X_test[c] = X_test[c + '_STD']
#     X_train.drop([c + '_STD'], axis=1, inplace=True)
#     X_test.drop([c + '_STD'], axis=1, inplace=True)
# # 対数変換
# num_cols_Log = ['annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'State & Local Spending_mean', 'Gross State Product_mean', 'Population (million)_mean']
# 
# for c in num_cols_Log:
#     X_train[c + '_LOG'] = X_train[c].apply(np.log1p)
#     X_test[c + '_LOG'] = X_test[c].apply(np.log1p)
#     
#     plt.figure(figsize=[10,5])
#     plt.subplot(1,2,1)
#     X_train[c].hist(density=True, alpha=0.5, bins=20, color="blue")
#     X_test[c].hist(density=True, alpha=0.5, bins=20, color="red")
#     plt.xlabel(c)
#     plt.ylabel('density')
#     plt.subplot(1,2,2)
#     X_train[c + '_LOG'].hist(density=True, alpha=0.5, bins=20, color="blue")
#     X_test[c + '_LOG'].hist(density=True, alpha=0.5, bins=20, color="red")
#     plt.xlabel(c + '_LOG')
#     plt.ylabel('density')
#     plt.show()
# 
#     # 変換後データで置換して不要な列を削除
#     X_train[c] = X_train[c + '_LOG']
#     X_test[c] = X_test[c + '_LOG']
#     X_train.drop([c + '_LOG'], axis=1, inplace=True)
#     X_test.drop([c + '_LOG'], axis=1, inplace=True)
num_cols = X_train.select_dtypes(include='number').columns.values.tolist()

from sklearn.preprocessing import QuantileTransformer

# 学習データに基づいて複数列のRankGaussによる変換を定義
scaler = QuantileTransformer(n_quantiles=100, random_state=71, output_distribution='normal')

for c in num_cols:
    # 学習データに基づいてRankGaussによる変換を定義
    scaler.fit(X_train.iloc[:, X_train.columns == c])
    # 変換後データの列を作成
    X_train[c + '_RANKG'] = scaler.transform(X_train.iloc[:, X_train.columns == c])
    X_test[c + '_RANKG'] = scaler.transform(X_test.iloc[:, X_test.columns == c])
    
    if flg_info:
        plt.figure(figsize=[10,5])
        plt.subplot(1,2,1)
        X_train[c].hist(density=True, alpha=0.5, bins=20, color="blue")
        X_test[c].hist(density=True, alpha=0.5, bins=20, color="red")
        plt.xlabel(c)
        plt.ylabel('density')
        plt.subplot(1,2,2)
        X_train[c + '_RANKG'].hist(density=True, alpha=0.5, bins=20, color="blue")
        X_test[c + '_RANKG'].hist(density=True, alpha=0.5, bins=20, color="red")
        plt.xlabel(c + '_RANKG')
        plt.ylabel('density')
        plt.show()

    # 変換後データで置換して不要な列を削除
    X_train[c] = X_train[c + '_RANKG']
    X_test[c] = X_test[c + '_RANKG']
    X_train.drop([c + '_RANKG'], axis=1, inplace=True)
    X_test.drop([c + '_RANKG'], axis=1, inplace=True)
# 各列ごとの戦略          ユニーク数
#  column名             train     test
#  grade                    7        7      辞書順でLabel Encoding
#  sub_grade               35       35      辞書順でLabel Encoding 
#  emp_title            23506    10451      テキスト 
#  emp_length              11       11      数値型に変換（< 1 year = 0.5, n/a = 0.0, 10+ years = 10)  → 済み
#  home_ownership           5        4      Train で Target Encoding
#  issue_d                103       12      日付 
#  purpose                 14       13      Train で Target Encoding
#  title                 4832       12      purposeで代替えできるので捨てる
#  zip_code               857      839      Train で Target Encoding
#  addr_state              50       50      Train で Target Encoding
#  earliest_cr_line       586      567      日付 
#  initial_list_status      2        2      One-Hot Encoding
#  application_type         2        2      Train で Target Encoding
#
# これ以外のobject型もすべて Target Encoding
# 列を削除
X_train.drop(['title'], axis=1, inplace=True)
X_test.drop(['title'], axis=1, inplace=True)

# grade と sub_grade を Label Encoding
cats_label = ['grade', 'sub_grade']

for c in cats_label:
    le = LabelEncoder()
    le.fit(X_train[c])
    X_train[c] = le.transform(X_train[c])
    X_test[c] = le.transform(X_test[c])

X_train[cats_label]
# initial_list_status を One-hot Encoding
cats_onehot = ['initial_list_status']

X_train = pd.get_dummies(X_train, columns=cats_onehot)
X_test = pd.get_dummies(X_test, columns=cats_onehot)

X_train
cats_target = []
# Target Encoding
#cats_target = ['home_ownership', 'purpose', 'zip_code', 'addr_state', 'application_type', 'title']
#cats_target = ['home_ownership', 'purpose', 'zip_code', 'addr_state', 'application_type', 'title', 'issue_d', 'earliest_cr_line']

cats_target = X_train.select_dtypes(include=[object, 'datetime']).columns.values.tolist()
# for col in X_train.columns:
#     if X_train[col].dtype == 'object':
#         cats_target.append(col)

# 'emp_title' は テキスト型なので除外
cats_target.remove('emp_title')
print(cats_target)

# 変数をループしてTarget Encoding
for c in cats_target:
    # c列と目的変数のみの DataFrame を作成
    data_tmp = pd.DataFrame({c: X_train[c], 'target': Y_train})
    
    # ======= test に対する Target Encoding ================
    # train全体でカテゴリ値ごとの目的変数の平均値を集計
    target_mean = data_tmp.groupby(c)['target'].mean()
    # testの各カテゴリ値を平均値で置換
    X_test[c] = X_test[c].map(target_mean)
    
    # ======= train に対する Target Encoding ================
    # train の変換後の値を格納する配列を準備
    tmp = np.repeat(np.nan, X_train.shape[0])
    
    # KFoldクラスを用いて、顧客ID単位で分割する
    for tr_group_idx, va_group_idx in kf.split(unique_user_ids):
        # 顧客IDをtrain/valid（学習に使うデータ、バリデーションデータ）に分割する
        tr_groups = unique_user_ids[tr_group_idx]  # Training
        va_groups = unique_user_ids[va_group_idx]  # Validation
        
        # 各顧客のレコードの顧客IDがtrain/validのどちらに属しているかによって分割する
        is_tr = user_id.isin(tr_groups)
        is_va = user_id.isin(va_groups)

        # out-of-foldでカテゴリ値ごとの目的変数の平均値を集計
        target_mean = data_tmp[is_tr].groupby(c)['target'].mean()
        # 置換後の値を一時配列に格納しておく
        tmp[is_va] = X_train[c][is_va].map(target_mean)
    
    # trainの各カテゴリ値を平均値で置換
    X_train[c] = tmp
    
X_train[cats_target]
# 中央値で埋める
for col in X_train.columns:
    if X_train[col].dtype != 'object':
        median = X_train[col].median()
        X_train[col].fillna(median, inplace=True)
        X_test[col].fillna(median, inplace=True)

X_train
%%time
# X_train と X_test DataFrameの型を最適化
X_train = reduce_mem_usage(X_train)
X_test = reduce_mem_usage(X_test)
# Python メモリ8GBでkaggleを闘うときに知っておきたいこと - Qiita
#   https://qiita.com/hmdhmd/items/2efb620abda7b20c6711

#メモリを食っている変数を大きいほうから表示

print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val))/1024/1024 for val in dir()]],
                   index=['name','size_MB']).T.sort_values('size_MB', ascending=False).reset_index(drop=True).query('size_MB > 100')
     )
# テキストとそれ以外に分割
TXT_train = X_train.emp_title.copy()
TXT_test = X_test.emp_title.copy()

X_train.drop(['emp_title'], axis=1, inplace=True)
X_test.drop(['emp_title'], axis=1, inplace=True)

# 小文字にそろえる
TXT_train = TXT_train.str.lower()
TXT_test = TXT_test.str.lower()

# テキストをTFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
tdidf = TfidfVectorizer(min_df=3, norm='l2', ngram_range=(1, 3), stop_words='english', max_features=TFIDF_max_features)

tdidf.fit(TXT_train.fillna('#'))
TXT_train = tdidf.transform(TXT_train.fillna('#'))
TXT_test = tdidf.transform(TXT_test.fillna('#'))
print('Vocabulary size: {}'.format(len(tdidf.vocabulary_)))
# TFIDFにかけたテキストをhstack
# X_train = sp.sparse.hstack([X_train.values, TXT_train])
# X_test = sp.sparse.hstack([X_test.values, TXT_test])

# TFIDFにかけた sparse matrix を、indexを元に戻しつつ DataFrame に変換
TXT_train_ = pd.DataFrame(TXT_train.toarray(), columns=tdidf.get_feature_names(), dtype=np.int8)
TXT_train_.index = X_train.index
TXT_test_ = pd.DataFrame(TXT_test.toarray(), columns=tdidf.get_feature_names(), dtype=np.int8)
TXT_test_.index = X_test.index

if flg_info:
    print(TXT_train_.dtypes)
    print(TXT_train_.info())

# 事前に特徴量リストが指定されていた場合は
# 使わないカラムを事前に刈って消費メモリを削減する

if len(feature_list) > 0:
    print('##### before ##########################')
    print(X_train.info())

    for col in X_train.columns:
        if col not in feature_list:
            X_train.drop([col], axis=1, inplace=True)
            X_test.drop([col], axis=1, inplace=True)
    
    print('##### after ##########################')
    print(X_train.info())
    
    print('##### before ##########################')
    print(TXT_train_.info())
    
    # 使わないカラムを事前に刈って消費メモリを削減する
    for col in TXT_train_.columns:
        if col not in feature_list:
            TXT_train_.drop([col], axis=1, inplace=True)
            TXT_test_.drop([col], axis=1, inplace=True)

    print('##### after ##########################')
    print(TXT_train_.info())

# 元のDataFrameの右横に結合しなおす
X_train = pd.concat([X_train, TXT_train_], axis=1)
X_test = pd.concat([X_test, TXT_test_], axis=1)

del TXT_train_, TXT_test_, TXT_train, TXT_test
gc.collect()

if flg_info:
    print(X_train.dtypes)
    print(X_train.info())

X_train
# 'grade' や 'title' など 「もともと合ったColumn名」 と 「TFIDFで生成されたColumn名」がたまたま重複する可能性がある。
# その時、「TFIDFで生成されたColumn」のほう（右に後から連結されたほう）を削除する
# 参考 : https://qiita.com/mynkit/items/6e8ddfa1c73363f6117e

if len(X_train.columns) != X_train.columns.nunique():
    print(X_train.columns[X_train.columns.duplicated(keep=False)])
    X_train = X_train.loc[:,~X_train.columns.duplicated()]
    X_test = X_test.loc[:,~X_test.columns.duplicated()]

#pd.concat([X_train, Y_train], axis=1).to_csv('train_after_feature_engineering.csv')
#X_test.to_csv('test_after_feature_engineering.csv')
# モデリングに使うライブラリ
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
# 事前に特徴量リストが指定されていた場合は
if len(feature_list) > 0:
    # その特徴量リストに絞る
    X_train = X_train[feature_list]
    X_test = X_test[feature_list]

# from hyperopt import fmin, tpe, hp, rand, Trials
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import roc_auc_score
# 
# from lightgbm import LGBMClassifier
# def objective(space):
#     scores = []
# 
#     # KFoldクラスを用いて、顧客ID単位で分割する（グループパーティション）
#     for i, (tr_group_idx, va_group_idx) in enumerate(kf.split(unique_user_ids)):
#         # 顧客IDをtrain/valid（学習に使うデータ、バリデーションデータ）に分割する
#         tr_groups = unique_user_ids[tr_group_idx]  # Training
#         va_groups = unique_user_ids[va_group_idx]  # Validation
# 
#         # 各顧客のレコードの顧客IDがtrain/validのどちらに属しているかによって分割する
#         is_tr = user_id.isin(tr_groups)
#         is_va = user_id.isin(va_groups)
#         X_train_, Y_train_ = X_train[is_tr], Y_train[is_tr]  # Training
#         X_valid_, Y_valid_ = X_train[is_va], Y_train[is_va]  # Validation
# 
#         # モデリング
#         clf = LGBMClassifier(n_estimators=9999, **space)
#         clf.fit(X_train_, Y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_valid_, Y_valid_)], verbose=False)
# 
#         # バリデーションデータを予測してAUCを計算し出力
#         Y_pred = clf.predict_proba(X_valid_)[:,1]
#         score = roc_auc_score(Y_valid_, Y_pred)
#         scores.append(score)
# 
#     scores = np.array(scores)
#     print(scores.mean())
#     
#     return -scores.mean()
# # パラメータ探索範囲を定義
# space ={
#         'max_depth': hp.choice('max_depth', np.arange(10, 30, dtype=int)),
#         'subsample': hp.uniform ('subsample', 0.8, 1),
#         'learning_rate' : hp.quniform('learning_rate', 0.025, 0.5, 0.025),
#         'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05)
#     }
# 
# trials = Trials()
# 
# best = fmin(fn=objective,
#               space=space, 
#               algo=tpe.suggest,
#               max_evals=20, 
#               trials=trials, 
#               rstate=np.random.RandomState(71) 
#              )
# 
# LGBMClassifier(n_estimators=9999, **best)
%%time
# TensorFlow/kerasの最新versionで疎行列周りでエラーが出るのでバージョン指定しています。
# 現状では未解決のようで、PyTorch使ったほうがいいかも。
!pip uninstall tensorflow -y
!pip install tensorflow==1.11.0

!pip uninstall keras -y
!pip install keras==2.2.4
from keras.layers import Input, Dense ,Dropout, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.callbacks import EarlyStopping

# シンプルなMLP

def create_model(input_dim):
    inp = Input(shape=(input_dim,), sparse=True) # 疎行列を入れる
#    inp = Input(shape=(input_dim,), sparse=False)
#    inp = Input(shape=(input_dim,))
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

%%time
# CVしてスコアを見てみる。
#df_scores = pd.DataFrame(index=[], columns=['Light GBM', 'Neural Network', 'k-Nearest Neighbors', 'Random Forest', 'CV_mean'])
#df_scores = pd.DataFrame(index=[], columns=['Light GBM', 'Neural Network', 'Random Forest', 'CV_mean'])
#df_scores = pd.DataFrame(index=[], columns=['Light GBM', 'Neural Network', 'CV_mean'])
#df_scores = pd.DataFrame(index=[], columns=['Light GBM', 'Neural Network', 'GradientBoost', 'XGBoost', 'CV_mean'])
df_scores = pd.DataFrame(index=[], columns=['Light GBM', 'Neural Network', 'GradientBoost', 'CV_mean'])
best_iterations = []
best_epochs = []
best_knc_n_neighbors = []
#df_feature_imp = pd.DataFrame()

Y_pred = pd.DataFrame()

#plt.figure(figsize=(25.0, 20.0))

# KFoldクラスを用いて、顧客ID単位で分割する（グループパーティション）
for i, (tr_group_idx, va_group_idx) in enumerate(kf.split(unique_user_ids)):
    # 顧客IDをtrain/valid（学習に使うデータ、バリデーションデータ）に分割する
    tr_groups = unique_user_ids[tr_group_idx]  # Training
    va_groups = unique_user_ids[va_group_idx]  # Validation

    # 各顧客のレコードの顧客IDがtrain/validのどちらに属しているかによって分割する
    is_tr = user_id.isin(tr_groups)
    is_va = user_id.isin(va_groups)
#    X_train_, Y_train_ = X_train[is_tr], Y_train[is_tr]  # Training
#    X_valid_, Y_valid_ = X_train[is_va], Y_train[is_va]  # Validation
    X_train_, Y_train_ = sps.csr_matrix(X_train[is_tr], dtype=np.float32), Y_train[is_tr]  # Training
    X_valid_, Y_valid_ = sps.csr_matrix(X_train[is_va], dtype=np.float32), Y_train[is_va]  # Validation

    # === モデリング =============================
    # ##### Light GBM ###############
    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,
                            importance_type='split', learning_rate=0.05, max_depth=-1,
                            min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                            n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,
                            random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,
                            subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
    clf.fit(X_train_, Y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_valid_, Y_valid_)], verbose=False)
    best_iterations.append(clf.best_iteration_)
    print('CV#' + str(i) + ' : Light GBM - Best Iterations = ' + str(clf.best_iteration_))

    # ##### Neural Network #########
    model = create_model(X_train_.shape[1])
    es = EarlyStopping(monitor='val_loss', patience=0, verbose=0)
    history = model.fit(X_train_, Y_train_, batch_size=32, epochs=999, validation_data=(X_valid_, Y_valid_), callbacks=[es], verbose=0)
    best_epochs.append(len(history.history['val_loss']))
    print('CV#' + str(i) + ' : Neural Network - Best epochs = ' + str(len(history.history['val_loss'])))
    
    # ##### GradientBoost ##########
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train_, Y_train_)
    print('CV#' + str(i) + ' : GradientBoost')

    # ##### XGBoost ##########
#    xgb_params = {
#        'objective': 'binary:logistic',
#        'eval_metric': 'auc',
#    }
#    bst = xgb.train(xgb_params,
#                    xgb.DMatrix(X_train[is_tr], label=Y_train_),
#                    num_boost_round=100,  # 学習ラウンド数は適当
#                    )
#    print('CV#' + str(i) + ' : XGBoost')
    
#     # ##### k-Nearest Neighbors ####
#     # n_neighborsを探索する
#     k_scores = pd.Series()
#     for k in range(51, 101, 5):
#         print(str(k) + ', ', end="")
#         knc = KNeighborsClassifier(n_neighbors=k)
#         knc.fit(X_train_, Y_train_)
#         k_scores[str(k)] = roc_auc_score(Y_valid_, knc.predict_proba(X_valid_)[:,1])
# #    #print(k_scores)
# #    #print(k_scores.idxmax())
#     # 探索したn_neighborsで再度作成
#     knc = KNeighborsClassifier(n_neighbors=int(k_scores.idxmax()))
# #    knc = KNeighborsClassifier(n_neighbors=100)
#     knc.fit(X_train_, Y_train_)
#     best_knc_n_neighbors.append(int(k_scores.idxmax()))
# #    best_knc_n_neighbors.append(100)
#     print('\nCV#' + str(i) + ' : k-Nearest Neighbors - Best n_neighbors = ' + k_scores.idxmax())
# #    print('CV#' + str(i) + ' : k-Nearest Neighbors - Best n_neighbors = ' + str(100))
    
#     # ##### Random Forest ##########
#     rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=5, max_features=3, max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=-1,
#             oob_score=True, random_state=71, verbose=0, warm_start=False)
#     rfc.fit(X_train_, Y_train_)
#     print('CV#' + str(i) + ' : RandomForest')

    # バリデーションデータを予測してAUCを計算し出力
    Y_pred_ = pd.DataFrame()
    Y_pred_['Light GBM'] = clf.predict_proba(X_valid_)[:,1]
    Y_pred_['Neural Network'] = model.predict(X_valid_)
    Y_pred_['GradientBoost'] = gbc.predict_proba(X_valid_)[:,1]
#    Y_pred_['XGBoost'] = bst.predict(xgb.DMatrix(X_train[is_va]))
#    Y_pred_['k-Nearest Neighbors'] = knc.predict_proba(X_valid_)[:,1]
#    Y_pred_['Random Forest'] = rfc.predict_proba(X_valid_)[:,1]
    df_scores.loc['CV#' + str(i)] = [
        roc_auc_score(Y_valid_, Y_pred_['Light GBM']), 
        roc_auc_score(Y_valid_, Y_pred_['Neural Network']), 
        roc_auc_score(Y_valid_, Y_pred_['GradientBoost']), 
#        roc_auc_score(Y_valid_, Y_pred_['XGBoost']), 
#        roc_auc_score(Y_valid_, Y_pred_['k-Nearest Neighbors']), 
#        roc_auc_score(Y_valid_, Y_pred_['Random Forest']), 
        roc_auc_score(Y_valid_, Y_pred_.mean(axis=1))
    ]
    #print(df_scores)
    
    # CV Averaging アンサンブル
    Y_pred['Light GBM CV#' + str(i)] = clf.predict_proba(X_test, num_iteration=clf.best_iteration_)[:,1]
    Y_pred['Neural Network CV#' + str(i)] = model.predict(sps.csr_matrix(X_test))
    Y_pred['GradientBoost CV#' + str(i)] = gbc.predict_proba(sps.csr_matrix(X_test))[:,1]
#    Y_pred['XGBoost CV#' + str(i)] = bst.predict(xgb.DMatrix(X_test))
#    Y_pred['k-Nearest Neighbors CV#' + str(i)] = knc.predict_proba(X_test)[:,1]
#    Y_pred['Random Forest CV#' + str(i)] = rfc.predict_proba(X_test)[:,1]

    # feature importanceを保存
    #print("feature_importances_ counts : " + str(len(clf.feature_importances_)))
    #print("X_train_.columns     counts : " + str(len(X_train_.columns)))
#    feat_imp = pd.Series(clf.feature_importances_, index=X_train.columns, name="CV#" + str(i)).sort_values(ascending=False)
    #print("feat_imp index       counts : " + str(feat_imp.count()))
    #print("feat_imp idx unique  counts : " + str(feat_imp.index.nunique()))
    #print(feat_imp.shape)
#    df_feature_imp =  pd.concat([df_feature_imp, feat_imp.to_frame()], axis=1, sort=True)

    # feature importance Top を描画
#    plt.subplot(1,5,i+1)
#    sns.set_palette("husl")
#    sns.barplot(feat_imp.head(feature_imp_cnt).values, feat_imp.head(feature_imp_cnt).index)
#    plt.title('Top' + str(feature_imp_cnt) + ' Feature Importances in CV#' + str(i))
#    plt.xlabel('Feature Importance Score')
#    #plt.yticks(rotation=60)
    
    del X_train_, Y_train_, X_valid_, Y_valid_, Y_pred_
    gc.collect()

df_scores.loc['algo_mean'] = df_scores.mean(axis=0)
print(df_scores)
print('===================================')
print(df_scores['CV_mean'].values)
#print(np.mean(df_scores['CV_mean']))
if df_scores.at['algo_mean', 'CV_mean'] < 0.69783:
    print('Random_Baseline : 不可')
elif 0.69783 <= df_scores.at['algo_mean', 'CV_mean'] < 0.69889:
    print('Simple_Baseline : 可')
elif 0.69889 <= df_scores.at['algo_mean', 'CV_mean'] < 0.70169:
    print('Median_Baseline : 良')
elif 0.70169 <= df_scores.at['algo_mean', 'CV_mean'] < 0.70524:
    print('GradeA_Baseline : 優')
elif 0.70524 <= df_scores.at['algo_mean', 'CV_mean']:
    print('過去最高を更新 : 優')

# feature importance の高い順に Top "feature_imp_cnt"個 を出力
#df_feature_imp['mean'] = df_feature_imp.mean(axis=1)
#df_feature_imp['std'] = df_feature_imp.std(axis=1)
#feature_list = df_feature_imp.sort_values(by='mean', ascending=False).head(feature_imp_cnt).index
#print(feature_list.to_list())

# feature importance を表示
#plt.show()
#print(df_feature_imp[['mean', 'std']].sort_values(by='mean', ascending=False).head(feature_imp_cnt))
if flg_cut_features:
    # CVで上位だった特徴量、もしくは事前に指定された特徴量に絞る
    X_train = X_train[feature_list]
    X_test = X_test[feature_list]

    X_train
%%time
# Sparse Matrixに変換
X_train_columns = X_train.columns
X_train = sps.csr_matrix(X_train, dtype=np.float32)
X_test = sps.csr_matrix(X_test, dtype=np.float32)

gc.collect()
#メモリを食っている変数を大きいほうから表示
print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val))/1024/1024 for val in dir()]],
                   index=['name','size_MB']).T.sort_values('size_MB', ascending=False).reset_index(drop=True).query('size_MB > 100')
     )
%%time
# Light GBM
#clf = LGBMClassifier(**best)
clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,
                                importance_type='split', learning_rate=0.05, max_depth=-1,
                                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                                n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,
                                random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,
                                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
clf.fit(X_train, Y_train, eval_metric='auc', verbose=False)
%%time
# Neural Network
model = create_model(X_train.shape[1])
model.fit(X_train, Y_train, batch_size=32, epochs=np.max(best_epochs), verbose=0)
#model.fit(X_train, Y_train, batch_size=32, epochs=100, verbose=0)
%%time
# GradientBoost
gbc = GradientBoostingClassifier()
gbc.fit(X_train, Y_train)
#%%time
#xgb_params = {
#    'objective': 'binary:logistic',
#    'eval_metric': 'auc',
#}
#bst = xgb.train(xgb_params,
#                xgb.DMatrix(X_train, label=Y_train),
#                num_boost_round=100,  # 学習ラウンド数は適当
#                )
# %%time
# # k-Nearest Neighbors
# #knc = KNeighborsClassifier(n_neighbors=int(np.mean(best_knc_n_neighbors)))
# knc = KNeighborsClassifier(n_neighbors=80)
# knc.fit(X_train, Y_train)
# %%time
# # Random Forest
# rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#         max_depth=5, max_features=3, max_leaf_nodes=None,
#         min_impurity_decrease=0.0, min_impurity_split=None,
#         min_samples_leaf=1, min_samples_split=2,
#         min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=-1,
#         oob_score=True, random_state=71, verbose=0, warm_start=False)
# rfc.fit(X_train, Y_train)
%%time
# 全学習したモデルでtestに対して予測する
Y_pred['Light GBM'] = clf.predict_proba(X_test, num_iteration=np.max(best_iterations))[:,1]
#Y_pred['Light GBM'] = clf.predict_proba(X_test, num_iteration=100)[:,1]
#Y_pred['Light GBM'] = clf.predict_proba(X_test)[:,1]
%%time
Y_pred['Neural Network'] = model.predict(X_test)
%%time
Y_pred['GradientBoost'] = gbc.predict_proba(X_test)[:,1]
#%%time
#Y_pred['XGBoost'] = bst.predict(xgb.DMatrix(X_test))
# %%time
# Y_pred['k-Nearest Neighbors'] = knc.predict_proba(X_test)[:,1]
# %%time
# Y_pred['Random Forest'] = rfc.predict_proba(X_test)[:,1]
Y_pred
submission.loan_condition = Y_pred.mean(axis=1).values
submission.to_csv('submission.csv')
submission
# # feature importanceを出力する
# feat_imp = pd.Series(clf.feature_importances_, X_train_columns).sort_values(ascending=False)
# 
# plt.figure(figsize=(20.0, 20.0))
# sns.set_palette("husl")
# sns.barplot(feat_imp.head(feature_imp_cnt).values, feat_imp.head(feature_imp_cnt).index)
# plt.title('Top' + str(feature_imp_cnt) + ' Feature Importances')
# plt.xlabel('Feature Importance Score')
# #plt.yticks(rotation=20)
# plt.show()
# feat_imp.head(feature_imp_cnt)