import gc

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

import seaborn as sns



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



from sklearn.ensemble import GradientBoostingClassifier



import lightgbm as lgb

from lightgbm import LGBMClassifier



from catboost import CatBoostClassifier

from catboost import Pool



pd.set_option('display.max_columns', 100)
df_train = pd.read_csv("../input/homework-for-students2/train.csv", index_col=0, parse_dates=['issue_d', 'earliest_cr_line'])

df_test = pd.read_csv("../input/homework-for-students2/test.csv", index_col=0, parse_dates=['issue_d', 'earliest_cr_line'])
# df_train.drop(['application_type', 'collections_12_mths_ex_med', 'acc_now_delinq'], axis=1, inplace=True)

# df_test.drop(['application_type', 'collections_12_mths_ex_med', 'acc_now_delinq'], axis=1, inplace=True)
# df_train.drop(['initial_list_status', 'delinq_2yrs', 'pub_rec'], axis=1, inplace=True)

# df_test.drop(['initial_list_status', 'delinq_2yrs', 'pub_rec'], axis=1, inplace=True)
grade_rank = pd.DataFrame({'grade':['A', 'B', 'C', 'D', 'E', 'F', 'G'],

                           'grade_rank': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})
df_train = df_train.merge(grade_rank, on=['grade'], how='left')

df_train.drop('grade', axis=1, inplace=True)



df_test = df_test.merge(grade_rank, on=['grade'], how='left')

df_test.drop('grade', axis=1, inplace=True)
sub_grade_rank = pd.DataFrame({'sub_grade':['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', 'G5'],

                               'sub_grade_rank': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0]})
df_train = df_train.merge(sub_grade_rank, on=['sub_grade'], how='left')

df_train.drop('sub_grade', axis=1, inplace=True)



df_test = df_test.merge(sub_grade_rank, on=['sub_grade'], how='left')

df_test.drop('sub_grade', axis=1, inplace=True)
# grade_rankとsub_grade_rankでウェイトとる

df_train['grade_weight'] = df_train['grade_rank']/7 * df_train['sub_grade_rank']/35

df_test['grade_weight'] = df_test['grade_rank']/7 * df_test['sub_grade_rank']/35



# df_train['grade_weight'] = df_train['grade_rank']/7 + df_train['sub_grade_rank']/35

# df_test['grade_weight'] = df_test['grade_rank']/7 + df_test['sub_grade_rank']/35
df_train['passed_days'] = (df_train['issue_d'] - df_train['earliest_cr_line']).dt.days

df_test['passed_days'] = (df_test['issue_d'] - df_test['earliest_cr_line']).dt.days
df_train['loan_ratio'] = df_train['loan_amnt'] / df_train['annual_inc']

df_test['loan_ratio'] = df_test['loan_amnt'] / df_test['annual_inc']
df_train['installment_ratio'] = df_train['installment'] * 12 / df_train['annual_inc']

df_test['installment_ratio'] = df_test['installment'] * 12 / df_test['annual_inc']
df_train['revol_ratio'] = df_train['revol_bal'] / df_train['tot_cur_bal']

df_test['revol_ratio'] = df_test['revol_bal'] / df_test['tot_cur_bal']
# InfをNanに置き換える

df_train.replace([np.inf, -np.inf], np.nan, inplace=True)

df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
df_spi = pd.read_csv('../input/homework-for-students2/spi.csv', parse_dates=['date'])
df_spi['year'] = df_spi.date.dt.year

df_spi['month'] = df_spi.date.dt.month
df_spi_tmp = df_spi.groupby(['year', 'month'])['close'].mean()
# df_train, df_testに年月のカラムを作成

df_train['issue_year'] = df_train.issue_d.dt.year

df_train['issue_month'] = df_train.issue_d.dt.month

df_train['cr_line_year'] = df_train.earliest_cr_line.dt.year

df_train['cr_line_month'] = df_train.earliest_cr_line.dt.month



df_test['issue_year'] = df_test.issue_d.dt.year

df_test['issue_month'] = df_test.issue_d.dt.month

df_test['cr_line_year'] = df_test.earliest_cr_line.dt.year

df_test['cr_line_month'] = df_test.earliest_cr_line.dt.month
df_train = df_train.merge(df_spi_tmp, left_on=['issue_year', 'issue_month'], right_on=['year', 'month'], how='left').drop(['issue_year', 'issue_month'], axis=1)

df_train.rename(columns={'close': 'issue_close'}, inplace=True)

df_train = df_train.merge(df_spi_tmp, left_on=['cr_line_year', 'cr_line_month'], right_on=['year', 'month'], how='left').drop(['cr_line_year', 'cr_line_month'], axis=1)

df_train.rename(columns={'close': 'earliest_cr_line_close'}, inplace=True)



df_test = df_test.merge(df_spi_tmp, left_on=['issue_year', 'issue_month'], right_on=['year', 'month'], how='left').drop(['issue_year', 'issue_month'], axis=1)

df_test.rename(columns={'close': 'issue_close'}, inplace=True)

df_test = df_test.merge(df_spi_tmp, left_on=['cr_line_year', 'cr_line_month'], right_on=['year', 'month'], how='left').drop(['cr_line_year', 'cr_line_month'], axis=1)

df_test.rename(columns={'close': 'earliest_cr_line_close'}, inplace=True)
# issue_closeとearliest_cr_line_closeの差を取る

df_train['diff_close'] = df_train['issue_close'] - df_train['earliest_cr_line_close']

df_test['diff_close'] = df_test['issue_close'] - df_test['earliest_cr_line_close'] 
# 不要なカラムを削除

df_train.drop(['issue_d', 'earliest_cr_line'], axis=1, inplace=True)

df_test.drop(['issue_d', 'earliest_cr_line'], axis=1, inplace=True)
"""

# 日付データを月でカウントエンコーディング

df_train['issue_d'] = df_train['issue_d'].dt.month

df_test['issue_d'] = df_test['issue_d'].dt.month

df_train['earliest_cr_line'] = df_train['earliest_cr_line'].dt.month

df_test['earliest_cr_line'] = df_test['earliest_cr_line'].dt.month



f_cnt = ['issue_d', 'earliest_cr_line']



for i in f_cnt:

    summary_train = df_train[i].value_counts() / len(df_train)

    df_train[i] = df_train[i].map(summary_train)

    df_test[i] = df_test[i].map(summary_train)

    # X_testはX_trainでエンコーディング

"""
df_gdp = pd.read_csv('../input/homework-for-students2/US_GDP_by_State.csv')
# インパクトが小さいのでGross State Productは加えない

df_gdp_tmp = df_gdp.groupby(['State'], as_index=False)['State & Local Spending', 'Real State Growth %', 'Population (million)'].mean()
df_gdp_tmp = df_gdp_tmp[df_gdp_tmp['State'] !=  'All states combined']
df_states = pd.DataFrame({'addr_state': ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'],

                         'State': ['Alaska', 'Alabama', 'Arkansas', 'Arizona', 'California', 'Colorado', 'Connecticut', 'District of Columbia', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Iowa', 'Idaho', 'Illinois', 'Indiana', 'Kansas', 'Kentucky', 'Louisiana', 'Massachusetts', 'Maryland', 'Maine', 'Michigan', 'Minnesota', 'Missouri', 'Mississippi', 'Montana', 'North Carolina', 'North Dakota', 'Nebraska', 'New Hampshire', 'New Jersey', 'New Mexico', 'Nevada', 'New York', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Virginia', 'Vermont', 'Washington', 'Wisconsin', 'West Virginia', 'Wyoming']})
df_gdp_tmp = df_gdp_tmp.merge(df_states, on='State', how='left')
df_train = df_train.merge(df_gdp_tmp, on='addr_state', how='left')

df_test = df_test.merge(df_gdp_tmp, on='addr_state', how='left')
df_train.drop(['State'], axis=1, inplace=True)

df_test.drop(['State'], axis=1, inplace=True)
df_latlong = pd.read_csv('../input/homework-for-students2/statelatlong.csv')



df_latlong.rename(columns={'State': 'addr_state'}, inplace=True)
df_train = df_train.merge(df_latlong, on='addr_state', how='left')

df_test = df_test.merge(df_latlong, on='addr_state', how='left')
# df_zip = pd.read_csv('../input/homework-for-students2/free-zipcode-database.csv')
# df_zip_tmp = df_zip.groupby(['Zipcode'], as_index=False)['Lat', 'Long', 'Xaxis', 'Yaxis', 'Zaxis'].mean().dropna()
# intからstrに変換

# df_zip_tmp['Zipcode'] = df_zip_tmp['Zipcode'].astype(str)
# df_train/df_testのzip-codeの下二桁を削除

# df_train['zip_code'] = df_train['zip_code'].str.replace('xx', '')

# df_test['zip_code'] = df_test['zip_code'].str.replace('xx', '')
# df_train = df_train.merge(df_zip_tmp, left_on='zip_code', right_on='Zipcode', how='left').drop('Zipcode', axis=1)

# df_test = df_test.merge(df_zip_tmp, left_on='zip_code', right_on='Zipcode', how='left').drop('Zipcode', axis=1)
# df_train.info()
df_train['open_acc*loan_amnt'] = df_train['open_acc'] * df_train['loan_amnt']

df_test['open_acc*loan_amnt'] = df_test['open_acc'] * df_test['loan_amnt']
df_train['inq_last_6mths*grade_rank'] = df_train['inq_last_6mths'] * df_train['grade_rank']

df_test['inq_last_6mths*grade_rank'] = df_test['inq_last_6mths'] * df_test['grade_rank']
df_train['inq_last_6mths*sub_grade_rank'] = df_train['inq_last_6mths'] * df_train['sub_grade_rank']

df_test['inq_last_6mths*sub_grade_rank'] = df_test['inq_last_6mths'] * df_test['sub_grade_rank']
df_train['annual_inc/grade_rank'] = df_train['annual_inc'] / df_train['grade_rank']

df_test['annual_inc/grade_rank'] = df_test['annual_inc'] / df_test['grade_rank']
df_train['annual_inc/sub_grade_rank'] = df_train['annual_inc'] / df_train['sub_grade_rank']

df_test['annual_inc/sub_grade_rank'] = df_test['annual_inc'] / df_test['sub_grade_rank']
df_train['dti*grade_rank'] = df_train['dti'] * df_train['grade_rank']

df_test['dti*grade_rank'] = df_test['dti'] * df_test['grade_rank']
df_train['dti*sub_grade_rank'] = df_train['dti'] * df_train['sub_grade_rank']

df_test['dti*sub_grade_rank'] = df_test['dti'] * df_test['sub_grade_rank']
# title中にある大文字をすべて小文字に変換

df_train['title'] = df_train['title'].str.lower()

df_test['title'] = df_test['title'].str.lower()
"""

# 関数を定義

def fnc_title_flg(x):

    if 'debt' in x:

        return 1

    elif 'refinancing' in x:

        return 2

    elif 'credit' in x:

        return 3

    elif 'home' in x:

        return 4

    elif 'medical' in x:

        return 5

    else:

        return 6

"""
# df_train['debt_flg'] = df_train['title'].str.contains('debt') * 1

# df_train['refinancing_flg'] = df_train['title'].str.contains('refinancing') * 1

# df_train['credit_flg'] = df_train['title'].str.contains('credit') * 1

# df_train['home_flg'] = df_train['title'].str.contains('home') * 1

# df_train['medical_flg'] = df_train['title'].str.contains('medical') * 1

# df_train['consolidation_flg'] = df_train['title'].str.contains('consolidation') * 1
# df_test['debt_flg'] = df_test['title'].str.contains('debt') * 1

# df_test['refinancing_flg'] = df_test['title'].str.contains('refinancing') * 1

# df_test['credit_flg'] = df_test['title'].str.contains('credit') * 1

# df_test['home_flg'] = df_test['title'].str.contains('home') * 1

# df_test['medical_flg'] = df_test['title'].str.contains('medical') * 1

# df_test['consolidation_flg'] = df_test['title'].str.contains('consolidation') * 1
"""

# 関数を適用

df_train['title'].apply(fnc_title_flg)

"""
df_train['missing_amnt'] = df_train.isnull().sum(axis=1)

df_test['missing_amnt'] = df_test.isnull().sum(axis=1)
"""

# 関数を定義

def func_emp_length(x):

    if x == '< 1 year':

        return 0.5

    elif x == '1 year':

        return 1.0

    elif x == '2 years':

        return 2.0

    elif x == '3 years':

        return 3.0

    elif x == '4 years':

        return 4.0

    elif x == '5 years':

        return 5.0

    elif x == '6 years':

        return 6.0

    elif x == '7 years':

        return 7.0

    elif x == '8 years':

        return 8.0

    elif x == '9 years':

        return 9.0

    elif x == '10+ years':

        return 10.0

"""
"""

# 関数を適用

df_train['emp_length'] = df_train['emp_length'].apply(func_emp_length)

df_test['emp_length'] = df_test['emp_length'].apply(func_emp_length)

"""
y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = df_test
# emp_titleを分離

# TXT_train = X_train.emp_title.copy()

# TXT_test = X_test.emp_title.copy()
# 2値データのためワンホットエンコーディング

# f_oh = ['initial_list_status', 'application_type']



# encoder = OneHotEncoder(cols=f_oh)



# X_train = encoder.fit_transform(X_train)

# X_test = encoder.transform(X_test)
# とりあえずオーディナルエンコーデイング

# f_oe = ['initial_list_status']



# encoder = OrdinalEncoder(cols=f_oe)



# X_train = encoder.fit_transform(X_train)

# X_test = encoder.transform(X_test)
# 極端な小数クラスを含むためカウントエンコーディング



# f_cnt = ['delinq_2yrs', 'inq_last_6mths', 'zip_code', 'title']



# f_cnt = ['inq_last_6mths', 'zip_code', 'title']



# for i in f_cnt:

#    summary = X_train[i].value_counts() / len(X_train)

#    X_train[i] = X_train[i].map(summary)

#    X_test[i] = X_test[i].map(summary)
"""

# Otherクラスのサイズが１だが・・・



col = 'home_ownership'

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 



    

# X_trainのカテゴリ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)





X_train['home_ownership_2'] = enc_train

X_train.drop('home_ownership', axis=1, inplace=True)



X_test['home_ownership_2'] = enc_test

X_test.drop('home_ownership', axis=1, inplace=True)

"""
cols = ['mths_since_last_major_derog','mths_since_last_record', 'mths_since_last_delinq']
X_train[cols] = X_train[cols].fillna(999)

X_test[cols] = X_test[cols].fillna(999)
X_train['emp_length'].fillna('0', inplace=True) 

X_test['emp_length'].fillna('0', inplace=True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
f_std = ['loan_amnt', 'installment', 'annual_inc', 'mths_since_last_delinq', 'mths_since_last_record', 'revol_bal', 'mths_since_last_major_derog', 'tot_coll_amt', 'tot_cur_bal', 'State & Local Spending', 'issue_close', 'earliest_cr_line_close', 'passed_days', 'open_acc*loan_amnt', 'inq_last_6mths*grade_rank', 'inq_last_6mths*sub_grade_rank', 'open_acc*loan_amnt', 'inq_last_6mths*grade_rank', 'inq_last_6mths*sub_grade_rank', 'annual_inc/grade_rank', 'annual_inc/sub_grade_rank' ,'dti*grade_rank', 'dti*sub_grade_rank', 'Latitude', 'Longitude', 'diff_close']
X_train[f_std] = scaler.fit_transform(X_train[f_std])

X_test[f_std] = scaler.transform(X_test[f_std])
# emp_titleを除去

X_train.drop(['emp_title'], axis=1, inplace=True)

X_test.drop(['emp_title'], axis=1, inplace=True)
X_train.drop('City', axis=1, inplace=True)

X_test.drop('City', axis=1, inplace=True)
X_train_2 = X_train.fillna(X_train.median())

X_test_2 = X_test.fillna(X_test.median())
X_train_2['title'].fillna(X_train_2['purpose'], inplace=True)

X_test_2['title'].fillna(X_test_2['purpose'], inplace=True)
# バリデーション

from sklearn.model_selection import train_test_split



X_train_, X_val, y_train_, y_val = train_test_split(X_train_2, y_train, test_size=0.2, random_state=0)
# カテゴリのカラムを抽出

cols = ['emp_length', 'home_ownership', 'purpose', 'title', 'zip_code', 'addr_state', 'initial_list_status', 'application_type']



train_pool = Pool(X_train_, y_train_, cat_features=cols)

validate_pool = Pool(X_val, y_val, cat_features=cols)
# インスタンス作成

model = CatBoostClassifier(iterations=5000, random_seed=71, learning_rate=0.05, eval_metric='AUC')
# 学習

model.fit(train_pool, eval_set=validate_pool, early_stopping_rounds=20, use_best_model=True, plot=True)
y_pred = model.predict_proba(X_val)[:,1]

score = roc_auc_score(y_val, y_pred)

print(score)
# 欠損値の穴埋め

# TXT_train.fillna('Unemployed', inplace=True)

# TXT_test.fillna('Unemployed', inplace=True)
# tfidf = TfidfVectorizer(max_features=1000, use_idf=True)
# TXT_train = tfidf.fit_transform(TXT_train)

# TXT_test = tfidf.transform(TXT_test)
# X_train_2 = sp.sparse.hstack([X_train.values, TXT_train])

# X_test_2 = sp.sparse.hstack([X_test.values, TXT_test])
# X_train_3 = X_train_2.tocsr()

# X_test_3 = X_test_2.tocsr()
"""

# GBDT

scores = []



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, train_ix, test_ix in tqdm(skf.split(X_train_3, y_train)):

    X_train_, y_train_ = X_train_3[train_ix], y_train[train_ix]

    X_val, y_val = X_train_3[test_ix], y_train[test_ix]

    

        

    clf = GradientBoostingClassifier()

    

    clf.fit(X_train_, y_train_)

    y_pred = clf.predict_proba(X_test_)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)



    print('CV Score of Fold_%d is %f' % (i, score))

"""
"""

# LGBM

# TF-IDF抜き

scores = []



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

        

    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

                                importance_type='split', learning_rate=0.05, max_depth=-1,

                                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                                n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

                                random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    

    clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

    y_pred = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)



    print('CV Score of Fold_%d is %f' % (i, score))

"""
"""

# LGBM

# TF-IDFあり



scores = []



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train_3, y_train))):

    X_train_, y_train_ = X_train_3[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train_3[test_ix], y_train.values[test_ix]

    

        

    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

                                importance_type='split', learning_rate=0.05, max_depth=-1,

                                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                                n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

                                random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    

    clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

    y_pred = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)



    print('CV Score of Fold_%d is %f' % (i, score))

"""
"""

# TF-IDF抜き

print(np.mean(scores))

print(scores)

"""
"""

# TF-IDFあり

print(np.mean(scores))

print(scores)

"""
#重要性

# DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
# fig, ax = plt.subplots(figsize=(5, 8))

# lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')
# 全データで再学習し、testに対して予測する

#clf = GradientBoostingClassifier()



#clf.fit(X_train_3, y_train)



#y_pred = clf.predict_proba(X_test_3)[:,1]
"""

# 全データで再学習し、testに対して予測する

clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

                                importance_type='split', learning_rate=0.05, max_depth=-1,

                                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                                n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

                                random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)



clf.fit(X_train_3, y_train, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

y_pred = clf.predict_proba(X_test_3)[:,1]

"""
# sample submissionを読み込んで、予測値を代入の後、保存する

submission = pd.read_csv('../input/homework-for-students2/sample_submission.csv', index_col=0)



y_pred_2 = model.predict_proba(X_test_2)[:,1]

submission.loan_condition = y_pred_2

submission.to_csv('submission.csv')

submission.head()