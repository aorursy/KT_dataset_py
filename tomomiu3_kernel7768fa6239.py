# パッケージの読み込み

import gc

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

#スケールの標準化・正規化(LightGBMではほぼ不要)

from sklearn.preprocessing import StandardScaler, MinMaxScaler



from tqdm import tqdm_notebook as tqdm

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



import lightgbm as lgb

from lightgbm import LGBMRegressor



import re

from hyperopt import fmin, tpe, hp, rand, Trials

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 200)
# 元データの読み込み(CSV)



# 教師データとテストデータのファイル名指定

train_data_filename = '../input/exam-for-students20200129/train.csv'

test_data_filename = '../input/exam-for-students20200129/test.csv'



# ターゲット列名の指定

target = 'ConvertedSalary'



# 元データの読み込み(CSV)

df_train = pd.read_csv(train_data_filename, index_col=0)

df_test = pd.read_csv(test_data_filename, index_col=0)
df_train.describe()
df_test.describe()
df_train.drop(columns='MilitaryUS', inplace=True)

df_test.drop(columns='MilitaryUS', inplace=True)
#df_train = df_train.query('Country in ["Andorra","Aruba","Australia","Austria","Belgium","Bermuda","Brunei","Canada","Cayman Islands","Cyprus","Denmark","Faroe Islands","Finland","France","Germany","Greece","Greenland","Guam","Guernsey","Hong Kong","Iceland","Ireland","Isle of Man","Israel","Italy","Japan","Jersey","Korea, South","Kuwait","Liechtenstein","Luxembourg","Macau","Malta","Monaco","Netherlands","New Zealand","Norway","Portugal","Qatar","San Marino","Singapore","Slovenia","Spain","Sweden","Switzerland","Taiwan","United Arab Emirates","United Kingdom","United States"]')
df_test.shape
# 教師データからターゲットの分離

y_train = df_train[target]

X_train = df_train.drop(columns=[target])



X_test = df_test.copy()



# ターゲットがnullの行の除外

X_train = X_train[y_train.isnull()==False]

y_train = y_train[y_train.isnull()==False]
f = 'AdsPriorities7'



plt.figure(figsize=[7,7])

df_train[f].hist(density=True, alpha=0.5, bins=20)

# plt.hist(df_train[f], density=True, alpha=0.5, bins=20)でも可

# testデータに対する可視化を記入してみましょう

df_test[f].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(f)

plt.ylabel('density')

plt.show()
f = 'AdsActions'



plt.figure(figsize=[7,7])

df_test[f].value_counts(normalize=True).plot(kind='bar', alpha=0.5, color='b')

df_train[f].value_counts(normalize=True).plot(kind='bar', alpha=0.5, color='r')

plt.xlabel(f)

plt.ylabel('density')

plt.show()
X_train['Currency'].value_counts()
X_test['Currency'].value_counts()
X_test['CurrencySymbol'].value_counts()
# dtypeがobjectのカラム名とユニーク数を確認

cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, X_train[col].nunique())
# dtypeがobjectのカラム名とユニーク数を確認

for col in X_test.columns:

    if X_test[col].dtype == 'object':

        

        print(col, X_test[col].nunique())
cats
# ユニーク数が多いものは一旦除外

text_columns = ['DevType', 'CommunicationTools', 'FrameworkWorkedWith']



for col in text_columns:

    cats.remove(col)



# 教師データからテキスト列を除外

X_train.drop(columns=text_columns, inplace=True)

X_test.drop(columns=text_columns, inplace=True)
# Encoderの使い方 OneHotEncoder, OrdinalEncoder

# 0. エンコードしたい列の指定

col = 'Student'



# 1. 使いたいエンコーダを指定

# Ordinalで自分で値を指定したい場合は以下のようにmappingで指定(指定しないと出てきた順に番号が振られる)

encoder = OrdinalEncoder(mapping=[{'col':col,'mapping':{'No':0, 'Yes, part-time':1, 'Yes, full-time':2}}])



# 2. エンコーダの適用(教師データでfit_transform, テストデータでtransform)



X_train[col] = encoder.fit_transform(X_train[col])

X_test[col] = encoder.transform(X_test[col])



cats.remove(col)
X_train['Student'].value_counts()
col = 'CompanySize'



# 1. 使いたいエンコーダを指定

# Ordinalで自分で値を指定したい場合は以下のようにmappingで指定(指定しないと出てきた順に番号が振られる)

encoder = OrdinalEncoder(mapping=[{'col':col,'mapping':{'Fewer than 10 employees':0, '10 to 19 employees':10, '20 to 99 employees':20,

                                                        '100 to 499 employees':100, '500 to 999 employees':500, '1,000 to 4,999 employees':1000,

                                                        '5,000 to 9,999 employees':5000, '10,000 or more employees':10000}}])



# 2. エンコーダの適用(教師データでfit_transform, テストデータでtransform)



X_train[col] = encoder.fit_transform(X_train[col])

X_test[col] = encoder.transform(X_test[col])



cats.remove(col)
X_test[col].value_counts()
col = 'YearsCoding'



# 1. 使いたいエンコーダを指定

# Ordinalで自分で値を指定したい場合は以下のようにmappingで指定(指定しないと出てきた順に番号が振られる)

encoder = OrdinalEncoder(mapping=[{'col':col,'mapping':{'0-2 years':0, '3-5 years':3, '6-8 years':6,

                                                        '9-11 years':9, '12-14 years':12, '15-17 years':15,

                                                        '18-20 years':18, '21-23 years':21, '24-26 years':24,

                                                        '27-29 years':27, '30 or more years':30}}])



# 2. エンコーダの適用(教師データでfit_transform, テストデータでtransform)



X_train[col] = encoder.fit_transform(X_train[col])

X_test[col] = encoder.transform(X_test[col])



cats.remove(col)
X_test[col].value_counts()
col = 'YearsCodingProf'



# 1. 使いたいエンコーダを指定

# Ordinalで自分で値を指定したい場合は以下のようにmappingで指定(指定しないと出てきた順に番号が振られる)

encoder = OrdinalEncoder(mapping=[{'col':col,'mapping':{'0-2 years':0, '3-5 years':3, '6-8 years':6,

                                                        '9-11 years':9, '12-14 years':12, '15-17 years':15,

                                                        '18-20 years':18, '21-23 years':21, '24-26 years':24,

                                                        '27-29 years':27, '30 or more years':30}}])



# 2. エンコーダの適用(教師データでfit_transform, テストデータでtransform)



X_train[col] = encoder.fit_transform(X_train[col])

X_test[col] = encoder.transform(X_test[col])



cats.remove(col)
X_train[col].value_counts()
cols = ['JobSatisfaction', 'CareerSatisfaction']



# 1. 使いたいエンコーダを指定

# Ordinalで自分で値を指定したい場合は以下のようにmappingで指定(指定しないと出てきた順に番号が振られる)

for col in cols:

    encoder = OrdinalEncoder(mapping=[{'col':col,'mapping':{'Extremely dissatisfied':0, 'Moderately dissatisfied':1, 

                                                        'Slightly dissatisfied':2, 'Neither satisfied nor dissatisfied':3,

                                                        'Slightly satisfied':4, 'Moderately satisfied':5,

                                                        'Extremely dissatisfied':6}}])



# 2. エンコーダの適用(教師データでfit_transform, テストデータでtransform)

    X_train[col] = encoder.fit_transform(X_train[col])

    X_test[col] = encoder.transform(X_test[col])



    cats.remove(col)
X_train['CareerSatisfaction'].value_counts()
col = 'LastNewJob'



# 1. 使いたいエンコーダを指定

# Ordinalで自分で値を指定したい場合は以下のようにmappingで指定(指定しないと出てきた順に番号が振られる)

encoder = OrdinalEncoder(mapping=[{'col':col,'mapping':{"I've never had a job":0, 'Less than a year ago':1, 

                                                        'Between 1 and 2 years ago':2, 'Between 2 and 4 years ago':4,

                                                        'More than 4 years ago':8}}])



# 2. エンコーダの適用(教師データでfit_transform, テストデータでtransform)

X_train[col] = encoder.fit_transform(X_train[col])

X_test[col] = encoder.transform(X_test[col])

#enc_train = encoder.fit_transform(X_train[col])

#enc_test = encoder.transform(X_test[col])



cats.remove(col)
#enc_train[col].value_counts()
col = 'Age'



# 1. 使いたいエンコーダを指定

# Ordinalで自分で値を指定したい場合は以下のようにmappingで指定(指定しないと出てきた順に番号が振られる)

encoder = OrdinalEncoder(mapping=[{'col':col,'mapping':{"18 - 24 years old":0, '25 - 34 years old':1, 

                                                        '35 - 44 years old':2, '45 - 54 years old':3,

                                                        '55 - 64 years old':4, '65 years or older':5}}])



# 2. エンコーダの適用(教師データでfit_transform, テストデータでtransform)

X_train[col] = encoder.fit_transform(X_train[col])

X_test[col] = encoder.transform(X_test[col])

#enc_train = encoder.fit_transform(X_train[col])

#enc_test = encoder.transform(X_test[col])



cats.remove(col)
X_train[col].value_counts()
# カテゴリをOrdinalEncoding

oe = OrdinalEncoder(cols=cats)



X_train = oe.fit_transform(X_train)

X_test = oe.transform(X_test)
# 欠損値を補間

X_train.fillna(-1, inplace=True)

X_test.fillna(-1, inplace=True)
%%time



# 金額系なので、RMSLEで最適化してみる。

# 何も指定しないと大抵はRMSEで最適化される。ここでは対数を取っておくとちょうどRMSLEの最適化に相当する。

scores = []



cv_iteration = 0

groups = X_train.Country.values

y_pred_test = np.zeros(X_test.shape[0])



gkf = GroupKFold(n_splits=5)



for i, (train_ix, test_ix) in enumerate(tqdm(gkf.split(X_train, y_train, groups))):

    

    X_train_, y_train_, groups_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix], groups[train_ix]

    X_val, y_val, groups_val = X_train.iloc[test_ix], y_train.iloc[test_ix], groups[test_ix]

    

    print('Train Groups', np.unique(groups_train_))

    print('Val Groups', np.unique(groups_val))



    

    clf = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

                         importance_type='split', learning_rate=0.05, max_depth=-1,

                         min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                         n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

                         random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                         subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    

    clf.fit(X_train_, np.log1p(y_train_), early_stopping_rounds=20, eval_metric='rmse', eval_set=[(X_val, np.log1p(y_val))])

    y_pred = np.expm1(clf.predict(X_val))

    score = mean_squared_error(np.log1p(y_val), np.log1p(y_pred))**0.5

    scores.append(score)

    

    y_pred_test += np.expm1(clf.predict(X_test))

    cv_iteration += clf.best_iteration_

    print('CV Score of Fold_%d is %f' % (i, score))
print(np.mean(scores))

print(scores)

cv_iteration = (cv_iteration // 5) + 1

y_pred_test /= 5
# 全データで再学習し、testに対して予測する

clf = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

                     importance_type='split', learning_rate=0.05, max_depth=-1,

                     min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                     n_estimators=int(cv_iteration), n_jobs=-1, num_leaves=15, objective=None,

                     random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                     subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

clf.fit(X_train, np.log1p(y_train), eval_metric='rmse')

y_pred = np.expm1(clf.predict(X_test))
y_ens = (y_pred + y_pred_test) / 2
imp = DataFrame(clf.booster_.feature_importance(importance_type='gain'), index=X_train.columns, columns=['importance']).sort_values(['importance'], ascending=False)

imp
# こちらもスムーズな進行のために20分の１に間引いていますが、本番では"skiprows=lambda x: x%20!=0"を削除して用いてください。

#submission = pd.read_csv('../input/homework-for-students2/sample_submission.csv', index_col=0, skiprows=lambda x: x%20!=0)

submission = pd.read_csv('../input/exam-for-students20200129/sample_submission.csv', index_col=0)



submission.ConvertedSalary = y_pred

submission.to_csv('submission_base.csv')



submission.ConvertedSalary = y_pred_test

submission.to_csv('submission_cv.csv')



submission.ConvertedSalary = y_ens

submission.to_csv('submission_ens.csv')

submission