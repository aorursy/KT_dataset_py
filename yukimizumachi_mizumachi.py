# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pprint

from hyperopt import hp, tpe

from hyperopt.fmin import fmin

import datetime

from pandas import Series, DataFrame

from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer



import xgboost as xgb

import lightgbm as lgbm



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



from sklearn.ensemble import GradientBoostingClassifier



from sklearn.model_selection import GridSearchCV

# 指標を計算するため

from sklearn.metrics import accuracy_score, cohen_kappa_score, make_scorer, f1_score, recall_score

# 見た目を綺麗にするもの

import matplotlib.pyplot as plt

import seaborn as sns

# 保存

from sklearn.externals.joblib import dump

from sklearn.externals.joblib import load





import os

import glob

os.getcwd() 



pd.set_option('display.max_columns', 500)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import gc

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from sklearn.preprocessing import LabelEncoder



from tqdm import tqdm_notebook as tqdm

from category_encoders import OrdinalEncoder



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



import lightgbm as lgb

from lightgbm import LGBMClassifier
import pandas as pd

sample_submission = pd.read_csv("../input/exam-for-students20200129/sample_submission.csv", index_col=0)

df_test = pd.read_csv("../input/exam-for-students20200129/test.csv")

df_train = pd.read_csv("../input/exam-for-students20200129/train.csv")
df_info = pd.read_csv("../input/exam-for-students20200129/country_info.csv")
df_train.shape,df_test.shape,df_info.shape
df_info.head()
df_train.Age.head(10)
# サブテーブルをマージ - spi.csv をマージする

df_train = df_train.merge(df_info, on=['Country'], how='left')

df_test = df_test.merge(df_info, on=['Country'], how='left')
df_train
# sns.countplot(df_train.ConvertedSalary)

# plt.show()
#descriptive statistics summary

df_train['ConvertedSalary'].describe()
df_train['ConvertedSalary'] = df_train['ConvertedSalary'].apply(np.log1p)

#可視化

f = 'ConvertedSalary'



plt.figure(figsize=[7,7])

df_train[f].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(f)

plt.ylabel('density')

plt.show()
#欠損値があるものはフラグを立てる

df_train['SalaryType_flg'] = df_train.SalaryType.isnull()

df_test['SalaryType_flg'] = df_test.SalaryType.isnull()



df_train['Country_flg'] = df_train.Country.isnull()

df_test['Country_flg'] = df_test.Country.isnull()



df_train['Employment_flg'] = df_train.Employment.isnull()

df_test['Employment_flg'] = df_test.Employment.isnull()



df_train['Currency_flg'] = df_train.Currency.isnull()

df_test['Currency_flg'] = df_test.Currency.isnull()



df_train['YearsCodingProf'] = df_train.YearsCodingProf.isnull()

df_test['YearsCodingProf'] = df_test.YearsCodingProf.isnull()
X_train.head()
#XとYに分割する

y_train=df_train.ConvertedSalary

#目的変数をDROPして、行ではなくカラムなので、axis=1

X_train=df_train.drop(['ConvertedSalary'], axis=1)



#テストデータセット

X_test=df_test
#全項目からカテゴリ（object）を抜き出し（リスト化）、ユニークカウントをとる

# dtypeがobjectのカラム名とユニーク数を確認してみましょう。

cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, X_train[col].nunique())
#target_encording

target = 'ConvertedSalary'

col='SalaryType'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 

enc_test = enc_test.rename(columns={'purpose': 'purpose_te'})

    

# X_trainのカテゴリ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)

    

#結合

X_train=pd.concat([X_train, enc_train], axis=1)

X_test=pd.concat([X_test, enc_test], axis=1)

#カラム名を変更する場合

X_train = X_train.rename(columns={0: 'SalaryType_te'})

X_test = X_test.rename(columns={0: 'SalaryType_te'})
#target_encording

target = 'ConvertedSalary'

col='Country'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 

enc_test = enc_test.rename(columns={'purpose': 'purpose_te'})

    

# X_trainのカテゴリ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)

    

#結合

X_train=pd.concat([X_train, enc_train], axis=1)

X_test=pd.concat([X_test, enc_test], axis=1)

#カラム名を変更する場合

X_train = X_train.rename(columns={0: 'Country_te'})

X_test = X_test.rename(columns={0: 'Country_te'})
#target_encording

target = 'ConvertedSalary'

col='Employment'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 

enc_test = enc_test.rename(columns={'purpose': 'purpose_te'})

    

# X_trainのカテゴリ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)

    

#結合

X_train=pd.concat([X_train, enc_train], axis=1)

X_test=pd.concat([X_test, enc_test], axis=1)

#カラム名を変更する場合

X_train = X_train.rename(columns={0: 'Employment_te'})

X_test = X_test.rename(columns={0: 'Employment_te'})
# #target_encording

# target = 'ConvertedSalary'

# col='Currency'

# X_temp = pd.concat([X_train, y_train], axis=1)



# # X_testはX_trainでエンコーディングする

# summary = X_temp.groupby([col])[target].mean()

# enc_test = X_test[col].map(summary) 

# enc_test = enc_test.rename(columns={'purpose': 'purpose_te'})

    

# # X_trainのカテゴリ変数をoofでエンコーディングする

# skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





# enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



# for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

#     X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

#     X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



#     summary = X_train_.groupby([col])[target].mean()

#     enc_train.iloc[val_ix] = X_val[col].map(summary)

    

# #結合

# X_train=pd.concat([X_train, enc_train], axis=1)

# X_test=pd.concat([X_test, enc_test], axis=1)

# #カラム名を変更する場合

# X_train = X_train.rename(columns={0: 'Currency_te'})

# X_test = X_test.rename(columns={0: 'Currency_te'})
# #target_encording

# target = 'ConvertedSalary'

# col='MilitaryUS'

# X_temp = pd.concat([X_train, y_train], axis=1)



# # X_testはX_trainでエンコーディングする

# summary = X_temp.groupby([col])[target].mean()

# enc_test = X_test[col].map(summary) 

# enc_test = enc_test.rename(columns={'purpose': 'purpose_te'})

    

# # X_trainのカテゴリ変数をoofでエンコーディングする

# skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





# enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



# for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

#     X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

#     X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



#     summary = X_train_.groupby([col])[target].mean()

#     enc_train.iloc[val_ix] = X_val[col].map(summary)

    

# #結合

# X_train=pd.concat([X_train, enc_train], axis=1)

# X_test=pd.concat([X_test, enc_test], axis=1)

# #カラム名を変更する場合

# X_train = X_train.rename(columns={0: 'MilitaryUS_te'})

# X_test = X_test.rename(columns={0: 'MilitaryUS_te'})
# #target_encording

# target = 'ConvertedSalary'

# col='Age'

# X_temp = pd.concat([X_train, y_train], axis=1)



# # X_testはX_trainでエンコーディングする

# summary = X_temp.groupby([col])[target].mean()

# enc_test = X_test[col].map(summary) 

# enc_test = enc_test.rename(columns={'purpose': 'purpose_te'})

    

# # X_trainのカテゴリ変数をoofでエンコーディングする

# skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





# enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



# for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

#     X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

#     X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



#     summary = X_train_.groupby([col])[target].mean()

#     enc_train.iloc[val_ix] = X_val[col].map(summary)

    

# #結合

# X_train=pd.concat([X_train, enc_train], axis=1)

# X_test=pd.concat([X_test, enc_test], axis=1)

# #カラム名を変更する場合

# X_train = X_train.rename(columns={0: 'Age_te'})

# X_test = X_test.rename(columns={0: 'Age_te'})
# #target_encording

# target = 'ConvertedSalary'

# col='Student'

# X_temp = pd.concat([X_train, y_train], axis=1)



# # X_testはX_trainでエンコーディングする

# summary = X_temp.groupby([col])[target].mean()

# enc_test = X_test[col].map(summary) 

# enc_test = enc_test.rename(columns={'purpose': 'purpose_te'})

    

# # X_trainのカテゴリ変数をoofでエンコーディングする

# skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





# enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



# for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

#     X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

#     X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



#     summary = X_train_.groupby([col])[target].mean()

#     enc_train.iloc[val_ix] = X_val[col].map(summary)

    

# #結合

# X_train=pd.concat([X_train, enc_train], axis=1)

# X_test=pd.concat([X_test, enc_test], axis=1)

# #カラム名を変更する場合

# X_train = X_train.rename(columns={0: 'Student_te'})

# X_test = X_test.rename(columns={0: 'Student_te'})
# #target_encording

# target = 'ConvertedSalary'

# col='YearsCoding'

# X_temp = pd.concat([X_train, y_train], axis=1)



# # X_testはX_trainでエンコーディングする

# summary = X_temp.groupby([col])[target].mean()

# enc_test = X_test[col].map(summary) 

# enc_test = enc_test.rename(columns={'purpose': 'purpose_te'})

    

# # X_trainのカテゴリ変数をoofでエンコーディングする

# skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





# enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



# for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

#     X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

#     X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



#     summary = X_train_.groupby([col])[target].mean()

#     enc_train.iloc[val_ix] = X_val[col].map(summary)

    

# #結合

# X_train=pd.concat([X_train, enc_train], axis=1)

# X_test=pd.concat([X_test, enc_test], axis=1)

# #カラム名を変更する場合

# X_train = X_train.rename(columns={0: 'YearsCoding_te'})

# X_test = X_test.rename(columns={0: 'YearsCoding_te'})
# #target_encording

# target = 'ConvertedSalary'

# col='CompanySize'

# X_temp = pd.concat([X_train, y_train], axis=1)



# # X_testはX_trainでエンコーディングする

# summary = X_temp.groupby([col])[target].mean()

# enc_test = X_test[col].map(summary) 

# enc_test = enc_test.rename(columns={'purpose': 'purpose_te'})

    

# # X_trainのカテゴリ変数をoofでエンコーディングする

# skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





# enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



# for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

#     X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

#     X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



#     summary = X_train_.groupby([col])[target].mean()

#     enc_train.iloc[val_ix] = X_val[col].map(summary)

    

# #結合

# X_train=pd.concat([X_train, enc_train], axis=1)

# X_test=pd.concat([X_test, enc_test], axis=1)

# #カラム名を変更する場合

# X_train = X_train.rename(columns={0: 'CompanySize_te'})

# X_test = X_test.rename(columns={0: 'CompanySize_te'})
# #target_encording

# target = 'ConvertedSalary'

# col='CompanySize'

# X_temp = pd.concat([X_train, y_train], axis=1)



# # X_testはX_trainでエンコーディングする

# summary = X_temp.groupby([col])[target].mean()

# enc_test = X_test[col].map(summary) 

# enc_test = enc_test.rename(columns={'purpose': 'purpose_te'})

    

# # X_trainのカテゴリ変数をoofでエンコーディングする

# skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





# enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



# for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

#     X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

#     X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



#     summary = X_train_.groupby([col])[target].mean()

#     enc_train.iloc[val_ix] = X_val[col].map(summary)

    

# #結合

# X_train=pd.concat([X_train, enc_train], axis=1)

# X_test=pd.concat([X_test, enc_test], axis=1)

# #カラム名を変更する場合

# X_train = X_train.rename(columns={0: 'CompanySize_te'})

# X_test = X_test.rename(columns={0: 'CompanySize_te'})
#encoder = OrdinalEncoder()

encoder =OrdinalEncoder(cols=cats)



X_train[cats] = encoder.fit_transform(X_train[cats])

X_test[cats] = encoder.transform(X_test[cats])
X_train.fillna(-9999,inplace=True)

X_test.fillna(-9999,inplace=True)
# %%time

# import lightgbm as lgb

# import optuna.integration.lightgbm_tuner as lgb_tuner

# from sklearn.model_selection import train_test_split



# X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,random_state=71)



# train_dataset=lgb.Dataset(X_train,y_train)

# valid_dataset=lgb.Dataset(X_val,y_val,reference=train_dataset)



# params={"objective":"regression",

#                     "metric":"rmse"

#        }



 

# model1=lgb_tuner.train(params,

#                       train_set=train_dataset,

#                       valid_sets=[valid_dataset],

#                       num_boost_round=300,

#                       early_stopping_rounds=50

#                      )

       

# # Optuna で最適化したモデルの Holt-out データに対するスコア

# y_pred_tuned = model1.predict(X_val)

# tuned_metric = mean_squared_error(y_val, y_pred_tuned)**0.5

# print('tuned model metric: ', tuned_metric)



# print(np.mean(scores))

# print(scores)
# # 全データで再学習し、testに対して予測する

# #clf.fit(X_train, y_train)



# #y_pred1 = model1.predict(X_test)

# y_pred1= np.expm1(model1.predict(X_test))



# #y_pred = clf.predict(X_test)

# #y_pred = clf.predict_proba(X_test)
# lgb.plot_importance(model1, max_num_features=50,  importance_type='gain',figsize=(12, 25))
# # sample submissionを読み込んで、予測値を代入の後、保存する

# sample_submission = pd.read_csv("../input/exam-for-students20200129/sample_submission.csv", index_col=0)

# sample_submission.ConvertedSalary = y_pred



# now = datetime.datetime.now()

# filename = './output/log_' + now.strftime('%Y%m%d_%H%M%S') + '.csv'

# sample_submission.to_csv(now.strftime('%Y%m%d_%H%M%S') + '_sample_submission.csv')
import pandas as pd

sample_submission = pd.read_csv("../input/exam-for-students20200129/sample_submission.csv", index_col=0)

df_test = pd.read_csv("../input/exam-for-students20200129/test.csv")

df_train = pd.read_csv("../input/exam-for-students20200129/train.csv")
#欠損値があるものはフラグを立てる

df_train['SalaryType_flg'] = df_train.SalaryType.isnull()

df_test['SalaryType_flg'] = df_test.SalaryType.isnull()



df_train['Country_flg'] = df_train.Country.isnull()

df_test['Country_flg'] = df_test.Country.isnull()



df_train['Employment_flg'] = df_train.Employment.isnull()

df_test['Employment_flg'] = df_test.Employment.isnull()



df_train['Currency_flg'] = df_train.Currency.isnull()

df_test['Currency_flg'] = df_test.Currency.isnull()



df_train['YearsCodingProf'] = df_train.YearsCodingProf.isnull()

df_test['YearsCodingProf'] = df_test.YearsCodingProf.isnull()
df_train['ConvertedSalary'] = df_train['ConvertedSalary'].apply(np.log1p)

#XとYに分割する

y_train=df_train.ConvertedSalary

#目的変数をDROPして、行ではなくカラムなので、axis=1

X_train=df_train.drop(['ConvertedSalary'], axis=1)



#テストデータセット

X_test=df_test
#全項目からカテゴリ（object）を抜き出し（リスト化）、ユニークカウントをとる

# dtypeがobjectのカラム名とユニーク数を確認してみましょう。

cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, X_train[col].nunique())
#target_encording

target = 'ConvertedSalary'

col='SalaryType'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 

enc_test = enc_test.rename(columns={'purpose': 'purpose_te'})

    

# X_trainのカテゴリ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)

    

#結合

X_train=pd.concat([X_train, enc_train], axis=1)

X_test=pd.concat([X_test, enc_test], axis=1)

#カラム名を変更する場合

X_train = X_train.rename(columns={0: 'SalaryType_te'})

X_test = X_test.rename(columns={0: 'SalaryType_te'})
#target_encording

target = 'ConvertedSalary'

col='Country'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 

enc_test = enc_test.rename(columns={'purpose': 'purpose_te'})

    

# X_trainのカテゴリ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)

    

#結合

X_train=pd.concat([X_train, enc_train], axis=1)

X_test=pd.concat([X_test, enc_test], axis=1)

#カラム名を変更する場合

X_train = X_train.rename(columns={0: 'Country_te'})

X_test = X_test.rename(columns={0: 'Country_te'})
#target_encording

target = 'ConvertedSalary'

col='Employment'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 

enc_test = enc_test.rename(columns={'purpose': 'purpose_te'})

    

# X_trainのカテゴリ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)

    

#結合

X_train=pd.concat([X_train, enc_train], axis=1)

X_test=pd.concat([X_test, enc_test], axis=1)

#カラム名を変更する場合

X_train = X_train.rename(columns={0: 'Employment_te'})

X_test = X_test.rename(columns={0: 'Employment_te'})
#target_encording

target = 'ConvertedSalary'

col='Currency'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 

enc_test = enc_test.rename(columns={'purpose': 'purpose_te'})

    

# X_trainのカテゴリ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)

    

#結合

X_train=pd.concat([X_train, enc_train], axis=1)

X_test=pd.concat([X_test, enc_test], axis=1)

#カラム名を変更する場合

X_train = X_train.rename(columns={0: 'Currency_te'})

X_test = X_test.rename(columns={0: 'Currency_te'})
#target_encording

target = 'ConvertedSalary'

col='MilitaryUS'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 

enc_test = enc_test.rename(columns={'purpose': 'purpose_te'})

    

# X_trainのカテゴリ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)

    

#結合

X_train=pd.concat([X_train, enc_train], axis=1)

X_test=pd.concat([X_test, enc_test], axis=1)

#カラム名を変更する場合

X_train = X_train.rename(columns={0: 'MilitaryUS_te'})

X_test = X_test.rename(columns={0: 'MilitaryUS_te'})
#encoder = OrdinalEncoder()

encoder =OrdinalEncoder(cols=cats)



X_train[cats] = encoder.fit_transform(X_train[cats])

X_test[cats] = encoder.transform(X_test[cats])
X_train.fillna(-9999,inplace=True)

X_test.fillna(-9999,inplace=True)
%%time

import lightgbm as lgb

import optuna.integration.lightgbm_tuner as lgb_tuner

from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,random_state=71)



train_dataset=lgb.Dataset(X_train,y_train)

valid_dataset=lgb.Dataset(X_val,y_val,reference=train_dataset)



params={"objective":"regression",

                    "metric":"rmse"

       }



 

model2=lgb_tuner.train(params,

                      train_set=train_dataset,

                      valid_sets=[valid_dataset],

                      num_boost_round=300,

                      early_stopping_rounds=50

                     )

       

# Optuna で最適化したモデルの Holt-out データに対するスコア

y_pred_tuned = model2.predict(X_val)

tuned_metric = mean_squared_error(y_val, y_pred_tuned)**0.5

print('tuned model metric: ', tuned_metric)



# 全データで再学習し、testに対して予測する

#clf.fit(X_train, y_train)



#y_pred2 = model2.predict(X_test)

y_pred2= np.expm1(model2.predict(X_test))

#y_pred = clf.predict(X_test)

#y_pred = clf.predict_proba(X_test)
lgb.plot_importance(model2, max_num_features=50,  importance_type='gain',figsize=(12, 25))
import pandas as pd

sample_submission = pd.read_csv("../input/exam-for-students20200129/sample_submission.csv", index_col=0)

df_test = pd.read_csv("../input/exam-for-students20200129/test.csv")

df_train = pd.read_csv("../input/exam-for-students20200129/train.csv")





#欠損値があるものはフラグを立てる

df_train['SalaryType_flg'] = df_train.SalaryType.isnull()

df_test['SalaryType_flg'] = df_test.SalaryType.isnull()



df_train['Country_flg'] = df_train.Country.isnull()

df_test['Country_flg'] = df_test.Country.isnull()



df_train['Employment_flg'] = df_train.Employment.isnull()

df_test['Employment_flg'] = df_test.Employment.isnull()



df_train['Currency_flg'] = df_train.Currency.isnull()

df_test['Currency_flg'] = df_test.Currency.isnull()



df_train['YearsCodingProf'] = df_train.YearsCodingProf.isnull()

df_test['YearsCodingProf'] = df_test.YearsCodingProf.isnull()



df_train['ConvertedSalary'] = df_train['ConvertedSalary'].apply(np.log1p)



#XとYに分割する

y_train=df_train.ConvertedSalary

#目的変数をDROPして、行ではなくカラムなので、axis=1

X_train=df_train.drop(['ConvertedSalary'], axis=1)



#テストデータセット

X_test=df_test







#全項目からカテゴリ（object）を抜き出し（リスト化）、ユニークカウントをとる

# dtypeがobjectのカラム名とユニーク数を確認してみましょう。

cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, X_train[col].nunique())



#encoder = OrdinalEncoder()

encoder =OrdinalEncoder(cols=cats)



X_train[cats] = encoder.fit_transform(X_train[cats])

X_test[cats] = encoder.transform(X_test[cats])



X_train.fillna(-9999,inplace=True)

X_test.fillna(-9999,inplace=True)



%%time

import lightgbm as lgb

import optuna.integration.lightgbm_tuner as lgb_tuner

from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,random_state=71)



train_dataset=lgb.Dataset(X_train,y_train)

valid_dataset=lgb.Dataset(X_val,y_val,reference=train_dataset)



params={"objective":"regression",

                    "metric":"rmse"

       }



 

model3=lgb_tuner.train(params,

                      train_set=train_dataset,

                      valid_sets=[valid_dataset],

                      num_boost_round=300,

                      early_stopping_rounds=50

                     )

       

# Optuna で最適化したモデルの Holt-out データに対するスコア

y_pred_tuned = model3.predict(X_val)

tuned_metric = mean_squared_error(y_val, y_pred_tuned)**0.5

print('tuned model metric: ', tuned_metric)



# 全データで再学習し、testに対して予測する

#clf.fit(X_train, y_train)



y_pred3 = np.expm1(model3.predict(X_test))
lgb.plot_importance(model3, max_num_features=50,  importance_type='gain',figsize=(12, 25))
y_pred_x=0.5*y_pred2+0.5*y_pred3

#y_pred_x=y_pred3
# sample submissionを読み込んで、予測値を代入の後、保存する

sample_submission = pd.read_csv("../input/exam-for-students20200129/sample_submission.csv", index_col=0)

sample_submission.ConvertedSalary = y_pred_x



now = datetime.datetime.now()

filename = './output/log_' + now.strftime('%Y%m%d_%H%M%S') + '.csv'

sample_submission.to_csv(now.strftime('%Y%m%d_%H%M%S') + '_sample_submission.csv')


# #y_pred_x=y_pred4

# # こちらもスムーズな進行のために20分の１に間引いていますが、本番では"skiprows=lambda x: x%20!=0"を削除して用いてください。

# submission = pd.read_csv('../input/homework-for-students2/sample_submission.csv', index_col=0)

# len(submission)

# submission.loan_condition = y_pred_x

# submission.to_csv('submission.csv')



# submission.head()

# submission.describe()
