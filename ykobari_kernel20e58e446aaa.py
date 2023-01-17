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
import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



from sklearn.ensemble import GradientBoostingClassifier
#ハンズオンのスムーズな進行のために全体の20分の1だけ読み込むことにします。

#実際に課題でモデリングする際には"skiprows=lambda x: x%20!=0"を削除してください。

df_train = pd.read_csv('/kaggle/input/homework-for-students3/train.csv', index_col=0)

df_test =  pd.read_csv('/kaggle/input/homework-for-students3/test.csv', index_col=0)
df_train = df_train[df_train.issue_d.str.contains('(2014|2015)')]

# df_train = df_train[df_train.issue_d.str.contains('2015')]

floats = []

floats.append('loan_amnt')    

for col in df_train.columns:

    if (df_train[col].dtype == 'float64') and (df_train[col].nunique() >= 200):

        floats.append(col)

        

        print(col, df_train[col].nunique())

p01 = df_train[floats].quantile(0.01)

p99 = df_train[floats].quantile(0.99)



df_train[floats] = df_train[floats].clip(p01,p99,axis=1) 
df_train['emp_title'] = df_train['emp_title'].str[0:4]
havenullcol = df_train.columns[df_train.isnull().sum()!=0].values

havenullcol 

for col in havenullcol:

    df_train[col+'isnull'] = np.where(df_train[col].isnull(),1,0)

    df_test[col+'isnull'] = np.where(df_test[col].isnull(),1,0)
y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = df_test
X_train[floats].fillna('0', inplace=True)

X_test[floats].fillna('0', inplace=True)
from sklearn.preprocessing import quantile_transform

# 学習データとテストデータを結合した上でRankGaussによる変換を実施

X_all = pd.concat([X_train, X_test], axis=0)

X_all[floats] =quantile_transform(X_all[floats],  n_quantiles=100, random_state=0, output_distribution='normal')# 学習データとテストデータに再分割

X_train = X_all.iloc[:X_train.shape[0], :]

X_test = X_all.iloc[X_train.shape[0]:, :]
# dtypeがobjectのカラム名とユニーク数を確認してみましょう。

cats = []

for col in X_train.columns:

    if (X_train[col].dtype == 'object') or (X_train[col].nunique() < 200) :

        cats.append(col)

        

        print(col, X_train[col].nunique())
X_train[col].fillna('#', inplace=True)

X_test[col].fillna('#', inplace=True)
oe = OrdinalEncoder(cols=cats, return_df=False)



X_train[cats] = oe.fit_transform(X_train[cats])

X_test[cats] = oe.transform(X_test[cats])
# Adversarial Validation

from sklearn.model_selection import cross_val_predict

# 分割したデータがどちらに由来したものか区別するためのラベル

z_train, z_test = np.zeros(len(X_train)), np.ones(len(X_test))



X = np.concatenate([X_train, X_test], axis=0)

z = np.concatenate([z_train, z_test], axis=0)



# 要素が最初のデータセット由来なのか、次のデータセット由来なのか分類する

#clf = RandomForestClassifier(n_estimators=100,random_state=42)

import lightgbm as lgb

from lightgbm import LGBMClassifier



import matplotlib.pyplot as plt



plt.style.use('ggplot')

%matplotlib inline



clf = LGBMClassifier(boosting_type='gbdt', class_weight='balanced', colsample_bytree=0.71,

                    importance_type='split', learning_rate=0.25, max_depth=-1,

                     min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                     n_estimators=85, n_jobs=-1, num_leaves=8, objective=None,

                     random_state=71, reg_alpha=1.0, reg_lambda=1.0, silent=True,

                     subsample=0.9, subsample_for_bin=200000, subsample_freq=0)



# 分類して確率を計算する

z_pred_proba = cross_val_predict(clf, X, z, cv=5, method='predict_proba')



# # X_trainデータに対応する部分だけ取り出す

z_train_pred_proba = z_pred_proba[:X_train.shape[0]]

# # X_testデータに近いと判断された上位だけ取り出す

y_train =y_train.iloc[np.argsort(z_train_pred_proba[:, 0])][:X_train.shape[0]//2 ]

X_train =X_train.iloc[np.argsort(z_train_pred_proba[:, 0])][:X_train.shape[0]//2 ]

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)





for col in cats:

    # X_testはX_trainでエンコーディングする

    summary = X_temp.groupby([col])[target].mean()

    enc_test1 = X_test[col].map(summary) 





    # X_trainのカテゴリ変数をoofでエンコーディングする

    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





    enc_train1 = Series(np.zeros(len(X_train)), index=X_train.index)



    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([col])[target].mean()

        enc_train1.iloc[val_ix] = X_val[col].map(summary)



    X_train[col] = enc_train1

    X_test[col] = enc_test1
# # CVしてスコアを見てみる。層化抽出で良いかは別途よく考えてみてください。

# scores = []



# skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



# for i, (train_ix, test_ix) in enumerate(skf.split(X_train, y_train)):

#     X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

#     X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    

#     clf = GradientBoostingClassifier(n_estimators=85,  learning_rate=0.25 )

    

#     clf.fit(X_train_, y_train_)

#     y_pred = clf.predict_proba(X_val)[:,1]

#     score = roc_auc_score(y_val, y_pred)

#     scores.append(score)

    

#     print('CV Score of Fold_%d is %f' % (i, score))
# print(np.mean(scores))

# print(scores)
# # 全データで再学習し、testに対して予測する

# clf = GradientBoostingClassifier(n_estimators=85,  learning_rate=0.25 )   

# clf.fit(X_train, y_train)



# y_pred0 = clf.predict_proba(X_test)[:,1]
clf = LGBMClassifier(boosting_type='gbdt', class_weight='balanced', colsample_bytree=0.71,

                    importance_type='split', learning_rate=0.25, max_depth=-1,

                     min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                     n_estimators=85, n_jobs=-1, num_leaves=8, objective=None,

                     random_state=71, reg_alpha=1.0, reg_lambda=1.0, silent=True,

                     subsample=0.9, subsample_for_bin=200000, subsample_freq=0)



clf.fit(X_train, y_train,eval_metric='auc')





y_pred1 = clf.predict_proba(X_test)[:,1]
X_train.fillna(X_train.median(), inplace=True)

X_test.fillna(X_train.median(), inplace=True)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1.0) 

lr.fit(X_train, y_train) 

y_pred2= lr.predict(X_test)
# sample submissionを読み込んで、予測値を代入の後、保存する

# こちらもスムーズな進行のために20分の１に間引いていますが、本番では"skiprows=lambda x: x%20!=0"を削除して用いてください。

submission = pd.read_csv('/kaggle/input/homework-for-students3/sample_submission.csv', index_col=0)



submission.loan_condition = y_pred1 + y_pred2*0 

submission.to_csv('submission.csv')
submission.head()