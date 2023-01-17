# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#ライブラリインポート

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

from sklearn.preprocessing import LabelEncoder

import category_encoders as ce

from sklearn.preprocessing import StandardScaler



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import mean_squared_log_error



import lightgbm as lgb

from lightgbm import LGBMClassifier

from lightgbm import LGBMRegressor
df_train = pd.read_csv('../input/exam-for-students20200923/train.csv', index_col = 0)

df_test = pd.read_csv('../input/exam-for-students20200923/test.csv', index_col = 0)
X_train = df_train.drop('ConvertedSalary', axis=1)

y_train = df_train[['ConvertedSalary']]

X_test = df_test
X_train.head()
y_train.head()
X_test.head()
#objectのカラム名とユニーク数を確認

cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, X_train[col].nunique())
oe = ce.OrdinalEncoder(cols=cats, handle_unknown='impute')
#とりあえずカテゴリ変数をまとめてOrdinal Encoding

X_train_oe = oe.fit_transform(X_train)

X_test_oe = oe.fit_transform(X_test) 
X_train_oe.head()
X_test_oe.head()
#欠損値うめる(平均値)

X_train_oe.fillna(X_train_oe.mean(), inplace=True)

X_test_oe.fillna(X_test_oe.mean(), inplace=True)
X_train_oe.head()
#float,intのカラム名とユニーク数を確認

num_cats = []

for col in X_train.columns:

    if X_train[col].dtype ==  'int64' or X_train[col].dtype == 'float64':

        num_cats.append(col)

        

        print(col, X_train[col].nunique())
#回帰モデルのため標準化する

ss = StandardScaler()

ss.fit(X_train_oe[num_cats])

X_train_oe[num_cats] = ss.transform(X_train_oe[num_cats])

X_test_oe[num_cats] = ss.transform(X_test_oe[num_cats])
X_train_oe.head()
X_test_oe.head()
X_train = X_train_oe

X_test = X_test_oe



X = X_train.values

y = y_train.ConvertedSalary.values
#LightGBM のハイパーパラメータ

regressor = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=0.71,

                           importance_type='split', learning_rate=0.05, max_depth=-1,

                           min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                           n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None,

                           random_state=71, reg_alpha=1.0, reg_lambda=1.0, silent=True,

                           subsample=0.9, subsample_for_bin=200000, subsample_freq=0)

#層化抽出5分割

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

#Score格納用

rmsle_score = []

#iteration回数格納用

num_iteration = []
#CV実行

for train_ix, valid_ix in skf.split(X, y):

    X_train, y_train = X[train_ix], y[train_ix]

    X_valid, y_valid = X[valid_ix], y[valid_ix]

    

    fit_train = regressor.fit(X_train, y_train,

                              early_stopping_rounds=500,

                              eval_metric='rmsle',

                              eval_set=[(X_valid, y_valid)],

                              verbose=100)

    #予測

    p_valid = fit_train.predict(X_valid)

    #0判定

    p_valid = np.where(p_valid>0, p_valid, 0)

    #RMSLEを計算

    tmp_rmsle_score = np.sqrt(mean_squared_log_error(y_valid, p_valid))

    #スコアを追加

    rmsle_score.append(tmp_rmsle_score)

    #iteraion回数を追加

    tmp_num_iteration = fit_train.best_iteration_

    num_iteration.append(tmp_num_iteration)

    

    #スコア表示

    print('RMSLE',v_rmsle_score)

    print('Iteration',num_iteration)
#スコア平均

ave_rmsle_score = np.mean(rmsle_score)

print('RSMLE平均',ave_rmsle_score)

#iteration回数平均

ave_num_iteration = np.mean(num_iteration).round()

ave_num_iteration = np.int(ave_num_iteration)

print('iteration回数平均', ave_num_iteration)
#再学習

regressor = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=0.71,

                               importance_type='split', learning_rate=0.05, max_depth=-1,

                               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                               n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None,

                               random_state=71, reg_alpha=1.0, reg_lambda=1.0, silent=True,

                               subsample=0.9, subsample_for_bin=200000, subsample_freq=0,

                               num_iteration = ave_num_iteration)



fit = regressor.fit(X, y)
p_test = fit.predict(X_test)

p_test = np.where(p_test>0, p_test, 0)
#sample_submission.csvを読み込み

df_submission = pd.read_csv('../input/exam-for-students20200923/sample_submission.csv', index_col=0)
#予測結果を代入

df_submission.ConvertedSalary = p_test
#完成

df_submission.to_csv('submission.csv')