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
import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

import xgboost as xgb





# ラベルエンコーディングする関数 (train, test, カテゴリのdict)

def label_encoding(train: pd.DataFrame, test: pd.DataFrame, col_definition: dict):

    """

    col_definition: encode_col

    """

    n_train = len(train)

    train = pd.concat([train, test], sort=False).reset_index(drop=True)

    for f in col_definition['encode_col']:

        try:

            lbl = preprocessing.LabelEncoder()

            train[f] = lbl.fit_transform(list(train[f].values))

        except:

            print(f)

    test = train[n_train:].reset_index(drop=True)

    train = train[:n_train]

    return train, test
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sub = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
unique_cols = list(train.columns[train.nunique() == 1])

print(unique_cols)
duplicated_cols = list(train.columns[train.T.duplicated()])

print(duplicated_cols)
categorical_cols = list(train.select_dtypes(include=['object']).columns)

print(categorical_cols)
train[categorical_cols].head()
train, test = label_encoding(train, test, col_definition={'encode_col':categorical_cols})
train[categorical_cols].head()
X_train = train.drop(['Id', 'SalePrice'], axis=1)

y_train = np.log1p(train['SalePrice'])

X_test = test.drop(['Id', 'SalePrice'], axis=1)
y_preds = []

models = []

oof_train = np.zeros((len(X_train),))

cv = KFold(n_splits=5, shuffle=True, random_state=0)



for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):

    X_tr = X_train.loc[train_index, :]

    X_val = X_train.loc[valid_index, :]

    y_tr = y_train[train_index]

    y_val = y_train[valid_index]



    # 特徴量と目的変数をxgboostのデータ構造に変換する

    dtrain = xgb.DMatrix(X_tr, label=y_tr)

    dvalid = xgb.DMatrix(X_val, label=y_val)

    dtest = xgb.DMatrix(X_test)

    

    # ハイパラメータの設定

    params = {

        'objective': 'reg:squarederror',

        'eval_metric': 'rmse'

    }

    

    # 学習の実行

    # evalsには、学習データおよびバリデーションデータをセットする

    evals = [(dtrain, 'train'), (dvalid, 'eval')]



    model = xgb.train(params,

                dtrain,

                num_boost_round=1000,

                early_stopping_rounds=10,

                evals=evals

                )





    # バリデーションデータでの予測 (のちにCVスコア確認)

    oof_train[valid_index] = model.predict(dvalid)

    

    # 予測

#     y_pred = model.predict(dtest)

    y_pred = model.predict(dtest, ntree_limit=model.best_ntree_limit) # 最適な決定木の本数で予測を行う



    y_preds.append(y_pred)

    models.append(model)
print(f'CV: {np.sqrt(mean_squared_error(y_train, oof_train))}')
y_sub = sum(y_preds) / len(y_preds)

y_sub = np.expm1(y_sub)

sub['SalePrice'] = y_sub

sub.to_csv('submission.csv', index=False)

sub.head()