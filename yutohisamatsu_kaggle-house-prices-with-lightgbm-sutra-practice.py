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

# 前処理用のライブラリ

from sklearn import preprocessing

# トレーニング用とテスト用にわけバリデーションするライブラリ

from sklearn.model_selection import KFold

# 回帰モデルの結果を評価するライブラリ

from sklearn.metrics import mean_squared_error

import lightgbm as lgb
def label_encoding(train: pd.DataFrame, test: pd.DataFrame, col_definition: dict):

    """

    col_definition : encode_col

    """

    n_train = len(train)

    # .reset_index 連番で振り直し、元のindexを削除

    train = pd.concat([train, test], sort=False).reset_index(drop=True)

    for f in col_definition['encode_col']:

        try:

            # ラベルエンコーディング カテゴリ変数を

            lbl = preprocessing.LabelEncoder()

            # 変更したいデータを数値変数に変換

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
# pandas.DataFrameのメソッド nunique()・・・ユニークな要素の個数（重複を除いた件数）ユニークな要素の個数をintで返す

# nunique()の結果が1,つまり重複している要素があるカラムの個数をみる

unique_cols = list(train.columns[train.nunique() == 1])

print(unique_cols)
# duplicated()メソッドを呼び出すと、行ごとに各列要素すべてが重複するかを判定したSeriesを返す

# 列ごとに判定したい時はDataFrameをTで転置してからduplicated()

duplicated_cols = list(train.columns[train.T.duplicated()])

print(duplicated_cols)
# pandas.DataFrameのメソッドselect_dtypes()・・・特定のデータ型(dtype)の列だけを抽出（選択）する。

categorical_cols = list(train.select_dtypes(include=['object']).columns)

print(categorical_cols)
train[categorical_cols].head()
# train,testのデータセットを返す。カテゴリ変数はエンコードする

train, test = label_encoding(train, test, col_definition={'encode_col': categorical_cols})
train[categorical_cols].head()
X_train = train.drop(['Id', 'SalePrice'], axis=1)

# log1p・・・ SalePrice がゼロに近い場合でも正確な値が計算できる方法　log()は自然対数

y_train = np.log1p(train['SalePrice'])

X_test = test.drop(['Id', 'SalePrice'], axis=1)
# 各分割でのX_testに対する予測値を格納するリスト

y_preds = []



# 各分割で学習したモデルを格納するリスト

models = []



# 各分割での検証用データセット(X_val)に対する予測値を格納するnumpy.ndarray

# trainのoof 各分割でのoofに対する予測値を収納

oof_train = np.zeros((len(X_train),))



# 分割方法の設定

# n_splits(分割数) データセットを5つに分ける shuffle=Trueで、データセットを分割前にシャッフル

cv = KFold(n_splits=5, shuffle=True, random_state=0)



params = {

    'num_leaves' : 24,

    'max_depth' : 6,

    'objective' : 'regression', #目的関数・・「回帰」を解く

    'metric' : 'rmse',

    'learning_rate' : 0.05

}



# 設定した分割方法に基づいて、各分割での学習用・検証用データセットに対応するindexを取得

# enumerate　要素と同時に各分割のindex(fold_id)も取得

# 今回、fold_idは利用していないが、モデルを個別に保存する場合などにファイル名として使う



for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):



    # 各fold_idで取得したindexに基づいて、データセットを分割

    # X_tr,X_val = pandas.DataFrame, y_tr,y_val = numpy.ndarray indexの指定方法が両者で異なる点に注意

    X_tr = X_train.loc[train_index, :]

    X_val = X_train.loc[valid_index, :]

    y_tr = y_train[train_index]

    y_val = y_train[valid_index]

    

    # 学習用データセット、検証用データセットに分割

    lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_cols)

    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=categorical_cols)

    

    model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10, num_boost_round=1000, early_stopping_rounds=10)

    

    # 各分割での検証用データセット(X_val)に対する予測値を格納

    # oof_train(numpy.ndarray)

    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)

    

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)



    # 各分割でのX_testに対する予測値を、先に作成したy_predsというリストに格納

    y_preds.append(y_pred)

    

    # 各分割で学習したモデルを先に作成したmodelsというリストに格納

    models.append(model)
# CV・・・クロスバリデーション(交差検証)

# trainデータをさらにtestデータとtrainデータ(交差検証用のデータ)に分け、モデルを構築したあと交差検証用のデータを使ってモデルの精度の測定



# 平均二乗誤差 (MSE, Mean Squared Error) とは、実際の値と予測値の絶対値の 2 乗を平均

# MSE の平方根を 二乗平均平方根誤差 (RMSE: Root Mean Squared Error) 上記の MSE で、二乗したことの影響を平方根で補正

print(f'CV: {np.sqrt(mean_squared_error(y_train, oof_train))}')
# 正解率を計算するためのaccuracy_score

from sklearn.metrics import accuracy_score



accuracy_score(np.round(y_train), np.round(oof_train))
# ここら辺何してるの？

y_sub = sum(y_preds) / len(y_preds)



# e の x(y_sub) 乗から 1 を引いた値を返す・・・コンペの目的が、SalePriceの対数を返す(予測する)から

y_sub = np.expm1(y_sub)

sub['SalePrice'] = y_sub

sub.to_csv('submission.csv', index=False)

sub.head()