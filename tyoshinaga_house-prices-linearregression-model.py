# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
#データを確認
train_df.shape
train_df.columns
train_df.info()
#SalePriceの記述統計量
train_df['SalePrice'].describe()
#ヒストグラム
sns.distplot(train_df['SalePrice']);
#ロングテール
corr = train_df.select_dtypes(include = ['float64', 'int64']).corr()
corr['SalePrice'].sort_values(ascending = False)
#相関が0.6以上のものを独立変数として採用
features = train_df[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF']]
#独立変数の確認
features.info()
#欠損値を確認
features.isnull().sum()
#幸いにも欠損値なし
#テストデータにも同様の変数を格納
test_df_corr = test_df[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF']]
#テストデータの欠損値も確認
test_df_corr.isnull().sum()
#欠損値のある行を削除
test_df_corr_fix = test_df_corr.dropna(how='any')
import sklearn
from sklearn.linear_model import LinearRegression
#予測モデルの作成
target = train_df[['SalePrice']]
lreg = LinearRegression()
lreg.fit(features, target)
#回帰係数を確認
print(lreg.coef_)
#切片を確認
print(lreg.intercept_)
#テストデータで予測モデルの当てはめ
y_output = lreg.predict(test_df_corr_fix)
#出力をデータフレームに変換
y_output_df = pd.DataFrame(y_output)
y_output_df.columns = ['SalePrice']
y_output_df.head(10)
#アウトプットとしてIdのカラムが必要なため追加
Id = test_df[['Id']]
Y_output = pd.concat([Id,y_output_df],axis=1)
Y_output.head(10)
#CSVファイルへ書き出し
Y_output.to_csv("submission.csv",index=False)
#以下エラーにより断念したモデル
#目的変数をデータフレームから削除
#train_x = train_df.drop(["SalePrice"], axis = 1)
#変数別に欠損値を確認
#missing = train_df.isnull().sum()
#missing = missing[missing > 0]
#missing.sort_values(inplace=True)
#missing.plot.bar()
#欠損値を含むサンプルが600以上の変数を削除
#small_miss_data = train_df.dropna(axis=1,thresh = 800)
#大幅な欠損を含む列が削除されたか確認
#missing = small_miss_data.isnull().sum()
#missing = missing[missing > 0]
#missing.sort_values(inplace=True)
#missing.plot.bar()
#欠損値を含む行を削除
#fix_data = small_miss_data.dropna(how='any')
#X_multi = fix_data.drop('SalePrice',1)
#Y_target = fix_data.SalePrice