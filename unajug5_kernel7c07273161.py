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
import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

import numpy as np

from sklearn.linear_model import LinearRegression
# データの読み込み

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv') #訓練データ

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv') #テストデータ
# 先頭5行の確認

train.head(5)
train.columns
train.SalePrice.describe()


sns.distplot(train.SalePrice)


corrmat = train.corr()

corrmat
k = 10

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T) # .T(転置行列)を行う理由は、corrcoefで相関を算出する際に、各カラムの値を行毎にまとめなければならない為

sns.set(font_scale=1.25) # ヒートマップのフォントサイズを指定



# 算出した相関データをヒートマップで表示

sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea']

sns.pairplot(train[cols], height = 2.5)
train.sort_values(by = 'GrLivArea', ascending = False)[:2] #ascending=False は'GrLivArea'の高いデータから順に並べ替えられる
# 判明したデータのIdの値を指定して削除

train = train.drop(index = train[train['Id'] == 1299].index)

train= train.drop(index = train[train['Id'] == 524].index)
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea']

sns.pairplot(train[cols], height = 2.5)
# GrLivAreaを使用してモデルを学習

# XにGrLivArea、yにSalePriceをセット

X = train[["GrLivArea"]].values

y = train["SalePrice"].values



slr = LinearRegression() # アルゴリズムに線形回帰



slr.fit(X,y) # fit関数で学習



# 偏回帰係数を出力

print('傾き：{0}'.format(slr.coef_[0]))



# y切片(直線とy軸との交点)を出力

print('y切片: {0}'.format(slr.intercept_))
plt.scatter(X,y)

plt.plot(X,slr.predict(X),color='red')
# テストデータの内容確認

test.head(5)
# テストデータの GrLivArea の値をセット

X_test = test[["GrLivArea"]].values



# 学習済みのモデルから予測した結果をセット

y_test_pred = slr.predict(X_test)
y_test_pred
# df_testに SalePrice カラムを追加し、学習済みのモデルから予測した結果をセット

test["SalePrice"] = y_test_pred
# Id, SalePriceの2列だけ表示

test[["Id","SalePrice"]].head()
# Id, SalePriceの2列だけのファイルに変換

test[["Id","SalePrice"]].to_csv("submission.csv",index=False)
# SalePriceと相関の高いOverallQualとGrLivAreaを説明変数に使用

X = train[["OverallQual", "GrLivArea"]].values

y = train["SalePrice"].values



# アルゴリズムに線形回帰(Linear Regression)を採用

slr = LinearRegression()



slr.fit(X,y) # 学習



# 偏回帰係数はscikit-learnのcoefで取得

print('傾き：{0}'.format(slr.coef_))

a1, a2 = slr.coef_



# y切片(直線とy軸との交点)を出力

print('y切片: {0}'.format(slr.intercept_))

b = slr.intercept_
x, y, z = np.array(train["OverallQual"]), np.array(train["GrLivArea"]), np.array(train["SalePrice"].values)

fig = plt.figure()

ax = Axes3D(fig)

ax.scatter3D(np.ravel(x), np.ravel(y), np.ravel(z))



X, Y = np.meshgrid(np.arange(0, 12, 2), np.arange(0, 6000, 1000))

Z = a1 * X + a2 * Y + b

ax.plot_surface(X, Y, Z, alpha = 0.5, color = "red") #alphaで透明度を指定

ax.set_xlabel("OverallQual")

ax.set_ylabel("GrLivArea")

ax.set_zlabel("SalePrice")
# テストデータの内容確認(追加したSalePriceが消えている事)

test.head(5)
# テストデータの OverallQual と GrLivArea の値をセット

X_test = test[["OverallQual", "GrLivArea"]].values



# 学習済みのモデルから予測した結果をセット

y_test_pred = slr.predict(X_test)
# 学習済みのモデルから予測した結果を出力

y_test_pred
# df_testに SalePrice カラムを追加し、学習済みのモデルから予測した結果をセット

test["SalePrice"] = y_test_pred
# Id, SalePriceの2列だけ表示

test[["Id","SalePrice"]].head()
# Id, SalePriceの2列だけのファイルに変換

test[["Id","SalePrice"]].to_csv("submission.csv",index=False)



# Kaggleにcsvファイルを提出してスコア確認