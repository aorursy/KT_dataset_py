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
# 使用するモジュールをimportする

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

# scipyのstatsという統計関数をまとめたモジュール # norm は正規分布を扱う

from scipy.stats import norm

from scipy import stats

# 標準化のスケール変換

from sklearn.preprocessing import StandardScaler

# 警告の制御

import warnings

warnings.filterwarnings('ignore')

# notebook内に描写 plt.show()を省略できる

%matplotlib inline
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_train.columns
# 要約統計量

df_train['SalePrice'].describe()
# ヒストグラム

sns.distplot(df_train['SalePrice']);
# 歪度と尖度(ワイド/センド)

print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
# axis 1 横方向の結合

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.head()
# axis 0 縦方向の結合

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=0)

data.head()
#散布図 grlivarea/saleprice

# GrLivArea・・・リビングエリアの平方フィート(Above grade (ground) living area square feet)

var = 'GrLivArea'

# 横連結

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

# y軸の範囲を変更しつつ

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# TotalBsmtSF・・・地下室の総平方フィート(Total square feet of basement area)

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000));
# 箱ひげ図 boxplot

# OverallQual・・・全体的な材料と仕上げの品質(Overall material and finish quality)

var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

# Figureオブジェクトとそれに属する一つのAxesオブジェクトを同時に作成 figsizeは、図のサイズ

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y='SalePrice', data=data)

fig.axis(ymin=0, ymax=800000);
# YearBuilt・・・建設日(Original construction date)

var = 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(16,8))

fig = sns.boxplot(x=var, y='SalePrice', data=data)

fig.axis(ymin=0, ymax=800000);

# xのラベルテキストを回転させる

plt.xticks(rotation=90);
#correlation matrix

# 各列の間の相関係数が算出

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

# 第一引数・・Pandas のデータフレームを指定した場合、列名とインデックスが使用される

# vmax・・・カラーマップと値の範囲を関連付ける必要がある際に最小値、最大値を指定。指定しない場合、データや他の引数から推測

# square・・・True に設定すると、X 軸、Y 軸を正方形になるように調整し、軸を設定

sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix

k = 10 # ヒートマップで使う変数の数

# 目的変数(SalePrice)と相関係数が高い9個に絞る(nlargest・・・列の値が最も大きい最初のn行を降順で返す)

# SalePriceの各列名を取得

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

# numpyの相関を求めるcorrcoef() 行と列を入れ替える転値したDataFrameを入れている。

cm = np.corrcoef(df_train[cols].values.T)

# fontのスケール(大きさ)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# わかりづらかったところ

k = 10

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index



df_train[cols][:3]
# 行ごとに、配列に格納されている。（配列内の値は、各変数の値が一つずつ）

df_train[cols].values[:3]
# 列ごとに、配列に格納（配列内の値は、一つの変数に対する全ての値）

df_train[cols].values.T[:3]
cm = np.corrcoef(df_train[cols].values.T)

cm
# scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], size=2.5);
# missing data

# 要素をソートする・・・引数 ascending 昇順、降順

total = df_train.isnull().sum().sort_values(ascending=False)

# 全体のうち、欠損値は何割か

percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
(missing_data[missing_data['Total'] > 1]).index
df_train.loc[df_train['Electrical'].isnull()]
# dealing with missing data

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index, 1) # ここの「1」はなに？引数？ axis=1の略(列を指定削除)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index) # index=1379の行を削除

# チェック

df_train.isnull().sum().max()
df_train['SalePrice'] # 一次元 df_train['SalePrice'][:] も同じ意味 
df_train['SalePrice'][:,np.newaxis] # 二次元 208500を指定する場合、[0,0]
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);

saleprice_scaled
# 特定の列を基準にソート

# np.argsort()で基準となる行または列のインデックスを取得。



## print(a_2d)

# [[ 20   3 100]

#  [  1 200  30]

#  [300  10   2]]



## print(a_2d[:, 1]) → これが、やりたいから二元配列にしている。NumPyの、ndarrayの方が、標準化をする上で扱いやすいから。

# [  3 200  10]
saleprice_scaled[:,0]
# NumPyの関数numpy.sort()を2次元のNumPy配列ndarrayに適用すると、各行・各列の値が別々に昇順にソートされたndarrayを取得

# 特定の行または列を基準にソートしたい場合はnumpy.argsort()を使う。numpy.argsort()はソートされた値ではなくインデックスのndarrayを返す関数。

saleprice_scaled[:,0].argsort()
# standardizing data データの標準化

# NumPy配列ndarrayに新たな次元を追加する np.newaxis

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10] # 最初のindexは0

high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:] # 最後のindexは-1



print('outer range (low) of the distribution:')

print(low_range)

print('outer range (high) of the distribution:')

print(high_range)
# bivariate analysis saleprice/grlivarea

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000));
# deleting points

# ソートしたい列のラベル（列名）を第一引数（by）に指定、Falseで降順

# 上から2つ目が、削除するデータ

df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id']  == 1299].index)

df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
#bivariate analysis saleprice/grlivarea

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# ここでは、GrLivArea、TotalBsmtSFの2つしか二変量分析をしていなかったが、他の61つの特徴量についても行うのか？？

len(df_train.columns)
#histogram and normal probability plot ヒストグラムと正規確率プロット

# カーネル密度推定・・・データから推定した確率密度関数を描写しており、データの背後にあると想定される確率分布が可視化できる.

# fit = norm 正規分布

sns.distplot(df_train['SalePrice'], fit=norm);



# 新規のウィンドウを描画

fig = plt.figure()

# Q-Qプロット・・・観測値がある確率分布に従う場合の期待値と観測値のデータを２次元に表示したもの.

# データの青いプロットが、赤いラインに沿えばこのデータが正規分布に従っていると判断できる.

res = stats.probplot(df_train['SalePrice'], plot=plt)
#applying log transformation ログ変換の適用

# 底をeとするdf_train['SalePrice']の対数

df_train['SalePrice'] = np.log(df_train['SalePrice'])
#transformed histogram and normal probability plot 変換されたヒストグラムと正規確率プロット

sns.distplot(df_train['SalePrice'], fit=norm);



fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
#histogram and normal probability plot

# fit = norm 正規分布

sns.distplot(df_train['GrLivArea'], fit=norm)



fig = plt.figure()

# scipyモジュールのstats

res = stats.probplot(df_train['GrLivArea'], plot=plt)
# 対数変換

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
#transformed histogram and normal probability plot

sns.distplot(df_train['GrLivArea'], fit=norm);



fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
#histogram and normal probability plot

sns.distplot(df_train['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
# バイナリ変数を作成

# area > 0 だったら、1

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)

df_train['HasBsmt'] = 0

df_train.loc[df_train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
# 対数変換

df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
#histogram and normal probability plot

sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);



fig = plt.figure()

res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
# scatter plot

plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);
# scatter plot

plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);
df_train
# カテゴリ変数をダミーに変換する・・・数字ではない変数を数字に変換する

df_train = pd.get_dummies(df_train)

df_train