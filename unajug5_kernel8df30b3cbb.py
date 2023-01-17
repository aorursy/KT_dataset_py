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
#import some necessary librairies

 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

from sklearn.preprocessing import PowerTransformer

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

 

 

from scipy import stats

from scipy.stats import norm, skew #for some statistics

 

 

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

 
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
#学習データの変数を確認

train.columns
#目的変数である家の価格の要約統計量を表示する

train["SalePrice"].describe()
#目的変数である家の価格のヒストグラムを表示する

plt.figure(figsize=(20, 10))

sns.distplot(train['SalePrice'])
#歪度と尖度を計算

print("歪度: %f" % train['SalePrice'].skew())

print("尖度: %f" % train['SalePrice'].kurt())
#相関ヒートマップ

k = 20 # number of variables for heatmap

corrmat = train.corr()

cols = corrmat.nlargest(k, "SalePrice")["SalePrice"].index

cm = np.corrcoef(train[cols].values.T)

fig, ax = plt.subplots(figsize=(12, 10))

sns.set(font_scale=1.2)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt=".2f", annot_kws={"size": 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()

fig.savefig("output2.png")
#相関グラフを表示

df = train

# 相関の高いものを合成

#物件の広さを合計した変数を作成

train["TotalSF"] = train["1stFlrSF"] + train["2ndFlrSF"] + train["TotalBsmtSF"]

test["TotalSF"] = test["1stFlrSF"] + test["2ndFlrSF"] + test["TotalBsmtSF"]

train["Garage"] = train["GarageCars"] + train["GarageArea"]

test["Garage"] = test["GarageCars"] + test["GarageArea"]

train["Year"] = train["YearBuilt"] + train["YearRemodAdd"]

test["Year"] = test["YearBuilt"] + test["YearRemodAdd"]



cols = ["GarageCars","GarageArea","1stFlrSF", "2ndFlrSF", "TotalBsmtSF", "YearBuilt", "YearRemodAdd"]

for i in cols:

    df = df.drop(i, axis=1)
# number of variables for heatmap

k = 21

fig = plt.figure(figsize=(20,20))

# 各列の間の相関係数算出

corrmat = df.corr()

# リストの最大値から順にk個の要素を取得

cols = corrmat.nlargest(k, "SalePrice")["SalePrice"].index

# NumPyの相関係数を求める(最初はSalePriceなので不要)

for i in np.arange(1,k):

    X_train = df[cols[i]]

    Y_train = df[cols[0]]

    ax = fig.add_subplot(5,4,i)

    # 図やサブプロットのうち一番最後のものに描画

    sns.regplot(x=X_train, y=Y_train)

plt.tight_layout()

plt.show()
# number of variables for heatmap

k = 31

fig = plt.figure(figsize=(20,20))

# 各列の間の相関係数算出

corrmat = train.corr()

# リストの最大値から順にk個の要素を取得

cols = corrmat.nlargest(k, "SalePrice")["SalePrice"].index

# NumPyの相関係数を求める(最初はSalePriceなので不要)

for i in np.arange(1,k):

    X_train = train[cols[i]]

    Y_train = train[cols[0]]

    ax = fig.add_subplot(6,5,i)

    # 図やサブプロットのうち一番最後のものに描画

    sns.regplot(x=X_train, y=Y_train)

plt.tight_layout()

plt.show()
#外れ値を除外する

train = train.drop(train[(train['TotalSF']>7500) & (train['SalePrice']<300000)].index)



train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index)



train = train.drop(train[(train['Year']<4000) & (train['SalePrice']>450000)].index)



train = train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index)

train = train.drop(train[(train['OverallQual']<10) & (train['SalePrice']>500000)].index)
#外れ値除外後

# number of variables for heatmap

k = 31

fig = plt.figure(figsize=(20,20))

# 各列の間の相関係数算出

corrmat = train.corr()

# リストの最大値から順にk個の要素を取得

cols = corrmat.nlargest(k, "SalePrice")["SalePrice"].index

# NumPyの相関係数を求める(最初はSalePriceなので不要)

for i in np.arange(1,k):

    X_train = train[cols[i]]

    Y_train = train[cols[0]]

    ax = fig.add_subplot(6,5,i)

    # 図やサブプロットのうち一番最後のものに描画

    sns.regplot(x=X_train, y=Y_train)

plt.tight_layout()

plt.show()
#学習データを目的変数とそれ以外に分ける

train_x = train.drop("SalePrice",axis=1)

train_y = train["SalePrice"]



#学習データとテストデータを統合

all_data = pd.concat([train_x,test],axis=0,sort=True)



#IDのカラムは不必要なので別の変数に格納

train_ID = train['Id']

test_ID = test['Id']



all_data.drop("Id", axis = 1, inplace = True)



#それぞれのデータのサイズを確認

print("train_x: "+str(train_x.shape))

print("train_y: "+str(train_y.shape))

print("test_x: "+str(test.shape))

print("all_data: "+str(all_data.shape))
#欠損値

# search for missing data

import missingno as msno

msno.matrix(df=train, figsize=(20,14), color=(0.5,0,0))
#欠損値率グラフ

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(20,10))

(train.isnull().mean()[train.isnull().sum() > 0]).abs().plot.bar(ax=ax)

fig.savefig("output1.png", dpi=300)
#欠損値率グラフ

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(20,10))

(test.isnull().mean()[test.isnull().sum() > 0]).abs().plot.bar(ax=ax)
#データの欠損値を確認する

all_data_na = all_data.isnull().sum()[all_data.isnull().sum()>0].sort_values(ascending=False)

all_data_na



#欠損値の数をグラフ化

plt.figure(figsize=(20,10))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)
# 欠損値があるカラムをリスト化

na_col_list = all_data.isnull().sum()[all_data.isnull().sum()>0].index.tolist()



#欠損があるカラムのデータ型を確認

all_data[na_col_list].dtypes.sort_values()
#隣接した道路の長さ（LotFrontage）の欠損値の補完

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



#欠損値が存在するかつfloat型のリストを作成

float_list = all_data[na_col_list].dtypes[all_data[na_col_list].dtypes == "float64"].index.tolist()



#欠損値が存在するかつobject型のリストを作成

obj_list = all_data[na_col_list].dtypes[all_data[na_col_list].dtypes == "object"].index.tolist()



#float型の場合は欠損値を0で置換

all_data[float_list] = all_data[float_list].fillna(0)



#object型の場合は欠損値を"None"で置換

all_data[obj_list] = all_data[obj_list].fillna("None")



#欠損値が全て置換できているか確認

all_data.isnull().sum()[all_data.isnull().sum() > 0]
# カテゴリ変数に変換する

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str) #住宅の種類

all_data['YrSold'] = all_data['YrSold'].astype(str) #販売年

all_data['MoSold'] = all_data['MoSold'].astype(str) #販売月
#目的変数の対数log(x+1)をとる

train_y = np.log1p(train_y)



#分布を可視化

plt.figure(figsize=(20, 10))

sns.distplot(train_y)
#数値の説明変数のリストを作成

num_feats = all_data.dtypes[all_data.dtypes != "object" ].index



#各説明変数の歪度を計算

skewed_feats = all_data[num_feats].apply(lambda x: x.skew()).sort_values(ascending = False)



#グラフ化

plt.figure(figsize=(20,10))

plt.xticks(rotation='90')

sns.barplot(x=skewed_feats.index, y=skewed_feats)
#歪度の絶対値が0.5より大きい変数だけに絞る

skewed_feats_over = skewed_feats[abs(skewed_feats) > 0.5].index



#各変数の最小値を表示

for i in skewed_feats_over:

    print(min(all_data[i]))



#Yeo-Johnson変換

pt = PowerTransformer()

pt.fit(all_data[skewed_feats_over])



#変換後のデータで各列を置換

all_data[skewed_feats_over] = pt.transform(all_data[skewed_feats_over])



#各説明変数の歪度を計算

skewed_feats_fixed = all_data[skewed_feats_over].apply(lambda x: x.skew()).sort_values(ascending = False)



#グラフ化

plt.figure(figsize=(20,10))

plt.xticks(rotation='90')

sns.barplot(x=skewed_feats_fixed.index, y=skewed_feats_fixed)
#各カラムのデータ型を確認

all_data.dtypes.value_counts()
#カテゴリ変数となっているカラムを取り出す

cal_list = all_data.dtypes[all_data.dtypes=="object"].index.tolist()



#学習データにおけるカテゴリ変数のデータ数を確認

train_x[cal_list].info()
#カテゴリ変数をget_dummiesによるone-hot-encodingを行う

all_data = pd.get_dummies(all_data,columns=cal_list)



#サイズを確認

all_data.shape
#学習データとテストデータに再分割

train_x = all_data.iloc[:train_x.shape[0],:].reset_index(drop=True)

test = all_data.iloc[train_x.shape[0]:,:].reset_index(drop=True)



#サイズを確認

print("train_x: "+str(train_x.shape))

print("test: "+str(test.shape))
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

import lightgbm as lgb
# データの分割

train_x, valid_x, train_y, valid_y = train_test_split(

        train_x,

        train_y,

        test_size=0.3,

        random_state=0)
#特徴量と目的変数をxgboostのデータ構造に変換する

dtrain = xgb.DMatrix(train_x, label=train_y)

dvalid = xgb.DMatrix(valid_x,label=valid_y)



#パラメータを指定してGBDT

num_round = 5000

evallist = [(dvalid, 'eval'), (dtrain, 'train')]



evals_result = {}



#パラメータ

param = {

            'max_depth': 3,

            'eta': 0.01,

            'objective': 'reg:squarederror',

}



#学習の実行

bst = xgb.train(

                        param, dtrain,

                        num_round,

                        evallist,

                        evals_result=evals_result,

                        # 一定ラウンド回しても改善が見込めない場合は学習を打ち切る

                        early_stopping_rounds=1000

)
#学習曲線を可視化する

plt.figure(figsize=(20, 10))

train_metric = evals_result['train']['rmse']

plt.plot(train_metric, label='train rmse')

eval_metric = evals_result['eval']['rmse']

plt.plot(eval_metric, label='eval rmse')

plt.grid()

plt.legend()

plt.xlabel('rounds')

plt.ylabel('rmse')

plt.ylim(0, 0.3)

plt.show()
#特徴量ごとの重要度を可視化する

ax = xgb.plot_importance(bst)

fig = ax.figure

fig.set_size_inches(20, 60)
dtest = xgb.DMatrix(test)

my_submission = pd.DataFrame()

my_submission["Id"] = test_ID

my_submission["SalePrice"] = np.exp(bst.predict(dtest))

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)