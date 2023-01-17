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

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

from scipy.special import boxcox1p



from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# 処理まえのデータの形状 shape・・・行数/列数を取得

print('Shape of Train set: {}'.format(df_train.shape))

print('Shape of Test set: {}'.format(df_test.shape))



# 後のモデリングのために保存するらしい

test_id = df_test['Id']

      

# 目的変数(Id)をドロップ

df_train.drop("Id", axis=1, inplace=True)

df_test.drop("Id", axis=1, inplace=True)

      

print('Shape of Train set after removing Id: {}'.format(df_train.shape))

print('Shape of Test set after removing Id: {}'.format(df_test.shape))
original = df_train['SalePrice']



# 線形回帰分析→データが正規分布に従うことを仮定している。

# Box-Cox変換・・・データの構造を正規分布に変換。この変換は変数に0や負の値が含まれている場合はうまく推定できない

df_train['SalePrice'] = stats.boxcox(df_train['SalePrice'])[0]



# フィットするパラメータ用 　norm.fit・・・ガウシアンフィッティング(正規分布に対する近似曲線（フィッティングカーブ）の関数を求める)

(mu, sigma) = norm.fit(df_train['SalePrice'])

(mu_norm, sigma_norm) = norm.fit(df_train['SalePrice'])



# 歪度と尖度

original_skew = original.skew()

original_kurt = original.kurt()



# 歪度と尖度（Box-Cox変換後）

transf_skew = df_train['SalePrice'].skew()

transf_kurt = df_train['SalePrice'].kurt()







plt.figure(figsize=(18,8))





# plot the distribution

# subplot 一つの描画キャンパスを複数の領域に分割 描画を2行2列に分割し、3番目の引数に、これからグラフを描くのに利用するサブ領域の番号

plt.subplot(221)

sns.distplot(original, fit=norm)



plt.legend(['Normal dist. \n$\mu=$ {:.2f} \n $\sigma=$ {:.2f} \n skewness= {:.2f} \n kurtosis= {:.2f}'.format(mu, sigma,original_skew,original_kurt)],loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice density distribution')



# QQ-plot (確率プロット)

plt.subplot(222)

res = stats.probplot(original, sparams=(2.5,), plot=plt, rvalue=True)

plt.title('SalePrice Probability Plot')



# plot Normalized distribution

plt.subplot(223)

sns.distplot(df_train['SalePrice'], fit=norm)



plt.legend(['Normal dist. \n$\mu=$ {:.2f} \n $\sigma=$ {:.2f} \n skewness= {:.2f} \n kurtosis= {:.2f}'.format(mu_norm, sigma_norm,transf_skew,transf_kurt)],loc='best')

plt.ylabel('Frequency')

plt.title('Normalized SalePrice density distribution')



# QQ-plot (変換後)

plt.subplot(224)

res = stats.probplot(df_train['SalePrice'], sparams=(2.5,), plot=plt, rvalue=True)

plt.title('Normalized SalePrice Probability Plot')

plt.tight_layout()
corrmat = df_train.corr()

corrmat
f, ax = plt.subplots(figsize=(16,8))

sns.heatmap(corrmat, vmax=.8, square=True, cmap='coolwarm'); # square=Trueで、x,y軸を正方形
# SalePriceと相関の高い10個の特徴量を見てみる

k = 10

# SalePriceと相関の高い10のカラムを取得

cols =corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

# 相関係数を求める

cm = np.corrcoef(df_train[cols].values.T) # df_train[cols].values.T ・・・ カラムの列ごとに値をまとめる



sns.set(font_scale=12)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True)
plt.figure(figsize=(16,8))



plt.subplot(221)

sns.scatterplot(x=df_train['GrLivArea'], y=df_train['SalePrice'])

plt.title('Original Data:\nGrLivArea vs SalePrice')



# 2つの外れ値を取り除く ascending False=降順

df_train = df_train.drop(df_train.sort_values(by = 'GrLivArea', ascending=False)[:2].index)



plt.subplot(222)

sns.scatterplot(x=df_train['GrLivArea'], y=df_train['SalePrice'])

plt.title('Outliers removed:\nGrLivArea vs SalePrice')



# Plotの座標部分が重なるのを、重ならないようにする

plt.tight_layout()
# Combining Train and Test data sets

all_data = pd.concat((df_train, df_test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)



# df_train/testの行数を取得

ntrain = df_train.shape[0] # 新しいtrainデータセット作成用

ntest = df_test.shape[0]



y_train = df_train['SalePrice'].values

y_test = df_train['SalePrice'].values
def report_missing_data(df):

    '''

    IN: Dataframe 

    OUT: Dataframe with reported count of missing values, % missing per column and per total data

    '''

    

    missing_count_per_column = df.isnull().sum() # 欠損値の数

    missing_count_per_column = missing_count_per_column[missing_count_per_column > 0]

    

    total_count_per_column = df.isnull().count() # 欠損値でない数

    total_cells = np.product(df.shape) # shapeで行数列数取得 np.productで行*列=セルの合計数

    

    # パーセント計算

    percent_per_column = 100 * missing_count_per_column / total_count_per_column

    percent_of_total = 100 * missing_count_per_column / total_cells

    

    # Creating new dataframe for reporting purposes only

    # axis=1, 横方向に連結

    # keys引数で各々のデータに対してつけるラベルを指定 ・・・被りのあるラベル(インデックス)が存在しても判別可能

    missing_data = pd.concat([missing_count_per_column, percent_per_column, percent_of_total], axis=1, keys=['Total_Missing', 'Percent_per_column','Percent_of_total'])

    

    missing_data = missing_data.dropna()

    missing_data.index.names = ['Feature']

    missing_data.reset_index(inplace=True)

    

    return missing_data.sort_values(by='Total_Missing', ascending=False)



df = report_missing_data(all_data)
plt.figure(figsize=(18,15))

plt.subplot(221)

sns.barplot(y='Feature', x='Total_Missing', data=df);

plt.title('Missing Data');
# 欠損値の扱う方針



Cat_toNone = ('PoolQC','Fence','MiscFeature','Alley','FireplaceQu','GarageType','GarageFinish',

              'GarageQual','GarageCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

              'BsmtFinType2','MasVnrType','MSSubClass')

    

Cat_toMode = ('MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd',

              'Utilities','SaleType','Functional')



Num_toZero = ('GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',

              'BsmtFullBath','BsmtHalfBath','GarageYrBlt','MasVnrArea')



for col in Cat_toNone:

    all_data[col] = all_data[col].fillna('None')

    

for col in Cat_toMode:

    all_data[col] = all_data[col].fillna(all_data[col].mode()[0]) # modeは最頻値

    

for col in Num_toZero:

    all_data[col] = all_data[col].fillna(0)
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

report_missing_data(all_data)
all_data._get_numeric_data() # 数値型の列のみ取り出すメソッド

all_data['Total_SF'] = all_data['1stFlrSF']+all_data['2ndFlrSF']+all_data['TotalBsmtSF']+all_data['GarageArea']+all_data['GrLivArea']+all_data['PoolArea']+all_data['WoodDeckSF']
def bcox_transform(df):

    '''

    IN: Original dataframe 

    OUT: Dataframe with box-cox normalized numerical values. 

    Specified skewness threshold 1 and -1 

    '''

    

    lam = 0.15

    for feat in df._get_numeric_data():

        if df[feat].skew() > 1.0 or df[feat].skew() < -1.0:

            df[feat] = boxcox1p(df[feat], lam)

    return df



all_data = bcox_transform(all_data)
all_data.head()
# ダミー変数

all_data = pd.get_dummies(all_data)
all_data.head()
# データを分割してtrainとtestセットに戻す

df_train = all_data[:ntrain]

df_test = all_data[ntrain:]
X_train = all_data[:df_train.shape[0]]

X_test = all_data[df_train.shape[0]:]

y = y_train
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score # クロスバリデーションを行う



n_folds = 15 # データを15分割する・・検定は15回繰り返される



# K-fold Root Mean Square Error Cross Validation

def k_rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(df_train.values) # テスト回数の取得(df_trainのvaluesの数)

    # sqrt 平方根を返す modelで分類機を指定、X特徴、Y目的、CV＝データを何分割するか

    # 層化k分割交差検証という手法を使って偏りの無いようにスコアを計算し、性能評価をする

    # 平均二乗偏差・・RMSE が 0 に近いほど見積もられる予測誤差が小さい、すなわち予測精度が高い

    rmse = np.sqrt(-cross_val_score(model, df_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return (rmse)
# モデル定義

model_ridge = Ridge()

model_lasso = Lasso()

model_elasticNet = ElasticNet(l1_ratio = 0.5)



# αはL1およびL2ノルムのペナルティの大きさを示しており、αが大きいほど正則化（ペナルティ具合）が大きくなります。

# l1_ratioは、L1, L2ペナルティのうち、L1の割合を示しています。



# 正則化の程度を制御するにはαというハイパーパラメーターを変化

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]



# k分割交差検証によるモデルの評価

cv_ridge = [k_rmsle_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]

cv_lasso = [k_rmsle_cv(Lasso(alpha = alpha)).mean() for alpha in alphas]

cv_elasticNet = [k_rmsle_cv(ElasticNet(alpha = alpha)).mean() for alpha in alphas]



cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_lasso = pd.Series(cv_lasso, index = alphas)

cv_elasticNet = pd.Series(cv_elasticNet, index = alphas)
cv_ridge
plt.figure(figsize=(20,10))



plt.subplot(241)

cv_ridge.plot(title = 'Ridge Validation Curve')

plt.xlabel('alpha')

plt.ylabel('rmse')



plt.subplot(242)

cv_lasso.plot(title = 'Lasso Validation Curve')

plt.xlabel('alpha')

plt.ylabel('rmse')



plt.subplot(243)

cv_elasticNet.plot(title = 'ElasticNet Validation Curve')

plt.xlabel('alpha')

plt.ylabel('rmse')



plt.subplot(244)

cv_ridge.plot()

cv_lasso.plot()

cv_elasticNet.plot()

plt.legend(labels=['Ridge', 'Lasso', 'ElasticNet'])

plt.title('Models Validation Curves')

plt.xlabel('alpha')

plt.ylabel('rmse')



# 重ならないように

plt.tight_layout()
model_ridge = Ridge(alpha=10)
model_ridge.fit(df_train, y_train)

model_ridge_pred = model_ridge.predict(df_train)



ridge_pred = np.expm1(model_ridge.predict(df_test.values)) #expm1(x)は、eのx乗から1を引いた値を返す
model_ridge_pred
ridge_pred
submission = pd.DataFrame()

submission['Id'] = test_id

submission['SalePrice'] = ridge_pred

submission.to_csv('house_pricing_submission.csv',index=False)