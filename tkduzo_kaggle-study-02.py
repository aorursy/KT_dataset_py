import numpy as np

import pandas as pd

pd.set_option('max_columns', 105)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

%matplotlib inline

sns.set()



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=DeprecationWarning)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# setting the number of cross validations used in the Model part 

nr_cv = 5



# target used for correlation 

use_log_val = True

if use_log_val:

    target = 'SalePrice_Log'

else:

    target = 'SalePrice'

    

# only columns with correlation above this threshold value

min_val_corr = 0.4
def get_best_score(grid):

    """

    グリッドサーチの結果から最も良いスコアを取得する

    

    Parameters

    ----------

    grid: GridSearchCV



    Returns

    -------

    best_score: 最も良いスコア

    """

    

    

    best_score = np.sqrt(-grid.best_score_)

    print("Best score:", best_score)    

    print("Best parameters:", grid.best_params_)

    print("Best estimator:", grid.best_estimator_)

    

    return best_score
def plot_corr_matrix(df, nr_c, targ) :

    """

    相関係数をヒートマップで可視化する

    

    Parameters

    ----------

    df: DataFrame

    nr_c: 説明変数のカラムの数

    targ: 目的変数のカラム

    """

    

    corr = df.corr()

    corr_abs = corr.abs()

    cols = corr_abs.nlargest(nr_c, targ)[targ].index

    cm = np.corrcoef(df[cols].values.T)



    plt.figure(figsize=(nr_c/1.5, nr_c/1.5))

    sns.set(font_scale=1.25)

    sns.heatmap(cm, linewidths=1.5, annot=True, square=True, 

                fmt='.2f', annot_kws={'size': 10}, 

                yticklabels=cols.values, xticklabels=cols.values

               )

    plt.show()
# csvファイルをDataFrameとして読み込む

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
# データサイズを確認

print("Train data:", df_train.shape)

print(" Test data:", df_test.shape)
# 要約統計量を確認

df_train.describe()
# SalePrice(目的変数)の分布を確認

sns.distplot(df_train['SalePrice']);

print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
# SalePriceの対数値をSalePrice_Logとして訓練データに追加

df_train['SalePrice_Log'] = np.log(df_train['SalePrice'])



# SalePrice_Logの分布を確認

sns.distplot(df_train['SalePrice_Log']);

print("Skewness: %f" % df_train['SalePrice_Log'].skew())

print("Kurtosis: %f" % df_train['SalePrice_Log'].kurt())



# SalePriceのデータを削除

df_train.drop('SalePrice', axis= 1, inplace=True)
# 数値データとカテゴリデータのインデックスを取得

numerical_feats = df_train.dtypes[df_train.dtypes != "object"].index

print("Number of Numerical features: ", len(numerical_feats))



categorical_feats = df_train.dtypes[df_train.dtypes == "object"].index

print("Number of Categorical features: ", len(categorical_feats))
# 数値データの確認

df_train[numerical_feats].head()
# 欠損値の割合を確認

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
# 欠損値(NaN)をNoneに置換 (選択項目は他カーネルを参考)

cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',

               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',

               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',

               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',

               'MSZoning', 'Utilities']

for col in cols_fillna:

    df_train[col].fillna('None',inplace=True)

    df_test[col].fillna('None',inplace=True)
# 残りの欠損値の割合を確認

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(5)
# 残りの項目の欠損値(NaN)を平均値に置換

df_train.fillna(df_train.mean(), inplace=True)

df_test.fillna(df_test.mean(), inplace=True)
# 残りの欠損値の割合を確認

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(5)
# 訓練データとテストデータに欠損値が含まれないことを確認

print("Train data has missing values: ", df_train.isnull().sum().sum())

print("Test  data has missing values: ", df_test.isnull().sum().sum())
###

# 説明変数と目的変数の相関性を可視化する

###



# subplotの数

nr_rows = 12

nr_cols = 3



# グラフ用のデータを作成

li_num_feats = list(numerical_feats)

li_not_plot = ['Id', 'SalePrice_Log']

li_plot_num_feats = [c for c in list(numerical_feats) if c not in li_not_plot]



# グラフの作成

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))



for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(li_plot_num_feats):

            sns.regplot(df_train[li_plot_num_feats[i]], df_train[target], ax = axs[r][c])

            stp = stats.pearsonr(df_train[li_plot_num_feats[i]], df_train[target])

            str_title = "r = " + "{0:.2f}".format(stp[0]) + "      " "p = " + "{0:.2f}".format(stp[1])

            axs[r][c].set_title(str_title,fontsize=11)

            

plt.tight_layout()    

plt.show()   
# 目的変数と相関が高い説明変数を確認

corr = df_train.corr()

corr_abs = corr.abs()



nr_num_cols = len(numerical_feats)

ser_corr = corr_abs.nlargest(nr_num_cols, target)[target]



cols_abv_corr_limit = list(ser_corr[ser_corr.values > min_val_corr].index)

cols_bel_corr_limit = list(ser_corr[ser_corr.values <= min_val_corr].index)



print(ser_corr)

print("*"*30)

print("List of numerical features with r above min_val_corr :")

print(cols_abv_corr_limit)

print("*"*30)

print("List of numerical features with r below min_val_corr :")

print(cols_bel_corr_limit)
# 目的変数との相関が高い説明変数を用いて相関行列を作成する

nr_feats = len(cols_abv_corr_limit)

plot_corr_matrix(df_train, nr_feats, target)
# モデル構築に使用するカラムを定義 (目的変数との相関係数が0.5以上の変数を使用)

feats = [

    'OverallQual', 

    'GrLivArea',

    'YearBuilt', 

    'YearRemodAdd', 

    'TotalBsmtSF', 

    '1stFlrSF', 

    'FullBath', 

    'TotRmsAbvGrd', 

    'GarageCars', 

    'GarageArea'

]
df_train_feat = df_train[feats].copy()

df_test_feat  = df_test[feats].copy()
# 正規化を行い、特徴量間のスケールを揃える

from sklearn.preprocessing import StandardScaler



sc_feat = StandardScaler()

df_train_feat_sc = sc_feat.fit_transform(df_train_feat)

df_test_feat_sc = sc_feat.transform(df_test_feat)



sc_target = StandardScaler()

df_train_target_sc = sc_target.fit_transform(df_train[target].values.reshape(-1, 1))
# 正規化したデータを確認

df_train_feat_sc = pd.DataFrame(df_train_feat_sc)

df_test_feat_sc = pd.DataFrame(df_test_feat_sc)



df_train_feat_sc.head()
df_train_target_sc = pd.DataFrame(df_train_target_sc)

df_train_target_sc.head()
# 訓練データ・テストデータを作成

X = df_train_feat.copy()

y = df_train[target]

X_test = df_test_feat.copy()



X_sc = df_train_feat_sc.copy()

y_sc = df_train_target_sc.copy()

X_test_sc = df_test_feat_sc.copy()



X.info()

X_test.info()
X.head()
y.head()
X_sc.head()
y_sc.head()
X_test.head()
X_test_sc.head()
# グリッドサーチを用いてパラメータ探索を行う

from sklearn.model_selection import GridSearchCV

score_calc = 'neg_mean_squared_error' # 平均二乗誤差
from sklearn.linear_model import LinearRegression



# 線形回帰モデルのパラメータ探索

linreg = LinearRegression()

parameters = {'fit_intercept':[True,False]}

grid_linear = GridSearchCV(linreg, parameters, cv=nr_cv, verbose=1, scoring=score_calc)

grid_linear.fit(X, y)



bscore_linear = get_best_score(grid_linear)
# 線形回帰モデル（正規化した説明変数で訓練）のパラメータ探索

linreg_sc = LinearRegression()

parameters = {'fit_intercept':[True,False]}

grid_linear_sc = GridSearchCV(linreg_sc, parameters, cv=nr_cv, verbose=1, scoring=score_calc)

grid_linear_sc.fit(X_sc, y)



bscore_linear_sc = get_best_score(grid_linear_sc)
# ベストスコアで線形回帰モデルを訓練する

linregr_all = LinearRegression(bscore_linear)

linregr_all.fit(X, y)



# テストデータに対して予測する

pred_linreg_all = linregr_all.predict(X_test)



# 予測結果の対数を外す

pred_linreg_all = np.exp(pred_linreg_all)



# 負の予測結果を平均値で埋める

pred_linreg_all[pred_linreg_all < 0] = pred_linreg_all.mean()
linregr_all.coef_
# submit用のファイルを作成

sub_linreg = pd.DataFrame()

sub_linreg['Id'] = df_test['Id']

sub_linreg['SalePrice'] = pred_linreg_all

sub_linreg.to_csv('linreg.csv',index=False)
sub_linreg.head()
# ベストスコアで線形回帰モデルを訓練する

linregr_all_sc = LinearRegression(bscore_linear_sc)

linregr_all_sc.fit(X_sc, y)



# テストデータに対して予測する

pred_linreg_all_sc = linregr_all_sc.predict(X_test_sc)



# 予測結果の対数を外す

pred_linreg_all_sc = np.exp(pred_linreg_all_sc)



# 負の予測結果を平均値で埋める

pred_linreg_all_sc[pred_linreg_all_sc < 0] = pred_linreg_all_sc.mean()
linregr_all_sc.coef_
# submit用のファイルを作成

sub_linreg_sc = pd.DataFrame()

sub_linreg_sc['Id'] = df_test['Id']

sub_linreg_sc['SalePrice'] = pred_linreg_all_sc

sub_linreg_sc.to_csv('linreg_sc.csv',index=False)
sub_linreg_sc.head()
from sklearn.linear_model import Ridge



# Ridge回帰モデルのパラメータ探索

ridge = Ridge()

parameters = {'alpha':np.logspace(-3, 5, 10), 'tol':[1e-06,5e-06,1e-05,5e-05]}

grid_ridge = GridSearchCV(ridge, parameters, cv=nr_cv, verbose=1, scoring = score_calc)

grid_ridge.fit(X, y)



bscore_ridge = get_best_score(grid_ridge)
# Ridge回帰モデル（正規化した説明変数を使用）のパラメータ探索

ridge_sc = Ridge()

parameters = {'alpha':np.logspace(-3, 5, 10), 'tol':[1e-06,5e-06,1e-05,5e-05]}

grid_ridge_sc = GridSearchCV(ridge_sc, parameters, cv=nr_cv, verbose=1, scoring=score_calc)

grid_ridge_sc.fit(X_sc, y)



bscore_ridge_sc = get_best_score(grid_ridge_sc)
# Ridge回帰モデルを訓練する

ridge_all = Ridge(bscore_ridge)

ridge_all.fit(X, y)



# テストデータに対して予測する

pred_ridge_all = ridge_all.predict(X_test)



# 予測結果の対数を外す

pred_ridge_all = np.exp(pred_ridge_all)



# 負の予測結果を平均値で埋める

pred_ridge_all[pred_ridge_all < 0] = pred_ridge_all.mean()
ridge_all.coef_
# submit用のファイルを作成

sub_ridge = pd.DataFrame()

sub_ridge['Id'] = df_test['Id']

sub_ridge['SalePrice'] = pred_ridge_all

sub_ridge.to_csv('ridge.csv',index=False)
sub_ridge.head()
# 正規化した説明変数を用いてRidge回帰モデルを訓練する

ridge_all_sc = Ridge(bscore_ridge_sc)

ridge_all_sc.fit(X_sc, y)



# テストデータに対して予測する

pred_ridge_all_sc = ridge_all_sc.predict(X_test_sc)



# 予測結果の対数を外す

pred_ridge_all_sc = np.exp(pred_ridge_all_sc)



# 負の予測結果を平均値で埋める

pred_ridge_all_sc[pred_ridge_all_sc < 0] = pred_ridge_all_sc.mean()
ridge_all_sc.coef_
# submit用のファイルを作成

sub_ridge_sc = pd.DataFrame()

sub_ridge_sc['Id'] = df_test['Id']

sub_ridge_sc['SalePrice'] = pred_ridge_all_sc

sub_ridge_sc.to_csv('ridge_sc.csv',index=False)
sub_ridge_sc.head()
ll