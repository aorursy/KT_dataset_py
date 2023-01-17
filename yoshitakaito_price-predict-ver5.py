import numpy as np
import pandas as pd
from scipy import stats
import collections
import itertools
import math
from scipy.stats import norm, skew
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('whitegrid')

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from category_encoders import BinaryEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import optuna.integration.lightgbm as lgb_optuna
import optuna
import xgboost as xgb
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
# テストデータ取り出し
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# 従属変数取り出し
train_SalePrice = train_df['SalePrice']
# 中身確認
train_df.head(5)
# Id は不要なので、削除して別に変数化
train_Id = train_df.Id
test_Id = test_df.Id

# Id列削除
train_df.drop('Id', axis=1, inplace=True)
test_df.drop('Id', axis=1, inplace=True)
# 変数の数が多いので、変数の意味を参照できる辞書を作成
fea_dic = {'MSSubClass': '住宅の種類',
           'MSZoning': '土地の種類',
           'LotFrontage': '住宅から通りまでの距離',
           'LotArea': '土地面積',
           'Street': '住宅までの道路タイプ（舗装かどうか）',
           'Alley': '住宅までの小道のタイプ（舗装かとか小道がないとか）',
           'LotShape': '住宅タイプ（標準的な形、とか変わった形とか）',
           'Landcontour': '住宅地の位置環境（平らな場所にあるか、丘にあるか、傾斜地にあるかとか）',
           'Utilities': 'ガスとか電気とかの設備状況',
           'LotConfig': '土地の区画位置（角とか）',
           'LandSlope': '土地の傾斜状況',
           'Neighborhood': '物件の場所',
           'Condition1': 'その他条件１（どの通りに面しているかとか）',
           'Condition2': 'その他条件２（どの通りに面しているかとか）',
           'BldgType': '住宅タイプ（１世帯用とか、２世帯用とか）',
           'HouseStyle': '住宅の形状（２階建てとか）',
           'OverallQual': '住宅設備の修繕状況',
           'OverallCond': '住宅自体の修繕状況',
           'YearBuilt': '建築年',
           'YearRemodAdd': '増築とか改築した場合の年',
           'RoofStyle7': '屋根の種類',
           'RoofMatl': '屋根の材質の種類',
           'Exterior1st': '住宅の外壁の種類',
           'Exterior2nd': '住宅の外壁の種類（２種類使っている場合）',
           'MasVnrType': '外壁レンガの種類',
           'MasVnrArea': 'レンガエリアの面積',
           'ExterQual': '屋外設備のクオリティ',
           'ExterCond': '屋外設備の状態',
           'Foundation': '住宅の基礎の種類',
           'BsmtQual': '地下室の高さ',
           'BsmtCond': '地下室の状態',
           'BsmtExposure': '庭の見晴らし具合？',
           'BsmtFinType1': '地下室の建築状況１（リビングルーム用になっているか、とかレクリエーションルーム用とか）',
           'BsmtFinSF1': '地下室の工事済み面積1',
           'BsmtFinType2': '地下室の建築状況２（リビングルーム用になっているか、とかレクリエーションルーム用とか）',
           'BsmtFinSF2': '地下室の工事済み面積2',
           'BsmtUnfSF': '地下室で建築（カーペットの設置とか）がされていない面積',
           'TotalBsmtSF': '地下室の総面積',
           'Heating': '暖房器具の種類',
           'HeatingQC':'暖房器具のコンディション',
           'CentralAir': 'セントラルエアーコンディショナーが付いているか',
           'Electrical': '電気システムの種類',
           '1stFlrSF': '１階の面積',
           '2ndFlrSF': '２階の面積',
           'LowQualFinSF': 'フロア全体で内装の品質が低いエリアの総面積',
           'GrLivArea': 'Above grade living area(居住するのに十分な内装や設備が住んだ部屋？）の面積',
           'BsmtFullBath': '地下にある full bathroomの数',
           'BsmtHalfBath': '地下にある half bathroomの数',
           'FullBath': '地上階にある full bathroomの数',
           'HalfBath': '地上階にある half bathroomの数',
           'BedroomAbvGr': '地上階にあるベッドルームの数',
           'KitchenAbvGr': '地上階にある台所の数',
           'KitchenQual': '台所のコンディション',
           'TotRmsAbvGrd': '地上階の総部屋数',
           'Functional': '住宅の機能性のランク',
           'Fireplaces': '暖炉の数',
           'FireplaceQu': '暖炉のコンディション',
           'GarageType': 'ガレージの種類（地下にあるとか、家にくっついているとか）',
           'GarageYrBlt': 'ガレージの建築年',
           'GarageFinish': 'ガレージの内装が終わっているか',
           'GarageCars': 'ガレージに入る車の台数',
           'GarageArea': 'ガレージの面積',
           'GarageQual': 'ガレージの品質',
           'GarageCond': 'ガレージのコンディション',
           'PavedDrive': '敷地内の道路の舗装状況',
           'WoodDeckSF': 'ウッドデッキエリアの面積',
           'OpenPorchSF': '屋外の屋根のあるテラスの面積',
           'EnclosedPorch': '縁側（外に突き出したリビングルーム）の面積',
           '3SsnPorch': '冷暖対応のポーチの面積',
           'ScreenPorch': 'scrennポーチの面積',
           'PoolArea': 'プールの面積',
           'PoolQC': 'プールのクオリティ',
           'Fence': 'フェンスのクオリティ',
           'MiscFeature': 'その他特徴（エレベーター付きとか）',
           'MiscVal': 'その他特徴のドル価値',
           'MoSold': '売りに出された月',
           'YrSold': '売りに出された年',
           'SaleType': '販売タイプ',
           'SaleCondition': '販売状況',
           'SalePrice': '住宅価格（※予測対象）'}
# null値のある変数の確認
null_col_train = ''
null_col_test = ''
null_train_list = []
null_test_list = []

for col in train_df.columns:
    check = train_df[col].isnull().value_counts()
    if len(check) != 1:
        null_col_train = null_col_train + col + ', '
        null_train_list.append(col)
        
for col in test_df.columns:
    check = test_df[col].isnull().value_counts()
    if len(check) != 1:
        null_col_test = null_col_test + col + ', '
        null_test_list.append(col)
        
print('訓練データのnull値あり変数：\n', null_col_train, '\n')
print('テストデータのnull値あり変数：\n', null_col_test)
# object型変数(Categorical)の抽出
object_list = train_df.select_dtypes(include='object').columns.to_list()

# int型変数(Categorical変数とNumerical変数が混ざっている)の抽出
int_list = train_df.select_dtypes(include='int').columns.to_list()

# float型変数(Numerical変数)の抽出
numeric_list = train_df.select_dtypes(include='float').columns.to_list()
# ojbect_listを名義変数リストと順序変数リストに分ける

# 名義変数
nominal_list = ['MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood',
                'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
                'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
                'GarageType', 'PavedDrive', 'SaleType', 'SaleCondition',
                'Street', 'Alley', 'Condition2', 'Heating', 'MiscFeature']
# 順序変数
ordinal_list = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical',
                'KitchenQual', 'Functional', 'GarageFinish','GarageQual', 'GarageCond',
                'Utilities', 'LandSlope', 'CentralAir', 'FireplaceQu', 'PoolQC',
                'Fence' ]
# Categorical-name変数
to_nominal_list = ['MSSubClass', 'BsmtFullBath', 'BsmtHalfBath', 'KitchenAbvGr',
                   'GarageCars']

# Categorical-ordinal変数
to_ordinal_list = ['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 
                   'MoSold', 'YrSold']

# numerical変数
to_num_list = ['LotArea', '1stFlrSF', '2ndFlrSF',
               'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr',
               'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF',
               'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
               'MiscVal', 'BsmtFinSF1', 'BsmtFinSF2',
               'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea', ]
numeric_list
numeric_list.remove('GarageYrBlt')
print("modified numeric_list: ",numeric_list)

to_ordinal_list.append('GarageYrBlt')
# 名義変数
all_nominal_list = nominal_list + to_nominal_list
# 順序変数
all_ordinal_list = ordinal_list + to_ordinal_list
# 量的変数
all_numeric_list = numeric_list + to_num_list
train_df.SalePrice.describe()
fig = plt.figure(figsize=(10, 4))
# ヒストグラム
sns.distplot(train_df['SalePrice'])
fig = plt.figure(figsize=(25, 25))
plt.subplots_adjust(hspace=0.6, wspace=0.4)
for i in range(len(all_nominal_list)):
    ax = fig.add_subplot(6, 5, i+1)
    sns.countplot(x=all_nominal_list[i], data=train_df, ax=ax)
plt.show()
# 可視化して分布状況確認（訓練データ・テストデータ合算の分布）
fig = plt.figure(figsize=(20,20))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(len(all_ordinal_list)):
    ax = fig.add_subplot(6, 5, i+1)
    sns.countplot(x=all_ordinal_list[i], data=train_df, ax=ax)
plt.show()
# ヒストグラムに可視化して分布状況確認
fig = plt.figure(figsize=(20,20))
plt.subplots_adjust(hspace=0.4,wspace=0.4)
for i in range(len(all_numeric_list)):
    ax = fig.add_subplot(5, 5, i+1)
    sns.distplot(train_df[train_df[all_numeric_list[i]].notnull()][all_numeric_list[i]], kde=False, ax=ax)
plt.show()
corr = train_df.corr()
fig = plt.figure(figsize=(12,12))
sns.heatmap(corr, vmax=.8, square=True)
# SalePriceと相関の高い変数トップ１０を確認
corr
cols_10 = corr.nlargest(11, 'SalePrice')['SalePrice'].index
corr_10 = np.corrcoef(train_df[cols_10].values.T) # 転置して変数を行毎にして計算
fig = plt.figure(figsize=(8,8))
sns.set(font_scale=1.25)
sns.heatmap(corr_10, cbar=True, annot=True, square=True,
            fmt='.2f', annot_kws={'size': 11},
            yticklabels=cols_10.values, xticklabels=cols_10.values)
plt.show()
# 可視化して分布状況確認
fig = plt.figure(figsize=(25,20))
plt.subplots_adjust(hspace=0.4,wspace=0.4)
for i in range(len(all_nominal_list)):
    ax = fig.add_subplot(6, 5, i+1)
    sns.boxplot(x=all_nominal_list[i],y='SalePrice', data=train_df, ax=ax)
plt.show()
fig = plt.figure(figsize=(20,15))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(len(cols_10)):
    ax = fig.add_subplot(3, 4, i+1)
    sns.scatterplot(x=cols_10[i], y='SalePrice', data=train_df, ax=ax)
plt.show()
train_df[(train_df.SalePrice < 200000) & (train_df.GrLivArea > 4000)][['GrLivArea', 'TotalBsmtSF', '1stFlrSF']]
# 外れ値除く
Gr_index = train_df[(train_df.SalePrice < 200000) & (train_df.GrLivArea > 4000)].index
train_df.drop(index=Gr_index, inplace=True)
train_Id.drop(index=Gr_index, inplace=True)
train_SalePrice.drop(index=Gr_index, inplace=True)
# 外れ値無くなったか確認
out_list = ['GrLivArea', 'TotalBsmtSF', '1stFlrSF']

fig = plt.figure(figsize=(15,6))
plt.subplots_adjust(wspace=0.6)
for i in range(len(out_list)):
    ax = fig.add_subplot(1, 3, i+1)
    sns.scatterplot(x=out_list[i], y='SalePrice', data=train_df, ax=ax)
plt.show()
# 欠損値確認用にデータ合体
train_num = len(train_df) # train_dfのデータ数保管
combined_df = pd.concat([train_df, test_df], ignore_index=True)
# 欠損値有り変数一覧
combined_df.isnull().sum().sort_values(ascending=False)[:40]
# PoolQC(プールのクオリティ) - プール無い物件がNAなので、文字列'None'で置き換え
combined_df.fillna({'PoolQC': 'None'}, inplace=True)

# MiscFeature(その他特徴) - 該当無しがNAなので、文字列'None'で置き換え
combined_df.fillna({'MiscFeature': 'None'}, inplace=True)

# Alley(住宅までの小道のタイプ) - 小道が無いのがNAなので、文字列'None'で置き換え
combined_df.fillna({'Alley': 'None'}, inplace=True)

# Fence(フェンスのクオリティ) - フェンスが無いのがNAなので、文字列'None'で置き換え
combined_df.fillna({'Fence': 'None'}, inplace=True)

# FireplaceQu(暖炉のコンディション) - 暖炉なしがNAなので、文字列'None'で置き換え
combined_df.fillna({'FireplaceQu': 'None'}, inplace=True)

# GarageFinish(ガレージの内装が終わっているか) - ガレージ無しがNAなので、文字列'None'で置き換え
combined_df.fillna({'GarageFinish': 'None'}, inplace=True)

# GarageCond(ガレージのコンディション) - ガレージ無しがNAなので、文字列'None'で置き換え
combined_df.fillna({'GarageCond': 'None'}, inplace=True)

# GarageQual(ガレージの品質) - ガレージ無しがNAなので、文字列'None'で置き換え
combined_df.fillna({'GarageQual': 'None'}, inplace=True)

# GarageType(ガレージの種類（地下にあるとか、家にくっついているとか）) - ガレージ無しがNAなので、文字列'None'で置き換え
combined_df.fillna({'GarageType': 'None'}, inplace=True)

# BsmtExposure(庭の見晴らし具合？) - 地下無しがNAなので、文字列'None'で置き換え
combined_df.fillna({'BsmtExposure': 'None'}, inplace=True)

# BsmtCond(地下室の状態) - 地下無しがNAなので、文字列'None'で置き換え
combined_df.fillna({'BsmtCond': 'None'}, inplace=True)

# BsmtQual(地下室の高さ) - 地下無しがNAなので、文字列'None'で置き換え
combined_df.fillna({'BsmtQual': 'None'}, inplace=True)

# BsmtFinType2(地下室の建築状況２) - 地下無しがNAなので、文字列'None'で置き換え
combined_df.fillna({'BsmtFinType2': 'None'}, inplace=True)

# BsmtFinType1(地下室の建築状況１) - 地下無しがNAなので、文字列'None'で置き換え
combined_df.fillna({'BsmtFinType1': 'None'}, inplace=True)
# LotFrontage(住宅から通りまでの距離)
#  - Neighborhood(物件エリアタイプ)毎に住宅までの距離は共通していると思われるので、
#    各NeighborhoodごとのLotFrontageの中央値を使って補完
combined_df['LotFrontage'] = combined_df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
# GarageYrBlt(ガレージの建築年) - ガレージ無ければ建築年が未記入になると考えられるので、0で置き換え
combined_df.fillna({'GarageYrBlt': 0}, inplace=True)

# GarageCars(ガレージに入る車の台数) - ガレージが無ければ台数も未記入になると考えられるので、0で置き換え
combined_df.fillna({'GarageCars': 0}, inplace=True)

# GarageArea(ガレージの面積) - ガレージが無ければ台数も未記入になると考えられるので、0で置き換え
combined_df.fillna({'GarageArea': 0}, inplace=True)
# MasVnrType(外壁レンガの種類) - レンガ使っていないのが欠損値と想定し、文字列'None'で置き換え
combined_df.fillna({'MasVnrType': 'None'}, inplace=True)

# MasVnrArea(レンガエリアの面積) - レンガ使っていないために欠損値と想定し、0で置き換え
combined_df.fillna({'MasVnrArea': 0}, inplace=True)
# 地下室が無ければ未記入になると想定し、0や'None'で置き換え

# BsmtFullBath(地下にある full bathroomの数)
combined_df.fillna({'BsmtFullBath': 0}, inplace=True)

# BsmtHalfBath(地下にある half bathroomの数)
combined_df.fillna({'BsmtHalfBath': 0}, inplace=True)

# TotalBsmtSF(地下室の総面積)
combined_df.fillna({'TotalBsmtSF': 0}, inplace=True)

# BsmtUnfSF(地下室で建築（カーペットの設置とか）がされていない面積)
combined_df.fillna({'BsmtUnfSF': 0}, inplace=True)

# BsmtFinSF2(地下室の工事済み面積2)
combined_df.fillna({'BsmtFinSF2': 0}, inplace=True)

# BsmtFinSF1(地下室の工事済み面積1)
combined_df.fillna({'BsmtFinSF1': 0}, inplace=True)
# Functional(住宅の機能性のランク)
combined_df['Functional'].value_counts()
# 殆どが'Typ'なので、'Typ'で埋める
combined_df.fillna({'Functional': 'Typ'}, inplace=True)
# Utilities(ガスとか電気とかの設備状況)

combined_df['Utilities'].value_counts()
# ほぼ全てが'AbbPub'なので、'AllPub'で埋める
# 実質的に定数と変わらないので、変数選択時に削除対象
combined_df.fillna({'Utilities': 'AllPub'}, inplace=True)
# Exterior1st(住宅の外壁の種類)
# Exterior2nd(住宅の外壁の種類（２種類使っている場合）)

print(combined_df['Exterior1st'].value_counts()[:5])
print('\n', combined_df['Exterior2nd'].value_counts()[:5])
# 外壁の種類と関連する変数が思い浮かばないので、最頻値の'VinylSd'で埋める
combined_df.fillna({'Exterior1st': 'VinylSd'}, inplace=True)
combined_df.fillna({'Exterior2nd': 'VinylSd'}, inplace=True)
# Electrical(電気システムの種類)

combined_df['Electrical'].value_counts()
# ほとんど'SBrkr'なので、'SBrkr'で埋める
combined_df.fillna({'Electrical': 'SBrkr'}, inplace=True)
# KitchenQual(台所のコンディション)

combined_df['KitchenQual'].value_counts()
# KitchenQual(台所のコンディション)と関連する変数が思い浮かばないので、最頻値で埋める
combined_df.fillna({'KitchenQual': 'TA'}, inplace=True)
# MSZoning(土地の種類)

# 分布を確認
combined_df['MSZoning'].value_counts()
# MSzoning(土地の種類)毎のLotFrontage(住宅までの距離)の中央値を算出
combined_df.groupby('MSZoning')['LotFrontage'].median()
# MSZoningが欠損のLotFrontageを確認
combined_df[combined_df['MSZoning'].isnull()]['LotFrontage']
# 1913,2214,2902 - 最頻値の'RL'で埋める
# 2248 - 'RH'で埋める

combined_df.loc[[1913, 2214, 2902], 'MSZoning'] = 'RL'
combined_df.loc[2248, 'MSZoning'] = 'RH'
combined_df['SaleType'].value_counts()
# 関連する変数が見つからないので、最頻値で埋める
combined_df.fillna({'SaleType': 'WD'}, inplace=True)
# 欠損値確認
combined_df.isnull().sum().sort_values(ascending=False)
# 分布確認
fig = plt.figure(figsize=(10, 4))
plt.subplots_adjust(wspace=0.4)

# ヒストグラム
ax = fig.add_subplot(1, 2, 1)
sns.distplot(train_SalePrice, fit=norm, ax=ax)

# QQプロット
ax2 = fig.add_subplot(1, 2, 2)
stats.probplot(train_SalePrice, plot=ax2)

plt.show()
# 対数変換
train_SalePrice = np.log1p(train_SalePrice)
# 変換後
fig = plt.figure(figsize=(10, 4))
plt.subplots_adjust(wspace=0.4)

# ヒストグラム
ax = fig.add_subplot(1, 2, 1)
sns.distplot(train_SalePrice, fit=norm, ax=ax)

# QQプロット
ax2 = fig.add_subplot(1, 2, 2)
stats.probplot(train_SalePrice, plot=ax2)

plt.show()
skewed_data = combined_df[int_list + numeric_list].apply(lambda x: skew(x)).sort_values(ascending=False)
skewed_data[:10]
skew_col = skewed_data[skewed_data > 10].index

# 可視化
fig = plt.figure(figsize=(10, 8))
for i in range(len(skew_col)):
    ax = fig.add_subplot(2, 3, i+1)
    try:
        sns.distplot(combined_df[skew_col[i]], fit=norm, ax=ax)
    except:
        # kde計算できない時は、kde=False
        sns.distplot(combined_df[skew_col[i]], fit=norm, kde=False, ax=ax)
plt.show()
# 対数変換
for i in range(len(skew_col)):
    combined_df[skew_col[i]] = np.log1p(combined_df[skew_col[i]])
# 可視化
# 可視化
fig = plt.figure(figsize=(10, 8))
for i in range(len(skew_col)):
    ax = fig.add_subplot(2, 3, i+1)
    try:
        sns.distplot(combined_df[skew_col[i]], fit=norm, ax=ax)
    except:
        # kde計算できない時は、kde=False
        sns.distplot(combined_df[skew_col[i]], fit=norm, kde=False, ax=ax)
plt.show()
var_df = combined_df[:train_num].copy()
# label encodin対象列取り出し
object_df = var_df[object_list]
# VarianceThresholdを使うため、一時的にobject型をlabel encoding
object_df = object_df.apply(LabelEncoder().fit_transform)
var_df[object_list] = object_df

# 正規化してスケーリング
SS = MinMaxScaler()
SS = SS.fit(var_df.values)
object_np = SS.transform(var_df.values)
var_df = pd.DataFrame(data=object_np, columns=var_df.columns)
# threshlod=0.01で定数に近いか判定
constant_filter = VarianceThreshold(threshold=0.01)
constant_filter.fit(var_df)

# 削除対象列取得
const_flag = constant_filter.get_support() # 定数に近い変数フラグ
const_cols = [x for x in var_df.columns if x not in var_df.columns[const_flag]]
print('削除対象列：', const_cols)
# 削除する前に、削除対象列の分布確認
fig = plt.figure(figsize=(15, 10))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

for i in range(len(const_cols)):
    ax = fig.add_subplot(2, 5, i+1)
    sns.distplot(var_df[const_cols[i]], kde=False, ax=ax)

plt.show()
# 実質定数の変数を削除

# 名義変数
all_nominal_list = list(set(all_nominal_list) - set(const_cols))
# 順序変数
all_ordinal_list = list(set(all_ordinal_list) - set(const_cols))
# 量的変数
all_numeric_list = list(set(all_numeric_list) - set(const_cols))
# CentralAir
CE_order = {'Y': 1, 'N': 0}
combined_df['CentralAir'] = combined_df['CentralAir'].map(CE_order).astype(int)

# BsmtExposure
BE_order = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
combined_df['BsmtExposure'] = combined_df['BsmtExposure'].map(BE_order).astype(int)

# ExterQual
EQ_order = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
combined_df['ExterQual'] = combined_df['ExterQual'].map(EQ_order).astype(int)

# GarageQual
GQ_order = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
combined_df['GarageQual'] = combined_df['GarageQual'].map(GQ_order).astype(int)

# LandSlope
LS_order = {'Sev': 0, 'Mod': 1, 'Gtl': 2}
combined_df['LandSlope'] = combined_df['LandSlope'].map(LS_order).astype(int)

# OverallCond - 不要

# HeatingQC
HQ_order = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
combined_df['HeatingQC'] = combined_df['HeatingQC'].map(HQ_order).astype(int)

# FireplaceQu
FQ_order = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
combined_df['FireplaceQu'] = combined_df['FireplaceQu'].map(FQ_order).astype(int)

# BsmtQual
BQ_order = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
combined_df['BsmtQual'] = combined_df['BsmtQual'].map(BQ_order).astype(int)

# KitchenQual
KQ_order = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
combined_df['KitchenQual'] = combined_df['KitchenQual'].map(KQ_order).astype(int)

# OverallQual - 不要

# GarageCond
GC_order = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
combined_df['GarageCond'] = combined_df['GarageCond'].map(GC_order).astype(int)

# GarageFinish
GF_order = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
combined_df['GarageFinish'] = combined_df['GarageFinish'].map(GF_order).astype(int)

# BsmtFinType2
BF_order = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
combined_df['BsmtFinType2'] = combined_df['BsmtFinType2'].map(BF_order).astype(int)

# BsmtFinType1
BF_order = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
combined_df['BsmtFinType1'] = combined_df['BsmtFinType1'].map(BF_order).astype(int)

# Functional
F_order = {'Sal':1, 'Sev':2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}
combined_df['Functional'] = combined_df['Functional'].map(F_order).astype(int)

# Electrical
E_order = {'Mix': 1, 'FuseP': 2, 'FuseF': 3, 'FuseA': 4, 'SBrkr': 5}
combined_df['Electrical'] = combined_df['Electrical'].map(E_order).astype(int)

# ExterCond
EC_order = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
combined_df['ExterCond'] = combined_df['ExterCond'].map(EC_order).astype(int)

# BsmtCond
BC_order = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
combined_df['BsmtCond'] = combined_df['BsmtCond'].map(BC_order).astype(int)

# Fence
Fe_order = {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
combined_df['Fence'] = combined_df['Fence'].map(Fe_order).astype(int)
# 対象変数
y_m_list = ['YearBuilt', 'YrSold', 'GarageYrBlt', 'YearRemodAdd', 'MoSold', 'SalePrice']
y_m_df = combined_df.loc[:train_num, y_m_list]

# 可視化
corr_y_m = y_m_df.corr()
corr_y_m
# 'YrSold'と'GarageYrBlt'を名義変数へ移動

all_nominal_list = all_nominal_list + ['YrSold', 'GarageYrBlt']

all_ordinal_list.remove('YrSold')
all_ordinal_list.remove('GarageYrBlt')
# 名義変数データ取り出し
BE_df = combined_df[all_nominal_list].copy()

# Binary Encoding
BE_df = BinaryEncoder(cols=all_nominal_list).fit_transform(BE_df)

# 変換前の変数削除
combined_df.drop(all_nominal_list, axis=1, inplace=True)

# Encoding後のデータ反映
combined_df[BE_df.columns] = BE_df

# 名義変数リストの更新
all_nominal_list = list(BE_df.columns)
# 対象変数のデータ取り出し
selected_df = combined_df[all_nominal_list + all_ordinal_list + all_numeric_list]

# データを訓練、テスト用に分割
train_df = selected_df.iloc[:train_num, :]
test_df = selected_df.iloc[train_num:, :]
selected_df.to_csv('/kaggle/working/selected_df_0616.csv', index=False)
train_SalePrice.to_csv('/kaggle/working/SalePrice_df_0616.csv', index=False)
# 評価関数作成
def rmse_r2(y, pred_y):
    rmse = np.sqrt(mean_squared_error(y, pred_y))
    r2 = r2_score(y, pred_y)
    print("test's rmse score: {:.3f}".format(rmse))
    print("test's r2 score: {:.3f}".format(r2))
# 評価用にデータ分割
train_x, test_x, train_y, test_y = train_test_split(train_df, train_SalePrice, test_size=0.25, random_state=2)
%%time

# pipeline設定
pipeline = Pipeline(
            [
             ('preprocessing', StandardScaler()),
             ('model', Ridge())
            ])

# パラメータ設定
# params = {'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
params = {'model__alpha': np.linspace(10, 150, 50)}

# KFold設定
r_kfold = RepeatedKFold(n_splits=8, n_repeats=3, random_state=42)

# grid search
grid = GridSearchCV(pipeline, params, cv=r_kfold, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid.fit(train_x, train_y)

# パラメータの保存
Ridge_param = grid.best_params_

print('Best score: {:.3f}'.format(np.abs(grid.best_score_)))
print('Best parameter: ', grid.best_params_)
pred_y = grid.predict(test_x)
rmse_r2(test_y, pred_y)
%%time

# pipeline設定
pipeline = Pipeline(
            [
             ('preprocessing', StandardScaler()),
             ('model', Lasso())
            ])

# パラメータ設定
# params = {'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
params = {'model__alpha': [0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01]}

# KFold設定
r_kfold = RepeatedKFold(n_splits=8, n_repeats=3, random_state=42)

# grid search
grid = GridSearchCV(pipeline, params, cv=r_kfold, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid.fit(train_x, train_y)

# パラメータの保存
Lasso_param = grid.best_params_

print('Best score: {:.3f}'.format(np.abs(grid.best_score_)))
print('Best parameter: ', grid.best_params_, '\n')
pred_y = grid.predict(test_x)
rmse_r2(test_y, pred_y)
%%time

# pipeline設定
pipeline = Pipeline(
            [
             ('preprocessing', StandardScaler()),
             ('model', ElasticNet())
            ])

# パラメータ設定
# params = {'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        #   'model__l1_ratio': np.arange(0.0, 1.0, 0.1)}
params = {'model__alpha': [0.005, 0.0075, 0.01, 0.025, 0.05],
          'model__l1_ratio': [0.015,0.0175, 0.2, 0.25, 0.275]}

# KFold設定
r_kfold = RepeatedKFold(n_splits=8, n_repeats=3, random_state=42)

# grid search
grid = GridSearchCV(pipeline, params, cv=r_kfold, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid.fit(train_x, train_y)

# パラメータの保存
Elastic_param = grid.best_params_

print('Best score: {:.3f}'.format(np.abs(grid.best_score_)))
print('Best parameter: ', grid.best_params_, '\n')
pred_y = grid.predict(test_x)
rmse_r2(test_y, pred_y)
%%time

# pipeline設定
pipeline = Pipeline(
            [
             ('preprocessing', StandardScaler()),
             ('model', SVR())
            ])

# パラメータ設定
# params = {'model__kernel': ['rbf'],
#           'model__C' : [0.001, 0.01, 1, 10, 100,1000],
#           'model__gamma':[0.001, 0.01, 1, 10, 100,1000]}
params = {'model__kernel': ['rbf'],
          'model__C' : [0.05, 0.75, 1, 1.25, 1.5, 1.75, 2],
          'model__gamma':[0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01]}

# KFold設定
r_kfold = RepeatedKFold(n_splits=8, n_repeats=3, random_state=42)

# grid search
grid = GridSearchCV(pipeline, params, cv=r_kfold, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid.fit(train_x, train_y)

# パラメータの保存
SVR_param = grid.best_params_

print('Best score: {:.3f}'.format(np.abs(grid.best_score_)))
print('Best parameter: ', grid.best_params_, '\n')
pred_y = grid.predict(test_x)
rmse_r2(test_y, pred_y)
%%time

# パラメータ設定
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)] # 10個
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(start=10, stop=200, num = 10)]
min_samples_split = [2,5,10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

params = {'n_estimators': n_estimators,
          'max_features': max_features,
          'max_depth': max_depth,
          'min_samples_split': min_samples_split,
          'min_samples_leaf': min_samples_leaf,
          'bootstrap': bootstrap}

# KFold設定
r_kfold = RepeatedKFold(n_splits=8, n_repeats=3, random_state=42)

# grid search
grid = RandomizedSearchCV(RandomForestRegressor(), params, cv=r_kfold, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid.fit(train_x, train_y)

# パラメータの保存
RF_param = grid.best_params_

print('Best score: {:.3f}'.format(np.abs(grid.best_score_)))
print('Best parameter: ', grid.best_params_, '\n')
pred_y = grid.predict(test_x)
rmse_r2(test_y, pred_y)
%%time

# パラメータ設定
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)] # 10個
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(start=10, stop=200, num = 10)]
min_samples_split = [2,5,10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

params = {
    "loss":["huber"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8,11],
    "max_features":["log2","sqrt", "auto"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[50, 100, 150, 200, 250, 300]
    }

# KFold設定
r_kfold = RepeatedKFold(n_splits=8, n_repeats=3, random_state=42)

# grid search
grid = RandomizedSearchCV(GradientBoostingRegressor(), params, cv=r_kfold, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid.fit(train_x, train_y)

# パラメータの保存
GB_param = grid.best_params_

print('Best score: {:.3f}'.format(np.abs(grid.best_score_)))
print('Best parameter: ', grid.best_params_, '\n')
pred_y = grid.predict(test_x)
rmse_r2(test_y, pred_y)
# lightgbm用データ準備
lgb_train = lgb.Dataset(train_x, train_y)
lgb_test = lgb.Dataset(test_x, test_y)
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }
    # 訓練
    gbm = lgb_optuna.train(params,
                           lgb_train,
                           valid_sets=(lgb_train, lgb_test),
                           num_boost_round=10000,
                           early_stopping_rounds=100,
                           verbose_eval=50)
    # 予測とスコア
    predicted = gbm.predict(test_x)
    rmse = np.sqrt(mean_squared_error(test_y, predicted))
        
    return rmse
# トレーニング
study_gbm = optuna.create_study()
study_gbm.optimize(objective, timeout=1800)
print('best score: {:.3f}'.format(study_gbm.best_value))
print('parameters: ',study_gbm.best_params)
def objective(trial):
    # パラメータ設定
    n_estimators = trial.suggest_int('n_estimators', 0, 10000)
    max_depth = trial.suggest_int('max_depth', 1, 20)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 20)
    learning_rate = trial.suggest_discrete_uniform('learning_rate', 0.001, 0.01, 0.001)
    scale_pos_weight = trial.suggest_int('scale_pos_weight', 1, 100)
    subsample = trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1)
    colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1)

    xgb_ob = xgb.XGBRegressor(
        random_state=42,
        n_estimators = n_estimators,
        max_depth = max_depth,
        min_child_weight = min_child_weight,
        learning_rate = learning_rate,
        scale_pos_weight = scale_pos_weight,
        subsample = subsample,
        colsample_bytree = colsample_bytree
    )

    # 訓練・スコアリング
    xgb_ob.fit(train_x, train_y)
    predicted = xgb_ob.predict(test_x)
    rmse = np.sqrt(mean_squared_error(test_y, predicted))
        
    return rmse
# 訓練
study_xgb = optuna.create_study()
study_xgb.optimize(objective, timeout=1800)

# 結果
print('best score: {:.3f}'.format(study_xgb.best_value))
print('parameters: ', study_xgb.best_params)
# 各モデルのパラメータ
ridge_param = {'alpha': 50.0}
lasso_param = {'alpha': 0.0025}
elas_param = {'alpha': 0.0075, 'l1_ratio': 0.275}
svr_param = {'C': 2, 'gamma': 0.00075, 'kernel': 'rbf'}

# モデルオブジェクトのリスト
model_list = [Ridge(**ridge_param), Lasso(**lasso_param), ElasticNet(**elas_param), SVR(**svr_param)]

model_name = ['Ridge', 'Lasso', 'ElasticNet', 'SVR']
for i in range(len(model_list)):
    
    # pipeline設定
    pipeline = Pipeline(
                [
                 ('preprocessing', StandardScaler()),
                 ('model', model_list[i])
                ])

    scores = cross_val_score(pipeline, train_df, train_SalePrice, cv=8, scoring='neg_root_mean_squared_error', n_jobs=-1)
    scores = np.abs(scores)
    print('-- ',model_name[i],' --')
#     print('scores: ',scores)
    print('rmse_mean: {:.4f} rmse_std: {:.4f}'.format(np.mean(scores), np.std(scores)))
# 各モデルのパラメータ
RF_param = {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 
            'max_features': 'sqrt', 'max_depth': 157, 'bootstrap': False}
GB_param = {'subsample': 0.95, 'n_estimators': 250, 'min_samples_split': 0.1, 
            'min_samples_leaf': 0.13636363636363638, 'max_features': 'log2', 
            'max_depth': 8, 'loss': 'huber', 'learning_rate': 0.15}


# モデルオブジェクトのリスト
model_list = [RandomForestRegressor(**RF_param), GradientBoostingRegressor(**GB_param)]

model_name = ['Random Forest', 'Gradient Boosting']
for i in range(len(model_list)):

    scores = cross_val_score(model_list[i], train_df, train_SalePrice, cv=8, scoring='neg_root_mean_squared_error', n_jobs=-1)
    scores = np.abs(scores)
    print('-- ',model_name[i],' --')
#     print('scores: ',scores)
    print('rmse_mean: {:.4f} rmse_std: {:.4f}'.format(np.mean(scores), np.std(scores)))
# lightgbm用データ
lgb_data = lgb.Dataset(train_df.values, train_SalePrice.values)

# パラメータ
lgb_param = {'lambda_l1': 2.5456694926318675e-08, 'lambda_l2': 1.133230160071623e-08, 
             'num_leaves': 4, 'feature_fraction': 0.5607153808743172, 
             'bagging_fraction': 0.7817404071420171, 'bagging_freq': 1,
             'min_child_samples': 5, 'objective': 'regression', 'metric': 'rmse'}


cv_results = lgb.cv(lgb_param, 
                    lgb_data,
                    num_boost_round=10000,
                    nfold=8,
                    verbose_eval=20,
                    early_stopping_rounds=100,
                    stratified=False) # regressionの時は、stratified=False
# ブートラウンド毎のRMSE値取り出し
cv_rmse = cv_results['rmse-mean']
# ブートランド数array取り出し
round_num = np.arange(len(cv_rmse))

# 最良RMSEスコアと、その時のブートラウンド数
print('Best num_boost_round: ',len(cv_rmse))
print("Best CV's RMSE score: {:.3f}".format(cv_rmse[-1]))
print("std: {:.4f}".format(cv_results['rmse-stdv'][-1]))

# ブートラウンド毎のRMSEスコアの可視化
# fig = plt.figure()
# plt.xlabel('round')
# plt.ylabel('RMSE-mean')
# plt.plot(round_num, cv_rmse)
# plt.show()
# パラメータ
xgb_param = {'n_estimators': 998,'max_depth': 16, 'min_child_weight': 4,
             'learning_rate': 0.01, 'scale_pos_weight': 14, 'subsample': 0.5,
             'colsample_bytree': 0.9}

# xgbオブジェクト用意
xgb_ob = xgb.XGBRegressor(**xgb_param)

# 交差検証
kfold = KFold(n_splits=8, random_state=1)
scores = cross_val_score(xgb_ob, train_df, train_SalePrice, cv=kfold,
                         scoring='neg_root_mean_squared_error', n_jobs=-1)

# 結果
scores = np.abs(scores)
print('rmse_mean: {:.4f} / rmse_std: {:.4f}'.format(np.mean(scores), np.std(scores)))
# パラメータ
EN_param = {'alpha': 0.0075, 'l1_ratio': 0.275}

# 正規化
scaler = StandardScaler()
scaler.fit(train_df)
SS_train = scaler.transform(train_df)
SS_test = scaler.transform(test_df)

# モデリング
EN_ob = ElasticNet(**EN_param)
EN_ob.fit(SS_train, train_SalePrice)

# 予測
EN_predicted = EN_ob.predict(SS_test)
EN_ex_predicted = np.expm1(EN_predicted)
predicted_df = pd.DataFrame({'Id': test_Id.values, 'SalePrice': EN_ex_predicted})
predicted_df.to_csv('/kaggle/working/Ver6_ElasticNet_0619.csv', index=False)
# パラメータ
RF_param = {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 
            'max_features': 'sqrt', 'max_depth': 157, 'bootstrap': False}

# モデリング
RF_ob = RandomForestRegressor(**RF_param)
RF_ob.fit(train_df, train_SalePrice)

# 予測
RF_predicted = RF_ob.predict(test_df)
RF_ex_predicted = np.expm1(RF_predicted)
predicted_df = pd.DataFrame({'Id': test_Id.values, 'SalePrice': RF_ex_predicted})
predicted_df.to_csv('/kaggle/working/Ver6_RandomForest_0619.csv', index=False)
# パラメータ
xgb_param = {'n_estimators': 998,'max_depth': 16, 'min_child_weight': 4,
             'learning_rate': 0.01, 'scale_pos_weight': 14, 'subsample': 0.5,
             'colsample_bytree': 0.9}

# モデリング
xgb_ob = xgb.XGBRegressor(**xgb_param)
xgb_ob.fit(train_df, train_SalePrice)

# 予測
xgb_predicted = xgb_ob.predict(test_df)
xgb_ex_predicted = np.expm1(xgb_predicted)
predicted_df = pd.DataFrame({'Id': test_Id.values, 'SalePrice': xgb_ex_predicted})
predicted_df.to_csv('/kaggle/working/Ver6_xgboost_0619.csv', index=False)
# ElasticNetとXGBoostの予測結果の平均を提出
stacked_result = (EN_predicted + xgb_predicted) / 2
ex_stacked_reslut = np.expm1(stacked_result)
predicted_df = pd.DataFrame({'Id': test_Id.values, 'SalePrice': ex_stacked_reslut})
predicted_df.to_csv('/kaggle/working/Ver6_ElasticNet&XGBoost_0619.csv', index=False)