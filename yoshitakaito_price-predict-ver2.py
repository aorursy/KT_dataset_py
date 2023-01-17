import numpy as np

import pandas as pd

from scipy import stats

import collections

import itertools

import math

from matplotlib import pyplot as plt

import seaborn as sns

sns.set()

sns.set_style('whitegrid')



from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import LabelEncoder

from category_encoders import BinaryEncoder

from sklearn.preprocessing import OrdinalEncoder



# for imputation to missing values

# KNN

from sklearn.impute import KNNImputer

# MICE

# https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html

# from sklearn.experimental import enable_iterative_imputer

# from sklearn.impute import IterativeImputer



from sklearn.preprocessing import MinMaxScaler



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import cross_validate

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

# import lightgbm as lgb

import optuna.integration.lightgbm as lgb_optuna

import optuna

import lightgbm as lgb
# トレインデータ取り出し

train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

# テストデータ取り出し

test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



# データ合体

combined_df = pd.concat([train_df, test_df], ignore_index=True)

train_y = train_df['SalePrice'] # target value

combined_df.drop('SalePrice',axis=1, inplace=True)
train_df.info()
# 変数削除処理用のdf用意

del_col_train = train_df.copy()

# del_col_test = test_df.copy()
train_df.tail()
null_col_train = ''

null_col_test = ''

null_train_list = [] # train_dfの欠損値有り変数のリスト

null_test_list = [] # test_dfの欠損値有り変数のリスト



# train_dfの欠損値有り変数取り出し

for col in train_df.columns:

    check = train_df[col].isnull().value_counts()

    if len(check) != 1:

        # 欠損値有り変数

        null_col_train = null_col_train + col + ', '

        null_train_list.append(col)



# test_dfの欠損値有り変数取り出し

for col in test_df.columns:

    check = test_df[col].isnull().value_counts()

    if len(check) != 1:

        # 欠損値有り変数

        null_col_test = null_col_test + col + ', '

        null_test_list.append(col)



print('訓練データの欠損値有り変数の数： ',len(null_train_list))

print('変数名：', null_col_train, '\n')

print('テストデータの欠損値有り変数の数： ',len(null_test_list))

print('変数名：', null_col_test)
# 欠損値数のボーダー（訓練データ・テストデータ合計の８割以上ある変数のみ利用）

null_border = 2919 * 0.2



# trainデータで、欠損値が8割以上の変数抽出

null_count = combined_df.isnull().sum()

null_col_list = null_count[null_count >= null_border]

null_col_list = null_col_list.index.to_list()

print('削除対象変数：', null_col_list)



# 欠損値が8割以上の変数削除

del_col_train = del_col_train.drop(null_col_list, axis=1)

combined_df = combined_df.drop(null_col_list, axis=1)
# 残りの変数の欠損値状況

combined_df.isnull().sum().sort_values(ascending=False)[:10]


# 各変数の欠損値を最頻値で埋める

# mode()はdfで返すので、先頭データを取り出す。

del_col_train.fillna(del_col_train.mode().iloc[0], inplace=True)

# del_col_test.fillna(del_col_test.mode().iloc[0], inplace=True)



# Label Encoding対象列取り出し

object_list = del_col_train.select_dtypes(include='object').columns.to_list()

object_df = del_col_train[object_list]



# Label Encoding

object_df = object_df.apply(LabelEncoder().fit_transform)

del_col_train[object_df.columns] = object_df
# 分散0.1以下の変数削除

constant_filter = VarianceThreshold(threshold=0.1)

constant_filter.fit(del_col_train)

constant_list = [not temp for temp in constant_filter.get_support()]

# constant_listの中身は[False,True,False...]と各変数が対象かどうかを示すリスト

x_train_filter = constant_filter.transform(del_col_train)

x_train_filter.shape, del_col_train.shape
# 削除対象変数抽出

del_list = del_col_train.columns[constant_list]

print('削除対象変数：',del_list)



# 変数削除

del_col_train = del_col_train.drop(del_list, axis=1) # 同じ変数削除処理用

combined_df = combined_df.drop(del_list, axis=1)
# データを転置してduplicatedメソッドで重複行削除する

del_col_train_T = del_col_train.T

del_col_train_T = pd.DataFrame(del_col_train_T)

del_col_train_T.head()
del_col_train_T.shape
del_col_train_T.duplicated().sum()
# 不要な列('ID')の削除

combined_df.drop('Id', axis=1, inplace=True)
combined_df.shape
# object型変数(Categorical)の抽出

object_list = combined_df.select_dtypes(include='object').columns.to_list()



# int型変数(Categorical変数とNumerical変数が混ざっている)の抽出

int_list = combined_df.select_dtypes(include='int').columns.to_list()



# float型変数(Numerical変数)の抽出

numeric_list = combined_df.select_dtypes(include='float').columns.to_list()
object_list
# ojbect_listを名義変数リストと順序変数リストに分ける



# 名義変数

nominal_list = ['MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood',

                'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',

                'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',

                'GarageType', 'PavedDrive', 'SaleType', 'SaleCondition']

# 順序変数

ordinal_list = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',

                'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical',

                'KitchenQual', 'Functional', 'GarageFinish','GarageQual', 'GarageCond']
# Categorical-name変数

to_nominal_list = ['MSSubClass']



# Categorical-ordinal変数

to_ordinal_list = ['OverallQual', 'OverallCond']



# numerical変数

to_num_list = ['LotArea', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF',

               'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr',

               'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF',

               'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

               'MiscVal', 'MoSold', 'YrSold']
len(to_nominal_list+to_ordinal_list+to_num_list), len(int_list)

# 'Id列を除いているので、数がずれる'
# 名義変数

all_nominal_list = nominal_list + to_nominal_list

# 順序変数

all_ordinal_list = ordinal_list + to_ordinal_list

# 量的変数

all_numeric_list = numeric_list + to_num_list



# 上記リストには、Id変数は除いている
numeric_list
# 欠損値の確認

for col, num in combined_df.isnull().sum().sort_values(ascending=False).iteritems():

    if num != 0:

        print(col,' : ',num)
# - KNN imputation用に、categorical変数(object_list)を数値化する



KNN_combined_data = combined_df.copy()



# Categoricalデータ(文字列)取り出し

Cat_data = KNN_combined_data[object_list]



# Ordinal Encodersインスタンス用の辞書作成

ordinal_enc_dict = {}



# 対象変数をエンコーディング

for col_name in Cat_data:

    # 変数毎のordinal encoderインスタンス作成

    ordinal_enc_dict[col_name] = OrdinalEncoder()

    # 欠損値以外のデータ取り出し

    col = Cat_data[col_name]

    col_not_null = col[col.notnull()]

    reshaped_vals = col_not_null.values.reshape(-1, 1)

    # 欠損値以外のデータをエンコーディング

    encoded_vals = ordinal_enc_dict[col_name].fit_transform(reshaped_vals)

    # カテゴリ値をエンコーディング結果と入れ替え

    Cat_data.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)

    

# dfに反映

KNN_combined_data[object_list] = Cat_data
# # MICEで欠損値補完

# MICE_imputer = IterativeImputer()

# MICE_imputed_data = MICE_imputer.fit_transform(MICE_combined_data)



# # 補完済みデータを反映

# MICE_combined_data.iloc[:,:] = MICE_imputed_data



# MICE_combined_data.head()
# カテゴリカル変数データ取り出し

Category_df = KNN_combined_data[all_nominal_list + all_ordinal_list].copy()

# 数値型変数データ取り出し

Numeric_df = KNN_combined_data[all_numeric_list].copy()



# KNNで欠損値補完

KNN_imputer_cat = KNNImputer() # カテゴリカル変数用

KNN_imputer_num = KNNImputer() # 数値型変数用



# カテゴリカル変数の欠損値補完（補完値を丸めて整数にし、カテゴリの要素化）

Category_df = pd.DataFrame(np.round(KNN_imputer_cat.fit_transform(Category_df)), columns=Category_df.columns)

# 数値型変数の欠損値

Numeric_df = pd.DataFrame(KNN_imputer_num.fit_transform(Numeric_df), columns=Numeric_df.columns)



# 補完済みデータを反映

KNN_combined_data.loc[:, Category_df.columns] = Category_df

KNN_combined_data.loc[:, Numeric_df.columns] = Numeric_df



KNN_combined_data.info()
len(KNN_combined_data.columns)
# 最頻値、及び中央値で穴埋め



MeMo_combined_data = combined_df.copy()



# Categorical変数は、最頻値で埋める

Cat_mis_df = MeMo_combined_data[all_nominal_list + all_ordinal_list].copy()

Cat_mis_df.fillna(Cat_mis_df.mode().iloc[0], inplace=True)



# Continuous変数は、中央値で欠損値を埋める

Num_mis_df = MeMo_combined_data[all_numeric_list]

Num_mis_df.fillna(Num_mis_df.median(), inplace=True)



# dfに反映

MeMo_combined_data[all_nominal_list + all_ordinal_list] = Cat_mis_df

MeMo_combined_data[all_numeric_list] = Num_mis_df
# MICEで補完

fig = plt.figure(figsize=(15,6))

plt.subplots_adjust(wspace=0.4)

ax1 = fig.add_subplot(1,2,1)

g = sns.kdeplot(combined_df['LotFrontage'], color='Red', shade=True)

g = sns.kdeplot(KNN_combined_data['LotFrontage'], color='Blue', shade=True)

g.set_xlabel('LotFrontage')

g.set_ylabel('Frequency')

g.set_title('imputation by KNN')

g = g.legend(['before imputation', 'after imputation'])



# 中央値・最頻値で補完

ax2 = fig.add_subplot(1,2,2)

g2 = sns.kdeplot(combined_df['LotFrontage'], color='Red', shade=True)

g2 = sns.kdeplot(MeMo_combined_data['LotFrontage'], color='Blue', shade=True)

g2.set_xlabel('LotFrontage')

g2.set_ylabel('Frequency')

g2.set_title('imputation by Median')

g2 = g2.legend(['before imputation', 'after imputation'])



plt.show()
# KNNで補完

fig = plt.figure(figsize=(15,6))

plt.subplots_adjust(wspace=0.4)

ax1 = fig.add_subplot(1,2,1)

g = sns.kdeplot(combined_df['GarageYrBlt'], color='Red', shade=True)

g = sns.kdeplot(KNN_combined_data['GarageYrBlt'], color='Blue', shade=True)

g.set_xlabel('GarageYrBlt')

g.set_ylabel('Frequency')

g.set_title('imputation by KNN')

g = g.legend(['before imputation', 'after imputation'])



# 最頻値で補完

ax2 = fig.add_subplot(1,2,2)

g2 = sns.kdeplot(combined_df['GarageYrBlt'], color='Red', shade=True)

g2 = sns.kdeplot(MeMo_combined_data['GarageYrBlt'], color='Blue', shade=True)

g2.set_xlabel('GarageYrBlt')

g2.set_ylabel('Frequency')

g2.set_title('imputation by Median')

g2 = g2.legend(['before imputation', 'after imputation'])



plt.show()
# train_dfのインデックス範囲確認

combined_df.iloc[:1460,:]
# 訓練データのContinuous変数取り出し

KNN_cont_df = KNN_combined_data[all_numeric_list].iloc[:1460,:].copy()



# boxplot

fig = plt.figure(figsize=(15,30))

plt.subplots_adjust(hspace=0.2, wspace=0.8)

for i in range(len(all_numeric_list)):

    ax = fig.add_subplot(7, 5, i+1)

    sns.boxplot(y=KNN_cont_df[all_numeric_list[i]], data= KNN_cont_df, ax=ax)

plt.show()
# ヒストグラムで分布確認

binary_list = ['BsmtFinSF2', 'LowQualFinSF', 'EnclosedPorch',

               '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']



fig = plt.figure(figsize=(14,10))

plt.subplots_adjust(hspace=0.4,wspace=0.4)

bins = round(1 + math.log2(len(KNN_cont_df))) # スタージェスの公式によるビン数の決定

for i in range(len(binary_list)):

    ax = fig.add_subplot(2, 4, i+1)

    sns.distplot(KNN_cont_df[binary_list[i]], bins=bins, kde=False, ax=ax)

plt.show()
KNN_cont_df_y = pd.concat([KNN_cont_df, train_y], axis=1)

                           

fig = plt.figure(figsize=(14,10))

plt.subplots_adjust(hspace=0.4,wspace=0.6)

for i in range(len(binary_list)):

    ax = fig.add_subplot(2, 4, i+1)

    graph = sns.scatterplot(x=binary_list[i], y='SalePrice', data=KNN_cont_df_y, ax=ax)

    graph.axhline(163000, color='red') # SalesPriceの中央値を赤線表示

plt.show()
print('0以外のデータ数')

for i in range(len(binary_list)):

    x = KNN_cont_df[KNN_cont_df[binary_list[i]] > 0][binary_list[i]]

    print(binary_list[i], ' : ', len(x))
# ７変数削除

KNN_cont_df.drop(columns=binary_list, inplace=True)

KNN_combined_data.drop(columns=binary_list, inplace=True)

MeMo_combined_data.drop(columns=binary_list, inplace=True)



# 数値型変数名リストの更新

all_numeric_list = list(set(all_numeric_list) ^ set(binary_list))

# https://note.nkmk.me/python-list-common/
# 外れ値対象列

out_list = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 

            'BsmtFullBath', 'GarageCars', 'GarageArea', 'LotArea', 'YearBuilt', '1stFlrSF', 

            '2ndFlrSF', 'GrLivArea', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces',

            'WoodDeckSF', 'OpenPorchSF']
# 対数変換



KNN_log_df = KNN_cont_df[out_list].copy()

KNN_log_combined_data = KNN_combined_data.copy()

MeMo_log_combined_data = MeMo_combined_data.copy()



# 列ごとに対数変換

for col in out_list:

    # 訓練データ

    KNN_log_df[col] = KNN_log_df[col].apply(lambda x :np.log10(x + 1))

    # 合体データ

    KNN_log_combined_data[col] = KNN_log_combined_data[col].apply(lambda x :np.log10(x + 1))

    MeMo_log_combined_data[col] = MeMo_log_combined_data[col].apply(lambda x :np.log10(x + 1))
# boxplot

fig = plt.figure(figsize=(15,20))

plt.subplots_adjust(hspace=0.2, wspace=0.8)

for i in range(len(out_list)):

    ax = fig.add_subplot(4, 5, i+1)

    sns.boxplot(y=KNN_log_df[out_list[i]], data= KNN_log_df, ax=ax)

plt.show()
# %tile値で置き換え - 置き換えるのは訓練データのみ



KNN_cup_df = KNN_cont_df[out_list].copy()
# 5%tile, 95%tileで置き換え

for col in out_list:

    # 置き換え値

    upper_lim = KNN_cup_df[col].quantile(.95)

    lower_lim = KNN_cup_df[col].quantile(.05)

    

    # IQR

    Q1 = KNN_cup_df[col].quantile(.25)

    Q3 = KNN_cup_df[col].quantile(.75)

    IQR = Q3 - Q1

    outlier_step = 1.5 * IQR

    

    # 1.5IQR超える数値は95%tile値で埋める、下回る数値は5%tile値で埋める

    KNN_cup_df.loc[(KNN_cup_df[col] > (Q3 + outlier_step)), col] = upper_lim

    KNN_cup_df.loc[(KNN_cup_df[col] < (Q1 - outlier_step)), col] = lower_lim
# boxplot

fig = plt.figure(figsize=(15,20))

plt.subplots_adjust(hspace=0.2, wspace=0.8)

for i in range(len(out_list)):

    ax = fig.add_subplot(4, 5, i+1)

    sns.boxplot(y=KNN_cup_df[out_list[i]], data= KNN_cup_df, ax=ax)

plt.show()
# 合体データに反映



KNN_cup_combined_data = KNN_combined_data.copy()

MeMo_cup_combined_data = MeMo_combined_data.copy()



KNN_cup_combined_data.loc[:1459,KNN_cup_df.columns] = KNN_cup_df

MeMo_cup_combined_data.loc[:1459,KNN_cup_df.columns] = KNN_cup_df
MEMO_BE_df = MeMo_combined_data[all_nominal_list].copy()

KNN_BE_df = KNN_combined_data[all_nominal_list].copy()



# Binary Encoding

MEMO_BE_df = BinaryEncoder(cols=all_nominal_list).fit_transform(MEMO_BE_df)

KNN_BE_df = BinaryEncoder(cols=all_nominal_list).fit_transform(KNN_BE_df)



# 変換前変数削除

MeMo_combined_data.drop(all_nominal_list, axis=1, inplace=True)

MeMo_log_combined_data.drop(all_nominal_list, axis=1, inplace=True)

MeMo_cup_combined_data.drop(all_nominal_list, axis=1, inplace=True)



KNN_combined_data.drop(all_nominal_list, axis=1, inplace=True)

KNN_log_combined_data.drop(all_nominal_list, axis=1, inplace=True)

KNN_cup_combined_data.drop(all_nominal_list, axis=1, inplace=True)





# 変換後データの反映

MeMo_combined_data[MEMO_BE_df.columns] = MEMO_BE_df

MeMo_log_combined_data[MEMO_BE_df.columns] = MEMO_BE_df

MeMo_cup_combined_data[MEMO_BE_df.columns] = MEMO_BE_df



KNN_combined_data[KNN_BE_df.columns] = KNN_BE_df

KNN_log_combined_data[KNN_BE_df.columns] = KNN_BE_df

KNN_cup_combined_data[KNN_BE_df.columns] = KNN_BE_df
all_ordinal_list
# Ordinal Encoding用辞書

# ※ knn imputationするときに、ラベルエンコーディングしたまま元に戻すの忘れていた！！！

#   なので、knn用にも辞書作成した。



# ExterQual

knn_ExterQual = {3.0: 3, 2.0: 4, 0.0: 5, 1.0: 2}

memo_ExterQual = {'TA': 3, 'Gd': 4, 'Ex': 5, 'Fa': 2}



KNN_combined_data['ExterQual'] = KNN_combined_data['ExterQual'].map(knn_ExterQual)

KNN_log_combined_data['ExterQual'] = KNN_log_combined_data['ExterQual'].map(knn_ExterQual)

KNN_cup_combined_data['ExterQual'] = KNN_cup_combined_data['ExterQual'].map(knn_ExterQual)



MeMo_combined_data['ExterQual'] = MeMo_combined_data['ExterQual'].map(memo_ExterQual)

MeMo_log_combined_data['ExterQual'] = MeMo_log_combined_data['ExterQual'].map(memo_ExterQual)

MeMo_cup_combined_data['ExterQual'] = MeMo_cup_combined_data['ExterQual'].map(memo_ExterQual)





# ExterCond

knn_ExterCond = {4.0: 3, 2.0: 4, 0.0: 5, 1.0: 2, 3.0: 1}

memo_ExterCond = {'TA': 3, 'Gd': 4, 'Ex': 5, 'Fa': 2, 'Po': 1}



KNN_combined_data['ExterCond'] = KNN_combined_data['ExterCond'].map(knn_ExterCond)

KNN_log_combined_data['ExterCond'] = KNN_log_combined_data['ExterCond'].map(knn_ExterCond)

KNN_cup_combined_data['ExterCond'] = KNN_cup_combined_data['ExterCond'].map(knn_ExterCond)



MeMo_combined_data['ExterCond'] = MeMo_combined_data['ExterCond'].map(memo_ExterCond)

MeMo_log_combined_data['ExterCond'] = MeMo_log_combined_data['ExterCond'].map(memo_ExterCond)

MeMo_cup_combined_data['ExterCond'] = MeMo_cup_combined_data['ExterCond'].map(memo_ExterCond)





# BsmtQual

knn_BsmtQual = {3.0: 3, 2.0: 4, 0.0: 5, 1.0: 2}

memo_BsmtQual = {'TA': 3, 'Gd': 4, 'Ex': 5, 'Fa': 2}



KNN_combined_data['BsmtQual'] = KNN_combined_data['BsmtQual'].map(knn_BsmtQual)

KNN_log_combined_data['BsmtQual'] = KNN_log_combined_data['BsmtQual'].map(knn_BsmtQual)

KNN_cup_combined_data['BsmtQual'] = KNN_cup_combined_data['BsmtQual'].map(knn_BsmtQual)



MeMo_combined_data['BsmtQual'] = MeMo_combined_data['BsmtQual'].map(memo_BsmtQual)

MeMo_log_combined_data['BsmtQual'] = MeMo_log_combined_data['BsmtQual'].map(memo_BsmtQual)

MeMo_cup_combined_data['BsmtQual'] = MeMo_cup_combined_data['BsmtQual'].map(memo_BsmtQual)





# BsmtCond

knn_BsmtCond = {3.0: 3, 1.0: 4, 0.0: 2, 2.0: 1}

memo_BsmtCond = {'TA': 3, 'Gd': 4, 'Fa': 2, 'Po': 1}



KNN_combined_data['BsmtCond'] = KNN_combined_data['BsmtCond'].map(knn_BsmtCond)

KNN_log_combined_data['BsmtCond'] = KNN_log_combined_data['BsmtCond'].map(knn_BsmtCond)

KNN_cup_combined_data['BsmtCond'] = KNN_cup_combined_data['BsmtCond'].map(knn_BsmtCond)



MeMo_combined_data['BsmtCond'] = MeMo_combined_data['BsmtCond'].map(memo_BsmtCond)

MeMo_log_combined_data['BsmtCond'] = MeMo_log_combined_data['BsmtCond'].map(memo_BsmtCond)

MeMo_cup_combined_data['BsmtCond'] = MeMo_cup_combined_data['BsmtCond'].map(memo_BsmtCond)





# BsmtExposure

knn_BsmtExposure = {3.0: 2, 0.0: 4, 1.0: 5, 2.0: 3}

memo_BsmtExposure = {'No': 2, 'Av': 4, 'Gd': 5, 'Mn': 3}



KNN_combined_data['BsmtExposure'] = KNN_combined_data['BsmtExposure'].map(knn_BsmtExposure)

KNN_log_combined_data['BsmtExposure'] = KNN_log_combined_data['BsmtExposure'].map(knn_BsmtExposure)

KNN_cup_combined_data['BsmtExposure'] = KNN_cup_combined_data['BsmtExposure'].map(knn_BsmtExposure)



MeMo_combined_data['BsmtExposure'] = MeMo_combined_data['BsmtExposure'].map(memo_BsmtExposure)

MeMo_log_combined_data['BsmtExposure'] = MeMo_log_combined_data['BsmtExposure'].map(memo_BsmtExposure)

MeMo_cup_combined_data['BsmtExposure'] = MeMo_cup_combined_data['BsmtExposure'].map(memo_BsmtExposure)





# BsmtFinType1

knn_BsmtFinType1 = {2.0: 2, 5.0: 7, 0.0: 6, 4.0: 4, 1.0: 5, 3.0: 3}

memo_BsmtFinType1 = {'Unf': 2, 'GLQ': 7, 'ALQ': 6, 'Rec': 4, 'BLQ': 5, 'LwQ': 3}



KNN_combined_data['BsmtFinType1'] = KNN_combined_data['BsmtFinType1'].map(knn_BsmtFinType1)

KNN_log_combined_data['BsmtFinType1'] = KNN_log_combined_data['BsmtFinType1'].map(knn_BsmtFinType1)

KNN_cup_combined_data['BsmtFinType1'] = KNN_cup_combined_data['BsmtFinType1'].map(knn_BsmtFinType1)



MeMo_combined_data['BsmtFinType1'] = MeMo_combined_data['BsmtFinType1'].map(memo_BsmtFinType1)

MeMo_log_combined_data['BsmtFinType1'] = MeMo_log_combined_data['BsmtFinType1'].map(memo_BsmtFinType1)

MeMo_cup_combined_data['BsmtFinType1'] = MeMo_cup_combined_data['BsmtFinType1'].map(memo_BsmtFinType1)





# BsmtFinType2

knn_BsmtFinType2 = {5.0: 2, 4.0: 4, 3.0: 3, 1.0: 5, 0.0: 6, 2.0: 7}

memo_BsmtFinType2 = {'Unf': 2, 'Rec': 4, 'LwQ': 3, 'BLQ': 5, 'ALQ': 6, 'GLQ': 7}



KNN_combined_data['BsmtFinType2'] = KNN_combined_data['BsmtFinType2'].map(knn_BsmtFinType2)

KNN_log_combined_data['BsmtFinType2'] = KNN_log_combined_data['BsmtFinType2'].map(knn_BsmtFinType2)

KNN_cup_combined_data['BsmtFinType2'] = KNN_cup_combined_data['BsmtFinType2'].map(knn_BsmtFinType2)



MeMo_combined_data['BsmtFinType2'] = MeMo_combined_data['BsmtFinType2'].map(memo_BsmtFinType2)

MeMo_log_combined_data['BsmtFinType2'] = MeMo_log_combined_data['BsmtFinType2'].map(memo_BsmtFinType2)

MeMo_cup_combined_data['BsmtFinType2'] = MeMo_cup_combined_data['BsmtFinType2'].map(memo_BsmtFinType2)





# HeatingQC

knn_HeatingQC = {0.0: 5, 4.0: 3, 2.0: 4, 1.0: 2, 3.0: 1}

memo_HeatingQC = {'Ex': 5, 'TA': 3, 'Gd': 4, 'Fa': 2, 'Po': 1}



KNN_combined_data['HeatingQC'] = KNN_combined_data['HeatingQC'].map(knn_HeatingQC)

KNN_log_combined_data['HeatingQC'] = KNN_log_combined_data['HeatingQC'].map(knn_HeatingQC)

KNN_cup_combined_data['HeatingQC'] = KNN_cup_combined_data['HeatingQC'].map(knn_HeatingQC)



MeMo_combined_data['HeatingQC'] = MeMo_combined_data['HeatingQC'].map(memo_HeatingQC)

MeMo_log_combined_data['HeatingQC'] = MeMo_log_combined_data['HeatingQC'].map(memo_HeatingQC)

MeMo_cup_combined_data['HeatingQC'] = MeMo_cup_combined_data['HeatingQC'].map(memo_HeatingQC)





# Electrical

knn_Electrical = {4.0: 5, 0.0: 4, 1.0: 3, 2.0: 2, 3.0: 1}

memo_Electrical = {'SBrkr': 5, 'FuseA': 4, 'FuseF': 3, 'FuseP': 2, 'Mix': 1}



KNN_combined_data['Electrical'] = KNN_combined_data['Electrical'].map(knn_Electrical)

KNN_log_combined_data['Electrical'] = KNN_log_combined_data['Electrical'].map(knn_Electrical)

KNN_cup_combined_data['Electrical'] = KNN_cup_combined_data['Electrical'].map(knn_Electrical)



MeMo_combined_data['Electrical'] = MeMo_combined_data['Electrical'].map(memo_Electrical)

MeMo_log_combined_data['Electrical'] = MeMo_log_combined_data['Electrical'].map(memo_Electrical)

MeMo_cup_combined_data['Electrical'] = MeMo_cup_combined_data['Electrical'].map(memo_Electrical)





# KitchenQual

knn_KitchenQual = {3.0: 3, 2.0: 4, 0.0: 5, 1.0: 2}

memo_KitchenQual = {'TA': 3, 'Gd': 4, 'Ex': 5, 'Fa': 2}



KNN_combined_data['KitchenQual'] = KNN_combined_data['KitchenQual'].map(knn_KitchenQual)

KNN_log_combined_data['KitchenQual'] = KNN_log_combined_data['KitchenQual'].map(knn_KitchenQual)

KNN_cup_combined_data['KitchenQual'] = KNN_cup_combined_data['KitchenQual'].map(knn_KitchenQual)



MeMo_combined_data['KitchenQual'] = MeMo_combined_data['KitchenQual'].map(memo_KitchenQual)

MeMo_log_combined_data['KitchenQual'] = MeMo_log_combined_data['KitchenQual'].map(memo_KitchenQual)

MeMo_cup_combined_data['KitchenQual'] = MeMo_cup_combined_data['KitchenQual'].map(memo_KitchenQual)





# Functional

knn_Functional = {6.0: 8, 3.0: 6, 2.0: 7, 4.0: 5, 0.0: 4, 1.0: 3, 5.0: 2}

memo_Functional = {'Typ': 8, 'Min2': 6, 'Min1': 7, 'Mod': 5, 'Maj1': 4, 'Maj2': 3, 'Sev': 2}



KNN_combined_data['Functional'] = KNN_combined_data['Functional'].map(knn_Functional)

KNN_log_combined_data['Functional'] = KNN_log_combined_data['Functional'].map(knn_Functional)

KNN_cup_combined_data['Functional'] = KNN_cup_combined_data['Functional'].map(knn_Functional)



MeMo_combined_data['Functional'] = MeMo_combined_data['Functional'].map(memo_Functional)

MeMo_log_combined_data['Functional'] = MeMo_log_combined_data['Functional'].map(memo_Functional)

MeMo_cup_combined_data['Functional'] = MeMo_cup_combined_data['Functional'].map(memo_Functional)





# GarageFinish

knn_GarageFinish = {2.0: 2, 1.0: 3, 0.0: 4}

memo_GarageFinish = {'Unf': 2, 'RFn': 3, 'Fin': 4}



KNN_combined_data['GarageFinish'] = KNN_combined_data['GarageFinish'].map(knn_GarageFinish)

KNN_log_combined_data['GarageFinish'] = KNN_log_combined_data['GarageFinish'].map(knn_GarageFinish)

KNN_cup_combined_data['GarageFinish'] = KNN_cup_combined_data['GarageFinish'].map(knn_GarageFinish)



MeMo_combined_data['GarageFinish'] = MeMo_combined_data['GarageFinish'].map(memo_GarageFinish)

MeMo_log_combined_data['GarageFinish'] = MeMo_log_combined_data['GarageFinish'].map(memo_GarageFinish)

MeMo_cup_combined_data['GarageFinish'] = MeMo_cup_combined_data['GarageFinish'].map(memo_GarageFinish)





# GarageQual

knn_GarageQual = {4.0: 4, 1.0: 3, 3.0: 5, 2.0: 2, 0.0: 6}

memo_GarageQual = {'TA': 4, 'Fa': 3, 'Gd': 5, 'Po': 2, 'Ex': 6}



KNN_combined_data['GarageQual'] = KNN_combined_data['GarageQual'].map(knn_GarageQual)

KNN_log_combined_data['GarageQual'] = KNN_log_combined_data['GarageQual'].map(knn_GarageQual)

KNN_cup_combined_data['GarageQual'] = KNN_cup_combined_data['GarageQual'].map(knn_GarageQual)



MeMo_combined_data['GarageQual'] = MeMo_combined_data['GarageQual'].map(memo_GarageQual)

MeMo_log_combined_data['GarageQual'] = MeMo_log_combined_data['GarageQual'].map(memo_GarageQual)

MeMo_cup_combined_data['GarageQual'] = MeMo_cup_combined_data['GarageQual'].map(memo_GarageQual)





# GarageCond

knn_GarageCond = {4.0: 4, 3.0: 3, 1.0: 5, 2.0: 2, 0.0: 6}

memo_GarageCond = {'TA': 4, 'Fa': 3, 'Gd': 5, 'Po': 2, 'Ex': 6}



KNN_combined_data['GarageCond'] = KNN_combined_data['GarageCond'].map(knn_GarageCond)

KNN_log_combined_data['GarageCond'] = KNN_log_combined_data['GarageCond'].map(knn_GarageCond)

KNN_cup_combined_data['GarageCond'] = KNN_cup_combined_data['GarageCond'].map(knn_GarageCond)



MeMo_combined_data['GarageCond'] = MeMo_combined_data['GarageCond'].map(memo_GarageCond)

MeMo_log_combined_data['GarageCond'] = MeMo_log_combined_data['GarageCond'].map(memo_GarageCond)

MeMo_cup_combined_data['GarageCond'] = MeMo_cup_combined_data['GarageCond'].map(memo_GarageCond)





# OverallQual - 不要



# OverallCond - 不要
# データをモデル作成用データ、提出用テストデータに分割

KNN_com_train = KNN_combined_data.loc[:1459, :]

KNN_com_test = KNN_combined_data.loc[1460:, :]

KNN_com_test.reset_index(drop=True, inplace=True)



KNN_log_train = KNN_log_combined_data.loc[:1459, :]

KNN_log_test = KNN_log_combined_data.loc[1460:, :]

KNN_log_test.reset_index(drop=True, inplace=True)



KNN_cup_train = KNN_cup_combined_data.loc[:1459, :]

KNN_cup_test = KNN_cup_combined_data.loc[1460:, :]

KNN_cup_test.reset_index(drop=True, inplace=True)
knn_com_train_x, knn_com_test_x, knn_com_train_y, knn_com_test_y = train_test_split(KNN_com_train, train_y, test_size=0.20, random_state=1)

knn_log_train_x, knn_log_test_x, knn_log_train_y, knn_log_test_y = train_test_split(KNN_log_train, train_y, test_size=0.20, random_state=1)

knn_cup_train_x, knn_cup_test_x, knn_cup_train_y, knn_cup_test_y = train_test_split(KNN_cup_train, train_y, test_size=0.20, random_state=1)
k_co_train_x, k_co_test_x, k_train_y, k_test_y = train_test_split(knn_com_train_x, knn_com_train_y, test_size=0.25, random_state=2)

k_lo_train_x, k_lo_test_x, k_train_y, k_test_y = train_test_split(knn_log_train_x, knn_log_train_y, test_size=0.25, random_state=2)

k_cu_train_x, k_cu_test_x, k_train_y, k_test_y = train_test_split(knn_cup_train_x, knn_cup_train_y, test_size=0.25, random_state=2)
# SVMとか使うため、データを正規化する

test = [k_co_test_x, k_lo_test_x, k_cu_test_x]

train = [k_co_train_x, k_lo_train_x, k_cu_train_x]

for index in range(len(train)):

    ob =  MinMaxScaler()

    ob = ob.fit(train[index].values)

    train[index].iloc[:,:] = ob.transform(train[index].values)

    test[index].iloc[:,:] = ob.transform(test[index].values)



# 目的変数の正規化

ob_y = MinMaxScaler()

ob_y = ob_y.fit(k_train_y.values.reshape(-1, 1))

k_train_mm_y = ob_y.transform(k_train_y.values.reshape(-1, 1)).reshape(876,) # ndarray

k_test_mm_y = ob_y.transform(k_test_y.values.reshape(-1, 1)).reshape(292,) # ndarray
%%time

# パラメータチューニング



# 対象データリスト

test = [k_co_test_x, k_lo_test_x, k_cu_test_x]

train = [k_co_train_x, k_lo_train_x, k_cu_train_x]



# パラメータ設定

param_grid = {'kernel': ['rbf'],

              'C' : [0.001, 0.01, 1, 10, 100,1000],

              'gamma':[0.001, 0.01, 1, 10, 100,1000]}



# グリッドサーチ

grid_search = GridSearchCV(SVR(), param_grid, scoring='r2', cv=5, n_jobs=-1, verbose=3)

grid_search.fit(k_co_train_x, k_train_mm_y)

print('--k_co_train_x--')

print('score: ', grid_search.score(k_co_test_x, k_test_mm_y))

print('best parameter: ', grid_search.best_params_)

print('score for train_data: ', grid_search.best_score_, '\n')



grid_search = GridSearchCV(SVR(), param_grid, scoring='r2', cv=5, n_jobs=-1, verbose=3)

grid_search.fit(k_lo_train_x, k_train_mm_y)

print('--k_lo_train_x--')

print('score: ', grid_search.score(k_lo_test_x, k_test_mm_y))

print('best parameter: ', grid_search.best_params_)

print('score for train_data: ', grid_search.best_score_, '\n')



grid_search = GridSearchCV(SVR(), param_grid, scoring='r2', cv=5, n_jobs=-1, verbose=3)

grid_search.fit(k_cu_train_x, k_train_mm_y)

print('--k_cu_train_x--')

print('score: ', grid_search.score(k_cu_test_x, k_test_mm_y))

print('best parameter: ', grid_search.best_params_)

print('score for train_data: ', grid_search.best_score_, '\n')

# SVR 結果

# --k_co_train_x--

# score:  0.7684488594786518

# best parameter:  {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}

# score for train_data:  0.7432499116982972 

    

# --k_lo_train_x--

# score:  0.7332236098844883

# best parameter:  {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}

# score for train_data:  0.7399968092457179 

    

# --k_cu_train_x--

# score:  0.7835344251160323

# best parameter:  {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}

# score for train_data:  0.7786450508775821 
%%time

# パラメータチューニング



n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)] # 10個

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(start=10, stop=200, num = 10)]

min_samples_split = [2,5,10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]



random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(rf, random_grid, n_iter=100, cv=5, n_jobs=-1, scoring='r2')

rf_random.fit(k_co_train_x, k_train_mm_y)

print('--k_co_train_x--')

print('score: ', rf_random.score(k_co_test_x, k_test_mm_y))

print('best parameter: ', rf_random.best_params_)

print('score for train_data: ', rf_random.best_score_, '\n')



rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(rf, random_grid, n_iter=100, cv=5, n_jobs=-1, scoring='r2')

rf_random.fit(k_lo_train_x, k_train_mm_y)

print('--k_lo_train_x--')

print('score: ', rf_random.score(k_lo_test_x, k_test_mm_y))

print('best parameter: ', rf_random.best_params_)

print('score for train_data: ', rf_random.best_score_, '\n')



rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(rf, random_grid, n_iter=100, cv=5, n_jobs=-1, scoring='r2')

rf_random.fit(k_cu_train_x, k_train_mm_y)

print('--k_cu_train_x--')

print('score: ', rf_random.score(k_cu_test_x, k_test_mm_y))

print('best parameter: ', rf_random.best_params_)

print('score for train_data: ', rf_random.best_score_, '\n')
# Random Forest結果



# --k_co_train_x--

# score:  0.818131125592503

# best parameter:  {'n_estimators': 800, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 94, 'bootstrap': False}

# score for train_data:  0.8547278946187443 

    

# --k_lo_train_x--

# score:  0.8134619117315625

# best parameter:  {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 94, 'bootstrap': False}

# score for train_data:  0.8579022859558515 



# --k_cu_train_x--

# score:  0.8127391888646067

# best parameter:  {'n_estimators': 600, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 157, 'bootstrap': False}

# score for train_data:  0.854126204494909 
# lgb用データ準備

lgb_k_co_train = lgb.Dataset(k_co_train_x, k_train_mm_y)

lgb_k_co_test = lgb.Dataset(k_co_test_x, k_test_mm_y)



lgb_k_lo_train = lgb.Dataset(k_lo_train_x, k_train_mm_y)

lgb_k_lo_test =lgb.Dataset(k_lo_test_x, k_test_mm_y)



lgb_k_cu_train = lgb.Dataset(k_cu_train_x, k_train_mm_y)

lgb_k_cu_test = lgb.Dataset(k_cu_test_x, k_test_mm_y)
# optuna呼び出し関数作成

def do_optuna(lgb_train, lgb_test, test_x, test_y, timeout=600):

    

    # optunaの目的関数

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

        # 予測と検証

        predicted = gbm.predict(test_x)

        r2 = r2_score(test_y, predicted)

        

        return r2

    

    study_gbm = optuna.create_study()

    study_gbm.optimize(objective, timeout=timeout)

    

    return study_gbm
lgb_train = [lgb_k_co_train, lgb_k_lo_train, lgb_k_cu_train]

lgb_test = [lgb_k_co_test, lgb_k_lo_test, lgb_k_cu_test]



test = [k_co_test_x, k_lo_test_x, k_cu_test_x]



name = ['lgb_k_co_train', 'lgb_k_lo_train', 'lgb_k_cu_train']



# k_train_mm_y

# k_test_mm_y

result_name_list = []

result_score_list = []

result_param_list = []

for index in range(len(lgb_train)):

    study_gbm = do_optuna(lgb_train[index],lgb_test[index], test[index], k_test_mm_y, 600)

    result_name_list.append(name[index])

    result_score_list.append(study_gbm.best_value)

    result_param_list.append(study_gbm.best_params)
for index in range(len(result_name_list)):

    print('--',result_name_list[index],'--')

    print('r2_score: ', result_score_list[index])

    print('parameter: ', result_param_list[index], '\n')
# lgb　結果



# -- lgb_k_co_train --

# r2_score:  0.8089451677267533

# parameter:  {'lambda_l1': 2.939485828631686, 'lambda_l2': 1.1455170853556762e-08, 'num_leaves': 193, 'feature_fraction': 0.8621259639283465, 'bagging_fraction': 0.8951703750289957, 'bagging_freq': 4, 'min_child_samples': 96} 



# -- lgb_k_lo_train --

# r2_score:  0.7996112555076951

# parameter:  {'lambda_l1': 0.5225005748212116, 'lambda_l2': 1.4539945349808068, 'num_leaves': 248, 'feature_fraction': 0.9988091211353286, 'bagging_fraction': 0.955811103728679, 'bagging_freq': 4, 'min_child_samples': 100}



# -- lgb_k_cu_train --

# r2_score:  0.8074673036311243

# parameter:  {'lambda_l1': 6.556231483463799e-07, 'lambda_l2': 3.5095285062296254e-06, 'num_leaves': 90, 'feature_fraction': 0.5486007641956313, 'bagging_fraction': 0.8841284852138223, 'bagging_freq': 2, 'min_child_samples': 93}
# SVR 結果：訓練データでは、外れ値をTukey法で処理したデータが一番スコアが高い。線形回帰モデルは外れ値の影響がでるという結果がわかる。

# --k_co_train_x--

# score:  0.7684488594786518

# score for train_data:  0.7432499116982972    

# --k_lo_train_x--

# score:  0.7332236098844883

# score for train_data:  0.7399968092457179 

# --k_cu_train_x--

# score:  0.7835344251160323

# score for train_data:  0.7786450508775821 

    

# Random Forest結果：どれも変わらない。RandomForestは外れ値や欠損値にあまり影響を受けないという結果がでている。

# --k_co_train_x--

# score:  0.818131125592503

# score for train_data:  0.8547278946187443 

# --k_lo_train_x--

# score:  0.8134619117315625

# score for train_data:  0.8579022859558515 

# --k_cu_train_x--

# score:  0.8127391888646067

# score for train_data:  0.854126204494909 



# lgb　結果：結果に違いがない。分類木モデルで、欠損値にも強いのでランダムフォレストと同じ結果がでている。

# -- lgb_k_co_train --

# r2_score:  0.8089451677267533

# -- lgb_k_lo_train --

# r2_score:  0.7996112555076951

# -- lgb_k_cu_train --

# r2_score:  0.8074673036311243
# SVMとか使うため、データを正規化する

train = [knn_com_train_x, knn_log_train_x, knn_cup_train_x]

test = [knn_com_test_x, knn_log_test_x, knn_cup_test_x]



for index in range(len(train)):

    ob =  MinMaxScaler()

    ob = ob.fit(train[index].values)

    train[index].iloc[:,:] = ob.transform(train[index].values)

    test[index].iloc[:,:] = ob.transform(test[index].values)



# 目的変数の正規化

ob_y = MinMaxScaler()

ob_y = ob_y.fit(knn_com_train_y.values.reshape(-1, 1))

k_train_mm_y = ob_y.transform(knn_com_train_y.values.reshape(-1, 1)).reshape(1168,) # ndarray

k_test_mm_y = ob_y.transform(knn_com_test_y.values.reshape(-1, 1)).reshape(292,) # ndarray
{'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}

train = [knn_com_train_x, knn_log_train_x, knn_cup_train_x]

test = [knn_com_test_x, knn_log_test_x, knn_cup_test_x]

name = ['knn_com_train_x', 'knn_log_train_x', 'knn_cup_train_x']



k_train_mm_y

k_test_mm_y

for index in range(len(train)):

    svr = SVR(C=10, gamma=0.001, kernel='rbf')

    svr.fit(train[index], k_train_mm_y)

    predicted = svr.predict(test[index])

    r2 = r2_score(k_test_mm_y, predicted)

    

    print('--',name[index],'--')

    print('r2_score: ', r2, '\n')
# SVR 結果

# -- knn_com_train_x --

# r2_score:  0.696768350115631 



# -- knn_log_train_x --

# r2_score:  0.6792301225593462 



# -- knn_cup_train_x --

# r2_score:  0.7108161760783807 
# パラメーター



# --k_co_train_x--

# score:  0.818131125592503

k_co = {'n_estimators': 800, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 94, 'bootstrap': False}

# --k_lo_train_x--

# score:  0.8134619117315625

k_lo = {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 94, 'bootstrap': False}

# --k_cu_train_x--

# score:  0.8127391888646067

k_cu =  {'n_estimators': 600, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 157, 'bootstrap': False} 
train = [knn_com_train_x, knn_log_train_x, knn_cup_train_x]

test = [knn_com_test_x, knn_log_test_x, knn_cup_test_x]

name = ['knn_com_train_x', 'knn_log_train_x', 'knn_cup_train_x']

params_list = [k_co, k_lo, k_cu]



k_train_mm_y

k_test_mm_y

for index in range(len(train)):

    rf_model = RandomForestRegressor(

                    n_estimators = params_list[index]['n_estimators'],

                    min_samples_split = params_list[index]['min_samples_split'],

                    min_samples_leaf = params_list[index]['min_samples_leaf'],

                    max_features = params_list[index]['max_features'],

                    max_depth = params_list[index]['max_depth'],

                    bootstrap = params_list[index]['bootstrap'])



    rf_model.fit(train[index], k_train_mm_y)

    predicted = rf_model.predict(test[index])

    r2 = r2_score(k_test_mm_y, predicted)

    

    print('--',name[index],'--')

    print('r2_score: ', r2, '\n')
# Random Forest 結果



# -- knn_com_train_x --

# r2_score:  0.8846821003428336 



# -- knn_log_train_x --

# r2_score:  0.8789790582604641 



# -- knn_cup_train_x --

# r2_score:  0.8775828852564139 
# パラメーター



# -- lgb_k_co_train --

# r2_score:  0.8089451677267533

lgb_k_co = {'lambda_l1': 2.939485828631686, 'lambda_l2': 1.1455170853556762e-08, 'num_leaves': 193, 'feature_fraction': 0.8621259639283465, 'bagging_fraction': 0.8951703750289957, 'bagging_freq': 4, 'min_child_samples': 96} 

# -- lgb_k_lo_train --

# r2_score:  0.7996112555076951

lgb_k_lo = {'lambda_l1': 0.5225005748212116, 'lambda_l2': 1.4539945349808068, 'num_leaves': 248, 'feature_fraction': 0.9988091211353286, 'bagging_fraction': 0.955811103728679, 'bagging_freq': 4, 'min_child_samples': 100}

# -- lgb_k_cu_train --

# r2_score:  0.8074673036311243

lgb_k_cu = {'lambda_l1': 6.556231483463799e-07, 'lambda_l2': 3.5095285062296254e-06, 'num_leaves': 90, 'feature_fraction': 0.5486007641956313, 'bagging_fraction': 0.8841284852138223, 'bagging_freq': 2, 'min_child_samples': 93}
# lgb用データ準備

lgb_k_co_train = lgb.Dataset(knn_com_train_x, k_train_mm_y)

lgb_k_co_test = lgb.Dataset(knn_com_test_x, k_test_mm_y)



lgb_k_lo_train = lgb.Dataset(knn_log_train_x, k_train_mm_y)

lgb_k_lo_test =lgb.Dataset(knn_log_test_x, k_test_mm_y)



lgb_k_cu_train = lgb.Dataset(knn_cup_train_x, k_train_mm_y)

lgb_k_cu_test = lgb.Dataset(knn_cup_test_x, k_test_mm_y)
train_list = [lgb_k_co_train, lgb_k_lo_train, lgb_k_cu_train]

test_list = [lgb_k_co_test, lgb_k_lo_test, lgb_k_cu_test]

name_list = ['lgb_k_co_train', 'lgb_k_lo_train', 'lgb_k_cu_train']

params_list = [lgb_k_co, lgb_k_lo, lgb_k_cu] 

test_or = [knn_com_test_x, knn_log_test_x, knn_cup_test_x]



for index in range(len(train_list)):

    params_list[index]['objective'] = 'regression'

    params_list[index]['metric'] = 'rmse'



    lgb_regressor = lgb.train(params_list[index],

                              train_list[index],

                              valid_sets=(train_list[index], test_list[index]),

                              num_boost_round=10000,

                              early_stopping_rounds=100,

                              verbose_eval=50)

    predicted = lgb_regressor.predict(test_or[index])

    r2 = r2_score(k_test_mm_y, predicted)



    print('--',name_list[index],'--')

    print('score: ', r2)
# lgb 結果



# -- lgb_k_co_train --

# score:  0.8000620059194133



# -- lgb_k_lo_train --

# score:  0.8505297520338411



# -- lgb_k_cu_train --

# score:  0.8826147308036887
# SVR 結果：訓練データより0.1ポイント下がっているが、傾向は訓練データと同じ。

# -- knn_com_train_x --

# r2_score:  0.696768350115631 

# -- knn_log_train_x --

# r2_score:  0.6792301225593462 

# -- knn_cup_train_x --

# r2_score:  0.7108161760783807 

    

# Random Forest 結果：訓練データと同じ傾向だが、スコアがすべて上がっている。

# -- knn_com_train_x --

# r2_score:  0.8846821003428336 

# -- knn_log_train_x --

# r2_score:  0.8789790582604641 

# -- knn_cup_train_x --

# r2_score:  0.8775828852564139 

    

# lgb 結果：訓練データと違い、前処理の違いでスコアに大きく開きがでている。よくわからない。

# -- lgb_k_co_train --

# score:  0.8000620059194133

# -- lgb_k_lo_train --

# score:  0.8505297520338411

# -- lgb_k_cu_train --

# score:  0.8826147308036887
# データをモデル作成用データ、提出用テストデータに分割

MeMo_com_train = MeMo_combined_data.loc[:1459, :]

MeMo_com_test = MeMo_combined_data.loc[1460:, :]

MeMo_com_test.reset_index(drop=True, inplace=True)



MeMo_log_train = MeMo_log_combined_data.loc[:1459, :]

MeMo_log_test = MeMo_log_combined_data.loc[1460:, :]

MeMo_log_test.reset_index(drop=True, inplace=True)



MeMo_cup_train = MeMo_cup_combined_data.loc[:1459, :]

MeMo_cup_test = MeMo_cup_combined_data.loc[1460:, :]

MeMo_cup_test.reset_index(drop=True, inplace=True)
MeMo_com_train_x, MeMo_com_test_x, MeMo_com_train_y, MeMo_com_test_y = train_test_split(MeMo_com_train, train_y, test_size=0.20, random_state=1)

MeMo_log_train_x, MeMo_log_test_x, MeMo_log_train_y, MeMo_log_test_y = train_test_split(MeMo_log_train, train_y, test_size=0.20, random_state=1)

MeMo_cup_train_x, MeMo_cup_test_x, MeMo_cup_train_y, MeMo_cup_test_y = train_test_split(MeMo_cup_train, train_y, test_size=0.20, random_state=1)
m_co_train_x, m_co_test_x, m_train_y, m_test_y = train_test_split(MeMo_com_train_x, MeMo_com_train_y, test_size=0.25, random_state=2)

m_lo_train_x, m_lo_test_x, m_train_y, m_test_y = train_test_split(MeMo_log_train_x, MeMo_log_train_y, test_size=0.25, random_state=2)

m_cu_train_x, m_cu_test_x, m_train_y, m_test_y = train_test_split(MeMo_cup_train_x, MeMo_cup_train_y, test_size=0.25, random_state=2)
# SVMとか使うため、データを正規化する

test = [m_co_test_x, m_lo_test_x, m_cu_test_x]

train = [m_co_train_x, m_lo_train_x, m_cu_train_x]

for index in range(len(train)):

    ob =  MinMaxScaler()

    ob = ob.fit(train[index].values)

    train[index].iloc[:,:] = ob.transform(train[index].values)

    test[index].iloc[:,:] = ob.transform(test[index].values)



# 目的変数の正規化

ob_y = MinMaxScaler()

ob_y = ob_y.fit(m_train_y.values.reshape(-1, 1))

m_train_mm_y = ob_y.transform(m_train_y.values.reshape(-1, 1)).reshape(876,) # ndarray

m_test_mm_y = ob_y.transform(m_test_y.values.reshape(-1, 1)).reshape(292,) # ndarray
%%time



test = [m_co_test_x, m_lo_test_x, m_cu_test_x]

train = [m_co_train_x, m_lo_train_x, m_cu_train_x]



# パラメータ設定

param_grid = {'kernel': ['rbf'],

              'C' : [0.001, 0.01, 1, 10, 100,1000],

              'gamma':[0.001, 0.01, 1, 10, 100,1000]}



grid_search = GridSearchCV(SVR(), param_grid, scoring='r2', cv=5, n_jobs=-1, verbose=3)

grid_search.fit(m_co_train_x, m_train_mm_y)

print('--m_co_train_x--')

print('score: ', grid_search.score(m_co_test_x, m_test_mm_y))

print('best parameter: ', grid_search.best_params_)

print('score for train_data: ', grid_search.best_score_, '\n')



grid_search = GridSearchCV(SVR(), param_grid, scoring='r2', cv=5, n_jobs=-1, verbose=3)

grid_search.fit(m_lo_train_x, m_train_mm_y)

print('--m_lo_train_x--')

print('score: ', grid_search.score(m_lo_test_x, m_test_mm_y))

print('best parameter: ', grid_search.best_params_)

print('score for train_data: ', grid_search.best_score_, '\n')



grid_search = GridSearchCV(SVR(), param_grid, scoring='r2', cv=5, n_jobs=-1, verbose=3)

grid_search.fit(m_cu_train_x, m_train_mm_y)

print('--m_cu_train_x--')

print('score: ', grid_search.score(m_cu_test_x, m_test_mm_y))

print('best parameter: ', grid_search.best_params_)

print('score for train_data: ', grid_search.best_score_, '\n')

# 結果

# --m_co_train_x--

# score:  0.7658823098119655

# best parameter:  {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}

# score for train_data:  0.743871359723415 



# --m_lo_train_x--

# score:  0.7262347654101845

# best parameter:  {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}

# score for train_data:  0.7411425893834339 

    

# --m_cu_train_x--

# score:  0.782641316481939

# best parameter:  {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}

# score for train_data:  0.7781370828846981 
%%time



n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)] # 10個

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(start=10, stop=200, num = 10)]

min_samples_split = [2,5,10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]



random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(rf, random_grid, n_iter=100, cv=5, n_jobs=-1, scoring='r2')

rf_random.fit(m_co_train_x, m_train_mm_y)

print('--m_co_train_x--')

print('score: ', rf_random.score(m_co_test_x, m_test_mm_y))

print('best parameter: ', rf_random.best_params_)

print('score for train_data: ', rf_random.best_score_, '\n')



rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(rf, random_grid, n_iter=100, cv=5, n_jobs=-1, scoring='r2')

rf_random.fit(m_lo_train_x, m_train_mm_y)

print('--m_lo_train_x--')

print('score: ', rf_random.score(m_lo_test_x, m_test_mm_y))

print('best parameter: ', rf_random.best_params_)

print('score for train_data: ', rf_random.best_score_, '\n')



rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(rf, random_grid, n_iter=100, cv=5, n_jobs=-1, scoring='r2')

rf_random.fit(m_cu_train_x, m_train_mm_y)

print('--m_cu_train_x--')

print('score: ', rf_random.score(m_cu_test_x, m_test_mm_y))

print('best parameter: ', rf_random.best_params_)

print('score for train_data: ', rf_random.best_score_, '\n')
# 結果

# --m_co_train_x--

# score:  0.8169809692704955

# best parameter:  {'n_estimators': 800, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 115, 'bootstrap': False}

# score for train_data:  0.8576439708953709 

    

# --m_lo_train_x--

# score:  0.8200414735238732

# best parameter:  {'n_estimators': 1400, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 178, 'bootstrap': False}

# score for train_data:  0.8600281729714944 

    

# --m_cu_train_x--

# score:  0.81425928442042

# best parameter:  {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 115, 'bootstrap': False}

# score for train_data:  0.8554667647489105 
# データ準備

lgb_m_co_train = lgb.Dataset(m_co_train_x, m_train_mm_y)

lgb_m_co_test = lgb.Dataset(m_co_test_x, m_test_mm_y)



lgb_m_lo_train = lgb.Dataset(m_lo_train_x, m_train_mm_y)

lgb_m_lo_test =lgb.Dataset(m_lo_test_x, m_test_mm_y)



lgb_m_cu_train = lgb.Dataset(m_cu_train_x, m_train_mm_y)

lgb_m_cu_test = lgb.Dataset(m_cu_test_x, m_test_mm_y)
# optuna呼び出し関数作成

def do_optuna(lgb_train, lgb_test, test_x, test_y, timeout=600):

    

    # optunaの目的関数

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

        # 予測と検証

        predicted = gbm.predict(test_x)

        r2 = r2_score(test_y, predicted)

        

        return r2

    

    study_gbm = optuna.create_study()

    study_gbm.optimize(objective, timeout=timeout)

    

    return study_gbm
lgb_train = [lgb_m_co_train, lgb_m_lo_train,lgb_m_cu_train]

lgb_test = [lgb_m_co_test, lgb_m_lo_test, lgb_m_cu_test]



test = [m_co_test_x, m_lo_test_x, m_cu_test_x]



name = ['lgb_m_co_train', 'lgb_m_lo_train', 'lgb_m_cu_train']



# m_train_mm_y

# m_test_mm_y



for index in range(len(lgb_train)):

    study_gbm = do_optuna(lgb_train[index],lgb_test[index], test[index], m_test_mm_y, 600)

    print('--',name[index],'--')

    print('r2_score: ', study_gbm.best_value)

    print('parameter: ', study_gbm.best_params, '\n')
# 結果

# --lgb_m_co_train--

# score:  0.8067264612699065

# parameter:  {'lambda_l1': 1.2959565489530635e-08, 'lambda_l2': 0.009287913965885396, 'num_leaves': 106, 'feature_fraction': 0.774699374099335, 'bagging_fraction': 0.7034573101976733, 'bagging_freq': 1, 'min_child_samples': 100} 



# --lgb_m_lo_train--

# score:  0.8087722110527145

# parameter:  {'lambda_l1': 0.00523995674721704, 'lambda_l2': 1.0503667299469358e-05, 'num_leaves': 151, 'feature_fraction': 0.4280809845134744, 'bagging_fraction': 0.9825559362975935, 'bagging_freq': 4, 'min_child_samples': 88} 



# --lgb_m_cu_train--

# score:  0.8036370558751217

# parameter:  {'lambda_l1': 0.00027794119069172866, 'lambda_l2': 0.005538594560966486, 'num_leaves': 251, 'feature_fraction': 0.5050381951744014, 'bagging_fraction': 0.502343739465335, 'bagging_freq': 7, 'min_child_samples': 90} 

# SVR 結果：KNNと同じ結果で、かつ欠損値埋め方法の違いによる結果にも違いがない。

# --m_co_train_x--

# score:  0.7658823098119655

# score for train_data:  0.743871359723415 

# --m_lo_train_x--

# score:  0.7262347654101845

# score for train_data:  0.7411425893834339 

# --m_cu_train_x--

# score:  0.782641316481939

# score for train_data:  0.7781370828846981 



    

# Random Forest 結果：KNNと同じ結果で、かつ欠損値埋め方法の違いによる結果にも違いがない。

# --m_co_train_x--

# score:  0.8169809692704955

# score for train_data:  0.8576439708953709    

# --m_lo_train_x--

# score:  0.8200414735238732

# score for train_data:  0.8600281729714944    

# --m_cu_train_x--

# score:  0.81425928442042

# score for train_data:  0.8554667647489105 



    

# lgb 結果：KNNと同じ結果で、かつ欠損値埋め方法の違いによる結果にも違いがない。

# --lgb_m_co_train--

# score:  0.8067264612699065

# --lgb_m_lo_train--

# score:  0.8087722110527145

# --lgb_m_cu_train--

# score:  0.8036370558751217

# SVMとか使うため、データを正規化する

train = [MeMo_com_train_x, MeMo_log_train_x, MeMo_cup_train_x]

test = [MeMo_com_test_x, MeMo_log_test_x, MeMo_cup_test_x]



for index in range(len(train)):

    ob =  MinMaxScaler()

    ob = ob.fit(train[index].values)

    train[index].iloc[:,:] = ob.transform(train[index].values)

    test[index].iloc[:,:] = ob.transform(test[index].values)



# 目的変数の正規化

ob_y = MinMaxScaler()

ob_y = ob_y.fit(MeMo_com_train_y.values.reshape(-1, 1))

m_train_mm_y = ob_y.transform(MeMo_com_train_y.values.reshape(-1, 1)).reshape(1168,) # ndarray

m_test_mm_y = ob_y.transform(MeMo_com_test_y.values.reshape(-1, 1)).reshape(292,) # ndarray
{'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}

train = [MeMo_com_train_x, MeMo_log_train_x, MeMo_cup_train_x]

test = [MeMo_com_test_x, MeMo_log_test_x, MeMo_cup_test_x]

name = ['MeMo_com_train_x', 'MeMo_log_train_x', 'MeMo_cup_train_x']



m_train_mm_y

m_test_mm_y

for index in range(len(train)):

    svr = SVR(C=10, gamma=0.001, kernel='rbf')

    svr.fit(train[index], m_train_mm_y)

    predicted = svr.predict(test[index])

    r2 = r2_score(m_test_mm_y, predicted)

    

    print('--',name[index],'--')

    print('r2_score: ', r2, '\n')
# 結果

# -- MeMo_com_train_x --

# r2_score:  0.6935284903582691 



# -- MeMo_log_train_x --

# r2_score:  0.6809276016584709 



# -- MeMo_cup_train_x --

# r2_score:  0.7104079465112133 
# パラメータ



# --m_co_train_x--

# score:  0.8169809692704955

m_co = {'n_estimators': 800, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 115, 'bootstrap': False}   

# --m_lo_train_x--

# score:  0.8200414735238732

m_lo = {'n_estimators': 1400, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 178, 'bootstrap': False}   

# --m_cu_train_x--

# score:  0.81425928442042

m_cu = {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 115, 'bootstrap': False}
train = [MeMo_com_train_x, MeMo_log_train_x, MeMo_cup_train_x]

test = [MeMo_com_test_x, MeMo_log_test_x, MeMo_cup_test_x]

name = ['MeMo_com_train_x', 'MeMo_log_train_x', 'MeMo_cup_train_x']

params_list = [m_co, m_lo, m_cu]



m_train_mm_y

m_test_mm_y

for index in range(len(train)):

    rf_model = RandomForestRegressor(

                    n_estimators = params_list[index]['n_estimators'],

                    min_samples_split = params_list[index]['min_samples_split'],

                    min_samples_leaf = params_list[index]['min_samples_leaf'],

                    max_features = params_list[index]['max_features'],

                    max_depth = params_list[index]['max_depth'],

                    bootstrap = params_list[index]['bootstrap'])



    rf_model.fit(train[index], m_train_mm_y)

    predicted = rf_model.predict(test[index])

    r2 = r2_score(m_test_mm_y, predicted)

    

    print('--',name[index],'--')

    print('r2_score: ', r2, '\n')
# 結果

# -- MeMo_com_train_x --

# r2_score:  0.8882536675172467 



# -- MeMo_log_train_x --

# r2_score:  0.8933828519794719 



# -- MeMo_cup_train_x --

# r2_score:  0.8769570070448025 
# パラメータ



# --lgb_m_co_train--

# score:  0.8067264612699065

lgb_m_co = {'lambda_l1': 1.2959565489530635e-08, 'lambda_l2': 0.009287913965885396, 'num_leaves': 106, 'feature_fraction': 0.774699374099335, 'bagging_fraction': 0.7034573101976733, 'bagging_freq': 1, 'min_child_samples': 100} 



# --lgb_m_lo_train--

# score:  0.8087722110527145

lgb_m_lo = {'lambda_l1': 0.00523995674721704, 'lambda_l2': 1.0503667299469358e-05, 'num_leaves': 151, 'feature_fraction': 0.4280809845134744, 'bagging_fraction': 0.9825559362975935, 'bagging_freq': 4, 'min_child_samples': 88} 



# --lgb_m_cu_train--

# score:  0.8036370558751217

lgb_m_cu = {'lambda_l1': 0.00027794119069172866, 'lambda_l2': 0.005538594560966486, 'num_leaves': 251, 'feature_fraction': 0.5050381951744014, 'bagging_fraction': 0.502343739465335, 'bagging_freq': 7, 'min_child_samples': 90} 

# lgb用データ準備

lgb_m_co_train = lgb.Dataset(MeMo_com_train_x, m_train_mm_y)

lgb_m_co_test = lgb.Dataset(MeMo_com_test_x, m_test_mm_y)



lgb_m_lo_train = lgb.Dataset(MeMo_log_train_x, m_train_mm_y)

lgb_m_lo_test =lgb.Dataset(MeMo_log_test_x, m_test_mm_y)



lgb_m_cu_train = lgb.Dataset(MeMo_cup_train_x, m_train_mm_y)

lgb_m_cu_test = lgb.Dataset(MeMo_cup_test_x, m_test_mm_y)
# モデリング



train_list = [lgb_m_co_train, lgb_m_lo_train, lgb_m_cu_train]

test_list = [lgb_m_co_test, lgb_m_lo_test, lgb_m_cu_test]

name_list = ['lgb_m_co_train', 'lgb_m_lo_train', 'lgb_m_cu_train']

params_list = [lgb_m_co, lgb_m_lo, lgb_m_cu] 

test_or = [MeMo_com_test_x, MeMo_log_test_x, MeMo_cup_test_x]



for index in range(len(train_list)):

    params_list[index]['objective'] = 'regression'

    params_list[index]['metric'] = 'rmse'



    lgb_regressor = lgb.train(params_list[index],

                              train_list[index],

                              valid_sets=(train_list[index], test_list[index]),

                              num_boost_round=10000,

                              early_stopping_rounds=100,

                              verbose_eval=50)

    predicted = lgb_regressor.predict(test_or[index])

    r2 = r2_score(m_test_mm_y, predicted)



    print('--',name_list[index],'--')

    print('score: ', r2)
# 結果

# -- lgb_m_co_train --

# score:  0.8601542977993738



# -- lgb_m_lo_train --

# score:  0.8856335731404262



# -- lgb_m_cu_train --

# score:  0.8304742584590994
# SVR 結果：傾向・スコアともにKNNで埋めたデータと同じ結果

# -- MeMo_com_train_x --

# r2_score:  0.6935284903582691 

# -- MeMo_log_train_x --

# r2_score:  0.6809276016584709 

# -- MeMo_cup_train_x --

# r2_score:  0.7104079465112133 



# Random Forest 結果：KNNより0.01ポイント高いが、傾向はKNNと同じ

# -- MeMo_com_train_x --

# r2_score:  0.8882536675172467 

# -- MeMo_log_train_x --

# r2_score:  0.8933828519794719 

# -- MeMo_cup_train_x --

# r2_score:  0.8769570070448025 

    

# lgb 結果：KNNと逆の結果。よくわからない

# -- lgb_m_co_train --

# score:  0.8601542977993738

# -- lgb_m_lo_train --

# score:  0.8856335731404262

# -- lgb_m_cu_train --

# score:  0.8304742584590994
# データをモデル作成用データ、提出用テストデータに分割

# KNN

KNN_com_train = KNN_combined_data.loc[:1459, :]

KNN_com_test = KNN_combined_data.loc[1460:, :]

KNN_com_test.reset_index(drop=True, inplace=True)



KNN_log_train = KNN_log_combined_data.loc[:1459, :]

KNN_log_test = KNN_log_combined_data.loc[1460:, :]

KNN_log_test.reset_index(drop=True, inplace=True)



KNN_cup_train = KNN_cup_combined_data.loc[:1459, :]

KNN_cup_test = KNN_cup_combined_data.loc[1460:, :]

KNN_cup_test.reset_index(drop=True, inplace=True)



# MeMo

MeMo_com_train = MeMo_combined_data.loc[:1459, :]

MeMo_com_test = MeMo_combined_data.loc[1460:, :]

MeMo_com_test.reset_index(drop=True, inplace=True)



MeMo_log_train = MeMo_log_combined_data.loc[:1459, :]

MeMo_log_test = MeMo_log_combined_data.loc[1460:, :]

MeMo_log_test.reset_index(drop=True, inplace=True)



MeMo_cup_train = MeMo_cup_combined_data.loc[:1459, :]

MeMo_cup_test = MeMo_cup_combined_data.loc[1460:, :]

MeMo_cup_test.reset_index(drop=True, inplace=True)
train_y = train_df['SalePrice'] # target value
# SVMとか使うため、データを正規化する

train = [KNN_com_train, KNN_cup_train]

test = [KNN_com_test, KNN_cup_test]



for index in range(len(train)):

    ob =  MinMaxScaler()

    ob = ob.fit(train[index].values)

    train[index].iloc[:,:] = ob.transform(train[index].values)

    test[index].iloc[:,:] = ob.transform(test[index].values)



# 目的変数の正規化

ob_y = MinMaxScaler()

ob_y = ob_y.fit(train_y.values.reshape(-1, 1))

train_mm_y = ob_y.transform(train_y.values.reshape(-1, 1)).reshape(1460,) # 訓練データは1460レコード
# -- KNN_com_train

# モデリング、予測

svr = SVR(C=10, gamma=0.001, kernel='rbf')

svr.fit(KNN_com_train, train_mm_y)

predicted = svr.predict(KNN_com_test)

predicted_price = ob_y.inverse_transform(predicted.reshape(-1, 1)).reshape(1459,) # 提出データは1459レコード

# 提出データ吐き出し

predicted_svr_knn_com = pd.DataFrame({'Id':test_df['Id'], 'SalePrice': predicted_price})

predicted_svr_knn_com.to_csv('/kaggle/working/svr_knn_com_0601.csv', index=False)



# -- KNN_cup_train

# モデリング、予測

svr = SVR(C=10, gamma=0.001, kernel='rbf')

svr.fit(KNN_cup_train, train_mm_y)

predicted = svr.predict(KNN_cup_test)

predicted_price = ob_y.inverse_transform(predicted.reshape(-1, 1)).reshape(1459,) # 提出データは1459レコード

# 提出データ吐き出し

predicted_svr_knn_cup = pd.DataFrame({'Id':test_df['Id'], 'SalePrice': predicted_price})

predicted_svr_knn_cup.to_csv('/kaggle/working/svr_knn_cup_0601.csv', index=False)

# パブリックスコア(RMSEスコア)：

# knn_com :0.23727

# knn_cup : 0.26670
# データを正規化する

train = [KNN_com_train, MeMo_com_train]

test = [KNN_com_test, MeMo_com_test]



for index in range(len(train)):

    ob =  MinMaxScaler()

    ob = ob.fit(train[index].values)

    train[index].iloc[:,:] = ob.transform(train[index].values)

    test[index].iloc[:,:] = ob.transform(test[index].values)



# 目的変数の正規化

ob_y = MinMaxScaler()

ob_y = ob_y.fit(train_y.values.reshape(-1, 1))

train_mm_y = ob_y.transform(train_y.values.reshape(-1, 1)).reshape(1460,) # 訓練データは1460レコード
# パラメータ



# --KNN_com_train--

k_co = {'n_estimators': 800, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 94, 'bootstrap': False}



# --MeMo_com_train--

m_co = {'n_estimators': 800, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 115, 'bootstrap': False}
# モデリングと予測



# --KNN_com_train

rf_model = RandomForestRegressor(

                n_estimators = k_co['n_estimators'],

                min_samples_split = k_co['min_samples_split'],

                min_samples_leaf = k_co['min_samples_leaf'],

                max_features = k_co['max_features'],

                max_depth = k_co['max_depth'],

                bootstrap = k_co['bootstrap'])



rf_model.fit(KNN_com_train, train_mm_y)

predicted = rf_model.predict(KNN_com_test)

predicted_price = ob_y.inverse_transform(predicted.reshape(-1, 1)).reshape(1459,) # 提出データは1459レコード

# 提出データ吐き出し

predicted_rf_knn_com = pd.DataFrame({'Id':test_df['Id'], 'SalePrice': predicted_price})

predicted_rf_knn_com.to_csv('/kaggle/working/rf_knn_com_0601.csv', index=False)



# --MeMo_com_train

rf_model = RandomForestRegressor(

                n_estimators = m_co['n_estimators'],

                min_samples_split = m_co['min_samples_split'],

                min_samples_leaf = m_co['min_samples_leaf'],

                max_features = m_co['max_features'],

                max_depth = m_co['max_depth'],

                bootstrap = m_co['bootstrap'])



rf_model.fit(MeMo_com_train, train_mm_y)

predicted = rf_model.predict(MeMo_com_test)

predicted_price = ob_y.inverse_transform(predicted.reshape(-1, 1)).reshape(1459,) # 提出データは1459レコード

# 提出データ吐き出し

predicted_rf_memo_com = pd.DataFrame({'Id':test_df['Id'], 'SalePrice': predicted_price})

predicted_rf_memo_com.to_csv('/kaggle/working/rf_memo_com_0601.csv', index=False)
# パブリックスコア(RMSE)：

# MeMo_com: 0.14648

# knn_com: 0.14936
# データを正規化する

train = [KNN_com_train, KNN_log_train, KNN_cup_train, MeMo_com_train, MeMo_log_train, MeMo_cup_train]

test = [KNN_com_test, KNN_log_test, KNN_cup_test, MeMo_com_test, MeMo_log_test, MeMo_cup_test]



for index in range(len(train)):

    ob =  MinMaxScaler()

    ob = ob.fit(train[index].values)

    train[index].iloc[:,:] = ob.transform(train[index].values)

    test[index].iloc[:,:] = ob.transform(test[index].values)



# 目的変数の正規化

ob_y = MinMaxScaler()

ob_y = ob_y.fit(train_y.values.reshape(-1, 1))

train_mm_y = ob_y.transform(train_y.values.reshape(-1, 1)).reshape(1460,) # 訓練データは1460レコード
# パラメータ



# -- lgb_k_co_train --

lgb_k_co = {'lambda_l1': 2.939485828631686, 'lambda_l2': 1.1455170853556762e-08, 'num_leaves': 193, 'feature_fraction': 0.8621259639283465, 'bagging_fraction': 0.8951703750289957, 'bagging_freq': 4, 'min_child_samples': 96} 

# -- lgb_k_lo_train --

lgb_k_lo = {'lambda_l1': 0.5225005748212116, 'lambda_l2': 1.4539945349808068, 'num_leaves': 248, 'feature_fraction': 0.9988091211353286, 'bagging_fraction': 0.955811103728679, 'bagging_freq': 4, 'min_child_samples': 100}

# -- lgb_k_cu_train --

lgb_k_cu = {'lambda_l1': 6.556231483463799e-07, 'lambda_l2': 3.5095285062296254e-06, 'num_leaves': 90, 'feature_fraction': 0.5486007641956313, 'bagging_fraction': 0.8841284852138223, 'bagging_freq': 2, 'min_child_samples': 93}



# --lgb_m_co_train--

lgb_m_co = {'lambda_l1': 1.2959565489530635e-08, 'lambda_l2': 0.009287913965885396, 'num_leaves': 106, 'feature_fraction': 0.774699374099335, 'bagging_fraction': 0.7034573101976733, 'bagging_freq': 1, 'min_child_samples': 100} 

# --lgb_m_lo_train--

lgb_m_lo = {'lambda_l1': 0.00523995674721704, 'lambda_l2': 1.0503667299469358e-05, 'num_leaves': 151, 'feature_fraction': 0.4280809845134744, 'bagging_fraction': 0.9825559362975935, 'bagging_freq': 4, 'min_child_samples': 88} 

# --lgb_m_cu_train--

lgb_m_cu = {'lambda_l1': 0.00027794119069172866, 'lambda_l2': 0.005538594560966486, 'num_leaves': 251, 'feature_fraction': 0.5050381951744014, 'bagging_fraction': 0.502343739465335, 'bagging_freq': 7, 'min_child_samples': 90} 
# データ準備

lgb_k_co_train = lgb.Dataset(KNN_com_train, train_mm_y)

lgb_k_lo_train = lgb.Dataset(KNN_log_train, train_mm_y)

lgb_k_cu_train = lgb.Dataset(KNN_cup_train, train_mm_y)



lgb_m_co_train = lgb.Dataset(MeMo_com_train, train_mm_y)

lgb_m_lo_train = lgb.Dataset(MeMo_log_train, train_mm_y)

lgb_m_cu_train = lgb.Dataset(MeMo_cup_train, train_mm_y)
train_list = [lgb_k_co_train, lgb_k_lo_train, lgb_k_cu_train, lgb_m_co_train, lgb_m_lo_train, lgb_m_cu_train]

test_list = [KNN_com_test, KNN_log_test, KNN_cup_test, MeMo_com_test, MeMo_log_test, MeMo_cup_test]

name_list = ['lgb_k_com_', 'lgb_k_log_', 'lgb_k_cup_', 'lgb_m_com_', 'lgb_m_log_', 'lgb_m_cup_']

params_list = [lgb_k_co, lgb_k_lo, lgb_k_cu, lgb_m_co, lgb_m_lo, lgb_m_cu]



for index in range(len(train_list)):

    params_list[index]['objective'] = 'regression'

    params_list[index]['metric'] = 'rmse'



    lgb_regressor = lgb.train(params_list[index],

                              train_list[index],

                              num_boost_round=10000,

#                               early_stopping_rounds=100,

                              verbose_eval=50)

    

    predicted = lgb_regressor.predict(test_list[index])

    predicted_price = ob_y.inverse_transform(predicted.reshape(-1, 1)).reshape(1459,) # 提出データは1459レコード

    # 提出データ吐き出し

    predicted_df = pd.DataFrame({'Id':test_df['Id'], 'SalePrice': predicted_price})

    file_name = '/kaggle/working/' + name_list[index] + '0601.csv'

    predicted_df.to_csv(file_name, index=False)
# label encoding用メモ



# # ExterQual

# TA(3.0) - 3

# Gd(2.0) - 4

# Ex(0.0) - 5

# Fa(1.0) - 2



# # ExterCond

# TA(4.0) - 3

# Gd(2.0) - 4

# Ex(0.0) - 5

# Fa(1.0) - 2

# Po(3.0) - 1



# # BsmtQual

# TA(3.0) - 3

# Gd(2.0) - 4

# Ex(0.0) - 5

# Fa(1.0) - 2



# # BsmtCond

# TA(3.0) - 3

# Gd(1.0) - 4

# Fa(0.0) - 2

# Po(2.0) - 1



# # BsmtExposure

# No(3.0) - 2

# Av(0.0) - 4

# Gd(1.0) - 5

# Mn(2.0) - 3



# # BsmtFinType1

# Unf(2.0) - 2

# GLQ(5.0) - 7

# ALQ(0.0) - 6

# Rec(4.0) - 4

# BLQ(1.0) - 5

# LwQ(3.0) - 3



# # BsmtFinType2

# Unf(5.0) - 2

# Rec(4.0) - 4

# LwQ(3.0) - 3

# BLQ(1.0) - 5

# ALQ(0.0) - 6

# GLQ(2.0) - 7



# # HeatingQC

# Ex(0.0) - 5

# TA(4.0) - 3

# Gd(2.0) - 4

# Fa(1.0) - 2

# Po(3.0) - 1



# # Electrical

# SBrkr(4.0) - 5

# FuseA(0.0) - 4

# FuseF(1.0) - 3

# FuseP(2.0) - 2

# Mix(3.0) - 1



# # KitchenQual

# TA(3.0) - 3

# Gd(2.0) - 4

# Ex(0.0) - 5

# Fa(1.0) - 2



# # Functional

# Typ(6.0) - 8

# Min2(3.0) - 6

# Min1(2.0) - 7

# Mod(4.0) - 5

# Maj1(0.0) - 4

# Maj2(1.0) - 3

# Sev(5.0) - 2



# # GarageFinish

# Unf(2.0) - 2

# RFn(1.0) - 3

# Fin(0.0) - 4



# # GarageQual

# TA(4.0) - 4

# Fa (1.0) - 3

# Gd(3.0) - 5

# Po(2.0) - 2

# Ex(0.0) - 6



# # GarageCond

# TA(4.0) - 4

# Fa(3.0) - 3

# Gd(1.0) - 5

# Po(2.0) - 2

# Ex(0.0) - 6


