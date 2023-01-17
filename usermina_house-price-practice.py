# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

from sklearn.linear_model import LinearRegression





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data_path="/kaggle/input/house-prices-advanced-regression-techniques/"



%matplotlib inline 



# Any results you write to the current directory are saved as output.
df = pd.read_csv(data_path+"train.csv")
null_val = df.isnull().sum()

percent = 100 * df.isnull().sum() / len(df)

missing_table = pd.concat([null_val, percent], axis=1)

missing_table=missing_table.rename(columns={0: 'missing_num', 1: 'missing_rate'})

missing_table[missing_table['missing_num']!=0]
# test missing

df_test=pd.read_csv(data_path+"test.csv")

null_val = df_test.isnull().sum()

percent = 100 * df_test.isnull().sum() / len(df)

missing_table = pd.concat([null_val, percent], axis=1)

missing_table=missing_table.rename(columns={0: 'missing_num', 1: 'missing_rate'})

missing_table[missing_table['missing_num']!=0]
df.describe()
sns.distplot(df.SalePrice)
df.corr()
# 数字を0で補完

def comp(df):

    na_col_list = df.isnull().sum(

    )[df.isnull().sum() > 0].index.tolist()  # 欠損を含むカラムをリスト化

    na_float_cols = df[na_col_list].dtypes[df[na_col_list].dtypes ==

                                           'float'].index.tolist()  

    na_int_cols = df[na_col_list].dtypes[df[na_col_list].dtypes ==

                                           'int'].index.tolist()

    for na_float_col in na_float_cols:

        df.loc[df[na_float_col].isnull(), na_float_col] = 0.0

    for na_int_col in na_int_cols:

        df.loc[df[na_int_col].isnull(), na_int_col] = 0

    return df

df=comp(df)

df
# SalesPriceとの相関が高い順に上位のデータを表示

num = 60

column="SalePrice"

df = df.select_dtypes(include=[int, float])

corrmat = df.corr()

cols = corrmat.abs().nlargest(num, column)[column].index

print(cols)

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.5)

fig, ax = plt.subplots(figsize=(num, num))

ax.set_ylim(len(cm), 0)

sns.heatmap(cm, cbar=True, annot=True, square=True, ax=ax, fmt='.2f', 

            annot_kws={'size': 15}, yticklabels=cols.values, xticklabels=cols.values)
def zscore_normalization(df, column, mean, std):

    '''

    Zscore normalization column data with argment(mean,std)

    '''

    def zscore(x): return (x - mean) / std

    df[column] = df[column].map(zscore)

    # all element 0

    if std == 0:

        df[column] = 0

    return df





def zscore_normalization_describe(df, df_describe):

    '''

    Zscore normalization df with df_describe(mean,std)

    '''

    columns = df.columns.tolist()

    for c in columns:

        df = zscore_normalization(

            df, c, df_describe.at["mean", c], df_describe.at["std", c])

    return df
# train data set

# 数字のみ残す

df_train=df.select_dtypes(include=['float', 'int'])



# 正解データ

df_teach=df_train['SalePrice']



# 相関係数0.4以上のcolを使用

# cols=['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',

#        'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt',

#        'YearRemodAdd', 'MasVnrArea', 'Fireplaces']

# df_train=df_train.loc[:,cols]



# Id以下の相関の列削除

cols=['Id', 'MiscVal', 'BsmtHalfBath', 'BsmtFinSF2']

df_train=df_train.drop(cols,axis=1)



# 0で補完

df_train=comp(df_train)



#trainで標準化

df_train_describe=df_train.describe()

df_train=zscore_normalization_describe(df_train, df_train_describe)



# 正解連結

df_train['SalePrice']=df_teach



df_train.to_csv("train_dataset.csv",index=False)

df_train
# test dataset



df_test=pd.read_csv(data_path+"test.csv")



df_test=df_test.select_dtypes(include=['float', 'int'])

# cols=['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',

#        'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt',

#        'YearRemodAdd', 'MasVnrArea', 'Fireplaces']

# df_test=df_test.loc[:,cols]



# Id以下の相関の列削除

cols=['Id', 'MiscVal', 'BsmtHalfBath', 'BsmtFinSF2']

df_test=df_test.drop(cols,axis=1)



# 欠損補完

df_test=comp(df_test)



#trainで標準化

df_test=zscore_normalization_describe(df_test, df_train_describe)



df_test.to_csv("test_dataset.csv",index=False)

df_test