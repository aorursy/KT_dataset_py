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
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm.notebook import tqdm as tqdm

from sklearn.metrics import mean_squared_error



from sklearn.ensemble import GradientBoostingClassifier

import lightgbm as lgb

from lightgbm import LGBMRegressor



from pylab import rcParams

from pandas import DataFrame, Series

import matplotlib.pyplot as plt

print(os.getcwd())


df_train = pd.read_csv('/kaggle/input/exam-for-students20200527/train.csv', index_col=0)

df_test = pd.read_csv('/kaggle/input/exam-for-students20200527/test.csv', index_col=0)

df_train.shape, df_test.shape
#df_train.head()

df_test.head()
# 全数値列

num_cols_for_auto_create = df_train.select_dtypes(include='number').columns.values.tolist()

print(num_cols_for_auto_create)
num_cols_for_auto_create = df_train.select_dtypes(include='object').columns.values.tolist()

print(num_cols_for_auto_create)
f = 'TimeToNearestStation'

df_train[f].value_counts()

#df_test[f].value_counts()

# データ揃える必要あり
f = 'FloorPlan'

df_train[f].value_counts()

#df_test[f].value_counts()

# ＋Sとる？？
f = 'FloorAreaRatio'



plt.figure(figsize=[7,7])

df_train[f].hist(density=True, alpha=0.5, bins=30,color='r')

df_test[f].hist(density=True, alpha=0.5, bins=30,color='b')

# testデータに対する可視化を記入してみましょう

plt.xlabel(f)

plt.show()

# SPIKEしている。とか色々
f = 'MinTimeToNearestStation'



plt.figure(figsize=[7,7])

df_train[f].hist(density=True, alpha=0.5, bins=30,color='r')

df_test[f].hist(density=True, alpha=0.5, bins=30,color='b')

# testデータに対する可視化を記入してみましょう

plt.xlabel(f)

plt.show()

# SPIKEしている。とか色々
f = 'MaxTimeToNearestStation'



plt.figure(figsize=[7,7])

df_train[f].hist(density=True, alpha=0.5, bins=30,color='r')

df_test[f].hist(density=True, alpha=0.5, bins=30,color='b')

# testデータに対する可視化を記入してみましょう

plt.xlabel(f)

plt.show()

# SPIKEしている。とか色々
f = 'Area'



plt.figure(figsize=[7,7])

df_train[f].hist(density=True, alpha=0.5, bins=30,color='r')

df_test[f].hist(density=True, alpha=0.5, bins=30,color='b')

# testデータに対する可視化を記入してみましょう

plt.xlabel(f)

plt.show()

# SPIKEしている。とか色々
f = 'Frontage'



plt.figure(figsize=[7,7])

df_train[f].hist(density=True, alpha=0.5, bins=30,color='r')

df_test[f].hist(density=True, alpha=0.5, bins=30,color='b')

# testデータに対する可視化を記入してみましょう

plt.xlabel(f)

plt.show()

# SPIKEしている。とか色々
f = 'TotalFloorArea'



plt.figure(figsize=[7,7])

df_train[f].hist(density=True, alpha=0.5, bins=30,color='r')

df_test[f].hist(density=True, alpha=0.5, bins=30,color='b')

# testデータに対する可視化を記入してみましょう

plt.xlabel(f)

plt.show()

# SPIKEしている。とか色々
y_train = df_train.TradePrice

X_train = df_train.drop(['TradePrice'],axis=1)

X_test=df_test

#緯度経度結合

#X_all = pd.concat([X_train, X_test], axis=0)



df_grp = df_gdp.rename(columns={'State & Local Spending':'Spending', 'Gross State Product':'GrossState', 'Real State Growth %':'RealState', 'Population (million)':'Population'})



#30-60minutes    0.5

#1H-1H30         1.25

#1H30-2H         1.5

#2H-             2.5





dict = {'30-60minutes':0.5, '1H-1H30':1.25, '1H30-2H':1.5, '2H-':2.5}

X_train['TimeToNearestStation'] = X_train['TimeToNearestStation'].replace(dict)

X_test['TimeToNearestStation'] = X_test['TimeToNearestStation'].replace(dict)



X_train['TimeToNearestStation']=X_train['TimeToNearestStation'].astype(float)

X_test['TimeToNearestStation']=X_test['TimeToNearestStation'].astype(float)



X_train['TimeToNearestStation'].value_counts()
# 元のカラムのうち、欠損値を含む列に対して欠損フラグ列を追加する

base_columns = X_train.columns



for col in base_columns:

    if X_train[col].isnull().any():

        X_train[col + '_isna'] = X_train[col].isnull().replace({True : 1, False : 0})

        X_test[col + '_isna'] = X_test[col].isnull().replace({True : 1, False : 0})



X_train.head()
import seaborn as sns

colormap = plt.cm.RdBu

plt.figure(figsize=(30,30))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(X_train.corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)

plt.savefig('heatmap.png')
num_cols = X_train.select_dtypes(include='number').columns.values.tolist()

num_cols
# ==== 数値特徴量生成 ====

# TRAIN.TEST分布は,近い(くっつけて実施)

# RANKGAUSで正規化

num_cols = X_train.select_dtypes(include='number').columns.values.tolist()





from sklearn.preprocessing import quantile_transform

# 複数列のRankGaussによる変換を定義

X_all = pd.concat([X_train, X_test], axis=0)

X_all[num_cols] = quantile_transform(X_all[num_cols] ,n_quantiles=100, random_state=0, output_distribution='normal',copy=True)



X_train = X_all.iloc[:X_train.shape[0], :]

X_test = X_all.iloc[X_train.shape[0]:, :]



for c in num_cols:

    plt.figure(figsize=[10,5])

    plt.subplot(1,2,1)

    X_train[c].hist(density=True, alpha=0.5, bins=20, color="blue")

    X_test[c].hist(density=True, alpha=0.5, bins=20, color="red")

    plt.xlabel(c)

    plt.ylabel('density')

    plt.show()



# dtypeがobjectのカラム名とユニーク数を確認

cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, X_train[col].nunique())

# OneHot

cats_onehot = ['Renovation']



X_train = pd.get_dummies(X_train, columns=cats_onehot)

X_test = pd.get_dummies(X_test, columns=cats_onehot)



X_train
num_cols_for_auto_create = X_train.select_dtypes(include='object').columns.values.tolist()

print(num_cols_for_auto_create)
#ターゲットエンコーディング

target = 'TradePrice'

#list_cols = ['purpose','home_ownership','addr_state']

list_cols =['Type', 'Region', 'Prefecture', 'Municipality', 'DistrictName', 'NearestStation', 'FloorPlan', 'LandShape', 'Structure', 'Use', 'Purpose', 'Direction', 'Classification', 'CityPlanning', 'Remarks']



X_temp = pd.concat([X_train, y_train], axis=1)



for col in list_cols:



    # X_testはX_trainでエンコーディングする

    summary = X_temp.groupby([col])[target].mean()

    print(summary)

#    enc_test = X_test[col].map(summary) 

    X_test[col] = X_test[col].map(summary)



    # X_trainのカテゴリ変数をoofでエンコーディングする

    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([col])[target].mean()

        enc_train.iloc[val_ix] = X_val[col].map(summary)



    X_train[col] = enc_train

    print(X_train[col].head())
num_cols_for_auto_create = X_test.select_dtypes(include='object').columns.values.tolist()

print(num_cols_for_auto_create)
# y_trainに、log(y+1)で変換



y_train_log1p = np.log1p(y_train) 

#y_train = np.expm1(y_train) 

#y_train

# 後で戻すのを忘れずに

#pred = np.expm1(pred)
#全件

clf = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

                            importance_type='split', learning_rate=0.05, max_depth=-1,

                            min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                            n_estimators=9999, n_jobs=-1, num_leaves=15, objective='regression',

                            random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                            subsample=1.0, subsample_for_bin=200000, subsample_freq=0)



clf.fit(X_train, y_train_log1p, eval_metric='rmse')

#testに対して予測する



#y_pred=gs.predict_proba(X_test)[:,1]

y_pred = clf.predict(X_test)

y_pred = pd.DataFrame(np.round(np.expm1(y_pred)))

#y_pred = pd.DataFrame(np.round(y_pred))



submission = pd.read_csv('/kaggle/input/exam-for-students20200527/sample_submission.csv')



submission.TradePrice = y_pred

submission.TradePrice

submission.to_csv('submission.csv', index=False)

submission