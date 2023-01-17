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
import matplotlib

import matplotlib.pyplot as plt

from scipy.stats import skew

from sklearn.linear_model import ElasticNet

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from itertools import product

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.metrics import mean_squared_error
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# ntrain = train.shape[0] // 1460

# ntest = test.shape[0] // 1459

# y_train = train.SalePrice.values

# all_data = pd.concat([train, test]).reset_index(drop=True)
# Select only confirmed features

# See https://www.kaggle.com/jimthompson/house-prices-advanced-regression-techniques/boruta-feature-importance-analysis



features = ['GrLivArea','OverallQual','2ndFlrSF','TotalBsmtSF','1stFlrSF','GarageCars','YearBuilt', 'GarageArea','ExterQual',

            'YearRemodAdd','FireplaceQu','GarageYrBlt','FullBath','MSSubClass','LotArea','Fireplaces','KitchenQual','MSZoning',

            'GarageType','BsmtFinSF1','Neighborhood','BsmtQual','TotRmsAbvGrd','HalfBath','BldgType','GarageFinish',

            'Foundation','BedroomAbvGr','HouseStyle','CentralAir','OpenPorchSF','HeatingQC','BsmtFinType1','BsmtUnfSF',

            'GarageCond','GarageQual','KitchenAbvGr','OverallCond','BsmtCond','MasVnrArea','BsmtFullBath','Exterior1st',

            'Exterior2nd','PavedDrive','WoodDeckSF','LandContour','MasVnrType','BsmtFinType2','Functional','Fence']
train = train[features+['SalePrice']]

test = test[['Id']+features]



# GrLivArea ~ Fence 列の全ての行を

all_data = pd.concat((train.loc[:,'GrLivArea':'Fence'],

                      test.loc[:,'GrLivArea':'Fence']))
all_data.shape
# log transform the target:

train['SalePrice'] = np.log1p(train['SalePrice'])



# log transform skewed numeric features: 歪んだ数値変数

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# 歪度計算

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))



skewed_feats = skewed_feats[skewed_feats > 0.75] # 絞る

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
#impute NA's, numerical features with the median, categorical values with most common value:



# objectを除外 = 数値変数→欠損を中央値で

for feature in all_data.select_dtypes(exclude=['object']).columns:

        all_data[feature].fillna(all_data[feature].median(), inplace = True)

        

# objectを抽出 = カテゴリ変数→欠損を最頻値で

for feature in all_data.select_dtypes(include=['object']).columns: 

        all_data[feature].fillna(all_data[feature].value_counts().idxmax(), inplace = True)
all_data = pd.get_dummies(all_data)

print(all_data.shape)
# train = all_data[:ntrain]

# test = all_data[ntrain:]



X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
train.shape
def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
alphas = [0.0005, 0.001, 0.01, 0.03, 0.05, 0.1]

l1_ratios = [1.5, 1.1, 1, 0.9, 0.8, 0.7, 0.5]
cv_elastic = [rmse_cv(ElasticNet(alpha = alpha, l1_ratio=l1_ratio)).mean() 

            for (alpha, l1_ratio) in product(alphas, l1_ratios)]
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

idx = list(product(alphas, l1_ratios))

p_cv_elastic = pd.Series(cv_elastic, index = idx)

p_cv_elastic.plot(title = "Validation - Just Do It")

plt.xlabel("alpha - l1_ratio")

plt.ylabel("rmse")
# Zoom in to the first 10 parameter pairs

matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

idx = list(product(alphas, l1_ratios))[:10]

p_cv_elastic = pd.Series(cv_elastic[:10], index = idx)

p_cv_elastic.plot(title = "Validation - Just Do It")

plt.xlabel("alpha - l1_ratio")

plt.ylabel("rmse")
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, X_train.values, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
# モデル学習

ENet.fit(X_train.values, y)



# trainデータ 推測

ENet_train_pred = ENet.predict(X_train.values)



# モデルの推測したものを検証 （スコア）

print('RMSLE score on train data:')

print(rmsle(y, ENet_train_pred))
# testデータ 推測

ENet_pred = np.expm1(ENet.predict(X_test.values))
sub = pd.DataFrame()

sub['Id'] = test['Id']

sub['SalePrice'] = ENet_pred

sub.to_csv('submission.csv',index=False)