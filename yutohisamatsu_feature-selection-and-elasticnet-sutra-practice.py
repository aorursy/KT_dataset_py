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

import numpy as np

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from scipy.stats import skew
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head()
# Select only confirmed features

# See https://www.kaggle.com/jimthompson/house-prices-advanced-regression-techniques/boruta-feature-importance-analysis



features = ['GrLivArea','OverallQual','2ndFlrSF','TotalBsmtSF','1stFlrSF','GarageCars','YearBuilt', 'GarageArea','ExterQual',

            'YearRemodAdd','FireplaceQu','GarageYrBlt','FullBath','MSSubClass','LotArea','Fireplaces','KitchenQual','MSZoning',

            'GarageType','BsmtFinSF1','Neighborhood','BsmtQual','TotRmsAbvGrd','HalfBath','BldgType','GarageFinish',

            'Foundation','BedroomAbvGr','HouseStyle','CentralAir','OpenPorchSF','HeatingQC','BsmtFinType1','BsmtUnfSF',

            'GarageCond','GarageQual','KitchenAbvGr','OverallCond','BsmtCond','MasVnrArea','BsmtFullBath','Exterior1st',

            'Exterior2nd','PavedDrive','WoodDeckSF','LandContour','MasVnrType','BsmtFinType2','Functional','Fence']
train = train[features + ['SalePrice']]

test = test[['Id'] + features]
train.shape
all_data = pd.concat((train.loc[:, 'GrLivArea':'Fence'], test.loc[:,'GrLivArea':'Fence']))
all_data.shape
all_data
# rcParams レイアウト調節

matplotlib.rcParams['figure.figsize'] = (17.0, 7.0)



prices = pd.DataFrame({"price": train['SalePrice'], "log(price+1)": np.log1p(train['SalePrice'])})



prices.hist();
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
all_data.head()
all_data.shape
# ダミー変数

all_data = pd.get_dummies(all_data).reset_index(drop=True)

all_data
#creating matrices for sklearn: sklearnの行と列を作成

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
X_train.shape
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import cross_val_score

from itertools import product # ループ実行を効率化するやつ



# RMSEを測る関数 (モデルごと)

def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring='neg_mean_squared_error', cv = 5))

    return(rmse)
alphas = [0.0005, 0.001, 0.01, 0.03, 0.05, 0.1]

l1_ratios = [1.5, 1.1, 1, 0.9, 0.8, 0.7, 0.5]
cv_elastic = [rmse_cv(ElasticNet(alpha = alpha, l1_ratio=l1_ratio)).mean() 

              for (alpha, l1_ratio) in product(alphas, l1_ratios)]
matplotlib.rcParams['figure.figsize'] = (17.0, 7.0)



idx = list(product(alphas, l1_ratios))



p_cv_elastic = pd.Series(cv_elastic, index = idx);



p_cv_elastic.plot(title = "Validation - Just Do It");

plt.xlabel("alpha - l1_ratio");

plt.ylabel("rmse");
# Zoom in to the first 10 parameter pairs

matplotlib.rcParams['figure.figsize'] = (17.0, 7.0)



idx = list(product(alphas, l1_ratios))[:10]



p_cv_elastic = pd.Series(cv_elastic[:10], index = idx);



p_cv_elastic.plot(title = "Validation - Just Do It");

plt.xlabel("alpha - l1_ratio");

plt.ylabel("rmse");
elastic = ElasticNet(alpha=0.0005, l1_ratio=0.9)
elastic.fit(X_train, y)
# let's look at the residuals as well: 残差

matplotlib.rcParams['figure.figsize'] = (8.0, 8.0)



preds = pd.DataFrame({"preds":elastic.predict(X_train), "true":y})



# 残差

preds["residuals"] = preds["true"] - preds["preds"]



preds.plot(x = "preds", y = "residuals",kind = "scatter");
# RSME(二乗平均平方根誤差)

# 実際の値と予測値のズレが小さければ小さいほど、当てはまりの良いモデルだって言えるよね、そのための指標がRMSEだよね

rmse = np.sqrt(np.mean((preds['true']-preds['preds'])**2))



print ('RMSE: {0:.4f}'.format(rmse))
# 決定係数（R⒉）

# 決定係数 (R2) モデルの当てはまりの良さを示す指標　最も当てはまりの良い場合、1 - 当てはまりの悪い場合、マイナスとなることも



from sklearn.metrics import r2_score



print('R^2 train: %.3f' %  r2_score(preds['true'], preds['preds']))
# 係数,率

coef = pd.Series(elastic.coef_, index = X_train.columns)
coef
print("Elastic Net picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
# 重要な係数

imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)



imp_coef.plot(kind = "barh");

plt.title("Coefficients in the Elastic Net Model");
#let's look at the residuals as well: さっきも見たけど、もう一回残差

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"preds":elastic.predict(X_train), "true":y});

preds["residuals"] = preds["true"] - preds["preds"];

preds.plot(x = "preds", y = "residuals",kind = "scatter");
preds = elastic.predict(X_test)

preds
preds = np.expm1(elastic.predict(X_test))

preds
# 提出

solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

solution.to_csv("elastic.csv", index = False)