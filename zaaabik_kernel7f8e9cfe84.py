import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import *

from scipy.stats import skew
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))

fill_na_none_columns = ["PoolQC", "Alley", "Fence", "FireplaceQu", 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']



all_data[fill_na_none_columns] = all_data[fill_na_none_columns].fillna("None")



all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)

    

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)

    

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')



all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data["Alley"] = all_data["Alley"].fillna("None")



all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])



all_data["Functional"] = all_data["Functional"].fillna("Typ")



all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

all_data['MiscFeature'] = all_data['MiscFeature'].fillna("None")

      



all_data = all_data.drop(['Utilities'], axis=1)



# train["MiscFeature"] = train["MiscFeature"].fillna("None")





all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head()

# all_data.isna().sum()
train["SalePrice"] = np.log1p(train["SalePrice"])



numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].skew(skipna=True) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)

all_data = all_data.fillna(all_data.mean())
X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
from sklearn.linear_model import * #Ridge, RidgeCV, ElasticNet, ElasticNetCV, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 10))

    return(rmse)
alphas = [0.05, 0.1, 0.2, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]

ridge = Ridge(alpha =0.1).fit(X_train, y)

alphas = [0.001,0.05, 0.1, 0.2, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_elastic_net = [rmse_cv(ElasticNet(alpha = alpha)).mean() 

            for alpha in alphas]

elastic = ElasticNet(alpha = 0.001).fit(X_train, y)



elastic_preds = np.expm1(elastic.predict(X_test))

ridge_preds = np.expm1(ridge.predict(X_test))

preds = 0.5 * elastic_preds +  ridge_preds * 0.5



predictions = pd.DataFrame({"SalePrice":preds, 'Id': test['Id']})

predictions.to_csv('res.csv', index=False)