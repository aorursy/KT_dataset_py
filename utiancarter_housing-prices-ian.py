# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

# Data dictionary can be found here: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
train.shape
test.shape
train.columns
#Separate and drop the 'Id' column

train_ID = train['Id']

test_ID = test['Id']



train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



sns.distplot(train['SalePrice'])
f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(train.corr(), vmax=.8, square=True)
k = 10 #number of variables for heatmap

cols = train.corr().nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
train_T = train[['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF',

                 '1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt']]

sns.set()

sns.pairplot(train_T, size = 2.5)

plt.show()
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

train_T = train[['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF',

                 '1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt']]

sns.set()

sns.pairplot(train_T, size = 2.5)

plt.show()
# Credit to Kaggle Kernal: Stacked Regressions to predict House Prices



from scipy import stats

from scipy.stats import norm, skew



sns.distplot(train['SalePrice'] , fit=norm);

(mu, sigma) = norm.fit(train['SalePrice'])

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
# Credit to Kaggle Kernal: Stacked Regressions to predict House Prices



#We use the numpy function log1p which  applies log(1+x) to all elements of the column

train["SalePrice"] = np.log1p(train["SalePrice"])



#Check the new distribution 

sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
# Separate SalePrice from the rest of the dataset

y_train = train['SalePrice'].values

train.drop('SalePrice', axis=1, inplace=True)
# Combine datasets to perform transformations



ntrain = train.shape[0]

ntest = test.shape[0]

combined = pd.concat([train, test])
combined.shape
missing = combined.isnull().sum()

missing = missing[missing != 0].sort_values(ascending=False)

missing
# Apply Nones to missing values

combined['PoolQC'] = combined['PoolQC'].fillna('None') # No entries treated as no pool

combined['MiscFeature'] = combined['MiscFeature'].fillna('None') # No entries treated as none

combined['Alley'] = combined['Alley'].fillna('None') # No entries treated as having no alley

combined['Fence'] = combined['Fence'].fillna('None') # No entries treated as having no fence

combined['FireplaceQu'] = combined['FireplaceQu'].fillna('None') # No entries treated as having no fireplace

combined['GarageType'] = combined['GarageType'].fillna('None') # No entries treated as having no Garage

combined['GarageFinish'] = combined['GarageFinish'].fillna('None') # No entries treated as having no Garage

combined['GarageQual'] = combined['GarageQual'].fillna('None') # No entries treated as having no Garage

combined['GarageCond'] = combined['GarageCond'].fillna('None') # No entries treated as having no Garage

combined['BsmtExposure'] = combined['BsmtExposure'].fillna('None') # No entries treated as having no Basement

combined['BsmtCond'] = combined['BsmtCond'].fillna('None') # No entries treated as having no Basement

combined['BsmtQual'] = combined['BsmtQual'].fillna('None') # No entries treated as having no Basement

combined['BsmtFinType2'] = combined['BsmtFinType2'].fillna('None') # No entries treated as having no Basement

combined['BsmtFinType1'] = combined['BsmtFinType1'].fillna('None') # No entries treated as having no Basement

combined['MasVnrType'] = combined['MasVnrType'].fillna('None') # No entries treated as having no masonry veneer

combined['MSSubClass'] = combined['MSSubClass'].fillna('None') # No entries treated as having no building class
# Apply zeroes to missing values

combined['GarageYrBlt'] = combined['GarageYrBlt'].fillna(0) # No entries treated as having no Garage-related numerics

combined['GarageCars'] = combined['GarageCars'].fillna(0) # No entries treated as having no Garage-related numerics

combined['GarageArea'] = combined['GarageArea'].fillna(0) # No entries treated as having no Garage-related numerics

combined['BsmtFullBath'] = combined['BsmtFullBath'].fillna(0) # No entries treated as having no Basement-related numerics

combined['BsmtHalfBath'] = combined['BsmtHalfBath'].fillna(0) # No entries treated as having no Basement-related numerics

combined['BsmtFinSF1'] = combined['BsmtFinSF1'].fillna(0) # No entries treated as having no Basement-related numerics

combined['BsmtFinSF2'] = combined['BsmtFinSF2'].fillna(0) # No entries treated as having no Basement-related numerics

combined['BsmtUnfSF'] = combined['BsmtUnfSF'].fillna(0) # No entries treated as having no Basement-related numerics

combined['TotalBsmtSF'] = combined['TotalBsmtSF'].fillna(0) # No entries treated as having no Basement-related numerics

combined['MasVnrArea'] = combined['MasVnrArea'].fillna(0) # No entries treated as having no masonry veneer-related numerics
# Apply stats to missing values

combined['LotFrontage'] = combined.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median())) # No entries given median of their respective neighborhood

combined['MSZoning'] = combined['MSZoning'].fillna(combined['MSZoning'].mode()[0]) # No entries given mode of MSZoning

combined['Electrical'] = combined['Electrical'].fillna(combined['Electrical'].mode()[0]) # No entries given mode of Electrical

combined['KitchenQual'] = combined['KitchenQual'].fillna(combined['KitchenQual'].mode()[0]) # No entries given mode of KitchenQual

combined['Exterior1st'] = combined['Exterior1st'].fillna(combined['Exterior1st'].mode()[0]) # No entries given mode of Exterior1st

combined['Exterior2nd'] = combined['Exterior2nd'].fillna(combined['Exterior2nd'].mode()[0]) # No entries given mode of Exterior2nd

combined['SaleType'] = combined['SaleType'].fillna(combined['SaleType'].mode()[0]) # No entries given mode of SaleType
# Apply strings to missing values

combined['Functional'] = combined['Functional'].fillna('Typ') # No entries treated as having 'Typ'
# Drop features for remaining features

combined.drop(['Utilities'], axis=1, inplace=True)
missing = combined.isnull().sum()

missing = missing[missing != 0].sort_values(ascending=False)

missing
combined['MSSubClass'] = combined['MSSubClass'].apply(str)
# Decided to leave out YrSold and OverallCond from LabelEncoder



from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'MoSold')



for c in cols:

    le = LabelEncoder() 

    le.fit(list(combined[c].values)) 

    combined[c] = le.transform(list(combined[c].values))
combined.shape
# Credit to Kaggle Kernal: Reach Top 10% With Simple Model On Housing Prices



numeric_feats = (combined.dtypes[combined.dtypes != "object"]).index



# Check the skew of all numerical features

skewed_feats = combined[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(15)
# Credit to Kaggle Kernal: Reach Top 10% With Simple Model On Housing Prices



skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #combined[feat] += 1

    combined[feat] = boxcox1p(combined[feat], lam)
combined = pd.get_dummies(combined)

combined.shape
train = combined[:ntrain]

test = combined[ntrain:]
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error
# Credit to Kaggle Kernal: Reach Top 10% With Simple Model On Housing Prices



n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
# Credit to Kaggle Kernal: Reach Top 10% With Simple Model On Housing Prices



KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# Credit to Kaggle Kernal: Reach Top 10% With Simple Model On Housing Prices



lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

score = rmsle_cv(lasso)

print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# Credit to Kaggle Kernal: Reach Top 10% With Simple Model On Housing Prices



ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# Credit to Kaggle Kernal: Reach Top 10% With Simple Model On Housing Prices



GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
RFR = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=100)

RFR.fit(train.values, y_train)  

score = rmsle_cv(RFR)

print("Random Forest score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
LassoMd = lasso.fit(train.values,y_train)

ENetMd = ENet.fit(train.values,y_train)

KRRMd = KRR.fit(train.values,y_train)

GBoostMd = GBoost.fit(train.values,y_train)
finalMd = (np.expm1(LassoMd.predict(test.values)) + np.expm1(ENetMd.predict(test.values)) + np.expm1(KRRMd.predict(test.values)) + np.expm1(GBoostMd.predict(test.values))) / 4

finalMd
submission = pd.DataFrame()

submission['Id'] = test_ID

submission['SalePrice'] = finalMd

submission.to_csv('submission.csv', index=False)