import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

%matplotlib inline
T_train = pd.read_csv("../input/train.csv")

T_test = pd.read_csv("../input/test.csv")
T_train.columns
train_x = T_train.drop(["SalePrice"], axis = 1)
df = pd.concat([train_x,T_test],axis = 0)
df.info()
df.head()
df.isnull().sum().sort_values(ascending=False) ##ascending = false gives high values to low
df=df.drop(['Id','MiscFeature','Fence','PoolQC','Alley'],axis=1)   #Including "Id"
num_col = df._get_numeric_data()   #going to get only numaric columns or attributes# Missing value imputation:

num_col.info()
##First let's do for Numarical attributes:



num_nulls = num_col.isnull().sum().sort_values(ascending=False)
def var_summary(x):

    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.quantile(0.01), x.quantile(0.05),x.quantile(0.10),x.quantile(0.25),x.quantile(0.50),x.quantile(0.75), x.quantile(0.90),x.quantile(0.95), x.quantile(0.99),x.max()], 

                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])



num_col.apply(lambda x: var_summary(x)).T
def var_summary(x):

    return pd.Series([x.quantile(0.01), x.quantile(0.05),x.quantile(0.10),x.quantile(0.25),x.quantile(0.50),x.quantile(0.75), x.quantile(0.90),x.quantile(0.95), x.quantile(0.99),x.max()], 

                  index=['P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])



num_col.apply(lambda x: var_summary(x)).T
num_col['LotArea']= num_col['LotArea'].clip_upper(num_col['LotArea'].quantile(0.99))

num_col['LotArea']= num_col['LotArea'].clip_upper(num_col['LotArea'].quantile(0.01))

num_col['MasVnrArea']= num_col['MasVnrArea'].clip_upper(num_col['MasVnrArea'].quantile(0.99))

num_col['BsmtFinSF1']= num_col['BsmtFinSF1'].clip_upper(num_col['BsmtFinSF1'].quantile(0.99))

num_col['BsmtFinSF2']= num_col['BsmtFinSF2'].clip_upper(num_col['BsmtFinSF2'].quantile(0.99))

num_col['BsmtUnfSF']= num_col['BsmtUnfSF'].clip_upper(num_col['BsmtUnfSF'].quantile(0.99))

num_col['TotalBsmtSF']= num_col['TotalBsmtSF'].clip_upper(num_col['TotalBsmtSF'].quantile(0.99))

num_col['1stFlrSF']= num_col['1stFlrSF'].clip_upper(num_col['1stFlrSF'].quantile(0.99))

num_col['2ndFlrSF']= num_col['2ndFlrSF'].clip_upper(num_col['2ndFlrSF'].quantile(0.99))

num_col['LowQualFinSF']= num_col['LowQualFinSF'].clip_upper(num_col['LowQualFinSF'].quantile(0.99))

num_col['GrLivArea']= num_col['GrLivArea'].clip_upper(num_col['GrLivArea'].quantile(0.99))

num_col['FullBath']= num_col['FullBath'].clip_upper(num_col['FullBath'].quantile(0.99))

num_col['BedroomAbvGr']= num_col['BedroomAbvGr'].clip_upper(num_col['BedroomAbvGr'].quantile(0.99))

num_col['TotRmsAbvGrd']= num_col['TotRmsAbvGrd'].clip_upper(num_col['TotRmsAbvGrd'].quantile(0.99))

num_col['Fireplaces']= num_col['Fireplaces'].clip_upper(num_col['Fireplaces'].quantile(0.99))

num_col['GarageYrBlt']= num_col['GarageYrBlt'].clip_upper(num_col['GarageYrBlt'].quantile(0.99))

num_col['GarageCars']= num_col['GarageCars'].clip_upper(num_col['GarageCars'].quantile(0.99))

num_col['GarageArea']= num_col['GarageArea'].clip_upper(num_col['GarageArea'].quantile(0.99))

num_col['WoodDeckSF']= num_col['WoodDeckSF'].clip_upper(num_col['WoodDeckSF'].quantile(0.99))

num_col['OpenPorchSF']= num_col['OpenPorchSF'].clip_upper(num_col['OpenPorchSF'].quantile(0.99))

num_col['EnclosedPorch']= num_col['EnclosedPorch'].clip_upper(num_col['EnclosedPorch'].quantile(0.99))

num_col['3SsnPorch']= num_col['3SsnPorch'].clip_upper(num_col['3SsnPorch'].quantile(0.99))

num_col['ScreenPorch']= num_col['ScreenPorch'].clip_upper(num_col['ScreenPorch'].quantile(0.99))

num_col['PoolArea']= num_col['PoolArea'].clip_upper(num_col['PoolArea'].quantile(0.99))

num_col['MiscVal']= num_col['MiscVal'].clip_upper(num_col['MiscVal'].quantile(0.99))
num_nulls
## Filling null values attributes with mean...



num_col['LotFrontage']=num_col['LotFrontage'].fillna(num_col['LotFrontage'].mean())  

num_col['GarageYrBlt']=num_col['GarageYrBlt'].fillna(num_col['GarageYrBlt'].mean())

num_col['MasVnrArea']=num_col['MasVnrArea'].fillna(num_col['MasVnrArea'].mean())

num_col['BsmtHalfBath']=num_col['BsmtHalfBath'].fillna(num_col['BsmtHalfBath'].mean())

num_col['BsmtFullBath']=num_col['BsmtFullBath'].fillna(num_col['BsmtFullBath'].mean())

num_col['GarageArea']=num_col['GarageArea'].fillna(num_col['GarageArea'].mean())

num_col['BsmtFinSF1']=num_col['BsmtFinSF1'].fillna(num_col['BsmtFinSF1'].mean())

num_col['BsmtFinSF2']=num_col['BsmtFinSF2'].fillna(num_col['BsmtFinSF2'].mean())

num_col['BsmtUnfSF']=num_col['BsmtUnfSF'].fillna(num_col['BsmtUnfSF'].mean())

num_col['TotalBsmtSF']=num_col['TotalBsmtSF'].fillna(num_col['TotalBsmtSF'].mean())

num_col['GarageCars']=num_col['GarageCars'].fillna(num_col['GarageCars'].mean())
num_col.isnull().sum().sum()
num_col.columns
dfcat_cols = df.drop(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',

       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',

       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',

       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',

       'MoSold', 'YrSold'],axis = 1)
dfcat_cols.info()
dfcat_cols.isnull().sum().sort_values(ascending = False)
dfcat_cols.describe().T ##here "T" is transpose
dfcat_cols["FireplaceQu"] = dfcat_cols["FireplaceQu"].fillna('Gd')

dfcat_cols["GarageCond"] = dfcat_cols["GarageCond"].fillna('TA')

dfcat_cols["GarageQual"] = dfcat_cols["GarageQual"].fillna('TA')

dfcat_cols["GarageFinish"] = dfcat_cols["GarageFinish"].fillna('Unf')

dfcat_cols["GarageType"] = dfcat_cols["GarageType"].fillna('Attchd')

dfcat_cols["BsmtCond"] = dfcat_cols["BsmtCond"].fillna('TA')

dfcat_cols["BsmtExposure"] = dfcat_cols["BsmtExposure"].fillna('No')

dfcat_cols["BsmtQual"] = dfcat_cols["BsmtQual"].fillna('TA')

dfcat_cols["BsmtFinType2"] = dfcat_cols["BsmtFinType2"].fillna('Unf')

dfcat_cols["BsmtFinType1"] = dfcat_cols["BsmtFinType1"].fillna('Unf')

dfcat_cols["MasVnrType"] = dfcat_cols["MasVnrType"].fillna('None')

dfcat_cols["MSZoning"] = dfcat_cols["MSZoning"].fillna('RL')

dfcat_cols["Utilities"] = dfcat_cols["Utilities"].fillna('AllPub')

dfcat_cols["Functional"] = dfcat_cols["Functional"].fillna('Typ')

dfcat_cols["Electrical"] = dfcat_cols["Electrical"].fillna('SBrkr')

dfcat_cols["KitchenQual"] = dfcat_cols["KitchenQual"].fillna('TA')

dfcat_cols["SaleType"] = dfcat_cols["SaleType"].fillna('WD')

dfcat_cols["Exterior2nd"] = dfcat_cols["Exterior2nd"].fillna('VinylSd')

dfcat_cols["Exterior1st"] = dfcat_cols["Exterior1st"].fillna('VinylSd')
dfcat_cols.isnull().sum().any()
df1 = pd.concat([num_col,dfcat_cols],axis = 1)
# check for null values

df1.isnull().sum().sum()
df1.head()
df1.info()
tt = df1[0:1460:]

test = df1[1461::]
tt.isnull().sum().sum()
test.isnull().sum().sum()
target = T_train.drop(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',

       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',

       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',

       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',

       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',

       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',

       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',

       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',

       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',

       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',

       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',

       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',

       'SaleCondition'],axis =1)
train = pd.concat([tt,target],axis =1)
train.isnull().sum().sum()
train.info()
test.info()
numaric_cols =train._get_numeric_data()

numaric_cols.columns
corr = numaric_cols.corr()['SalePrice']

corr[np.argsort(corr,axis=0)].sort_values(ascending=False)
num_corr=numaric_cols.corr()

plt.subplots(figsize=(13,10))

sns.heatmap(num_corr,square = True)
pp = numaric_cols.drop(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond',

          'BsmtFinSF1', 'BsmtFinSF2',

       'BsmtUnfSF', '2ndFlrSF', 'LowQualFinSF',

        'BsmtFullBath', 'BsmtHalfBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr',

        'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',

       'MoSold', 'YrSold'],axis =1)

cm =pp.corr()

sns.set(font_scale=1.35)

f, ax = plt.subplots(figsize=(10,10))

hm=sns.heatmap(cm, annot = True, vmax =.8)
pp.columns
nc = pp.rename(columns ={'1stFlrSF':'FirstFlrSF'})

#we rename this because
nc.columns
## Using statsmodel.formula.api we'll find the correlation



import statsmodels.formula.api as smf
lm=smf.ols('SalePrice~OverallQual+YearBuilt+YearRemodAdd+MasVnrArea+TotalBsmtSF+FirstFlrSF+GrLivArea+FullBath+TotRmsAbvGrd+Fireplaces+GarageYrBlt+GarageCars+GarageArea',nc).fit()
lm.summary()
lm.pvalues
nc['intercept'] = lm.params[0]
np.linalg.inv(nc[['OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'TotalBsmtSF',

       'FirstFlrSF', 'GrLivArea', 'FullBath', 'Fireplaces',

       'GarageArea']].corr().as_matrix())
#should be less than 5 only then we consider those attributes:

np.diag(np.linalg.inv(nc[['OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'TotalBsmtSF',

       'FirstFlrSF', 'GrLivArea', 'FullBath','Fireplaces', 'TotRmsAbvGrd', 'GarageArea']].corr().as_matrix()), 0)
#final numarical columns:

finalnum_cols = nc.drop([ "GarageCars", "GarageYrBlt", "TotRmsAbvGrd"],axis =1)

finalnum_cols.columns
cc = train.drop(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',

       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',

       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',

       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',

       'MoSold', 'YrSold', 'SalePrice'],axis = 1)
cc.columns
cc.isnull().sum().any()
# we should add SalePrice as categorical attributes does not have target(SalePrice)

categorical_col =pd.concat([cc,nc.SalePrice],axis=1)

categorical_col.columns
##Now we need to do stats model.api:



import statsmodels.api as sm

import statsmodels.formula.api as smf
lm1 = smf.ols('SalePrice ~MSZoning+Street+LotShape+LandContour+Utilities+LotConfig+LandSlope+Neighborhood+Condition1+Condition2+BldgType+HouseStyle+RoofStyle+RoofMatl+Exterior1st+Exterior2nd+MasVnrType+ExterQual+ExterCond+Foundation+BsmtQual+BsmtCond+BsmtExposure+BsmtFinType1+BsmtFinType2+Heating+HeatingQC+CentralAir+Electrical+KitchenQual+Functional+FireplaceQu+GarageType+GarageFinish+GarageQual+GarageCond+PavedDrive+SaleType+SaleCondition', categorical_col).fit()
lm1.summary()
import scipy.stats as stats
categorical_col.LotShape.value_counts()
s1 = categorical_col.SalePrice[categorical_col.LotShape=="Reg"]

s2 = categorical_col.SalePrice[categorical_col.LotShape=="IR1"]

s3 = categorical_col.SalePrice[categorical_col.LotShape=="IR2"]

s4 = categorical_col.SalePrice[categorical_col.LotShape=="IR3"]
stats.f_oneway(s1, s2, s3, s4)
categorical_col.LotConfig.value_counts()
s1 = categorical_col.SalePrice[categorical_col.LotConfig=="Inside"]

s2 = categorical_col.SalePrice[categorical_col.LotConfig=="Corner"]

s3 = categorical_col.SalePrice[categorical_col.LotConfig=="CulDSac"]

s4 = categorical_col.SalePrice[categorical_col.LotConfig=="FR2"]

s5 = categorical_col.SalePrice[categorical_col.LotConfig=="FR3"]
stats.f_oneway(s1, s2, s3, s4, s5)
categorical_col.BldgType.value_counts()
s1 = categorical_col.SalePrice[categorical_col.BldgType=="1Fam"]

s2 = categorical_col.SalePrice[categorical_col.BldgType=="TwnhsE"]

s3 = categorical_col.SalePrice[categorical_col.BldgType=="Duplex"]

s4 = categorical_col.SalePrice[categorical_col.BldgType=="Twnhs"]

s5 = categorical_col.SalePrice[categorical_col.BldgType=="2fmCon"]
stats.f_oneway(s1, s2, s3, s4, s5)
categorical_col.HouseStyle.value_counts()
s1 = categorical_col.SalePrice[categorical_col.HouseStyle=="1Story"]

s2 = categorical_col.SalePrice[categorical_col.HouseStyle=="2Story"]

s3 = categorical_col.SalePrice[categorical_col.HouseStyle=="1.5Fin"]

s4 = categorical_col.SalePrice[categorical_col.HouseStyle=="SLvl"]

s5 = categorical_col.SalePrice[categorical_col.HouseStyle=="SFoyer"]

s6 = categorical_col.SalePrice[categorical_col.HouseStyle=="1.5Unf"]

s7 = categorical_col.SalePrice[categorical_col.HouseStyle=="2.5Unf"]

s8 = categorical_col.SalePrice[categorical_col.HouseStyle=="2.5Fin"]
stats.f_oneway(s1, s2, s3, s4, s5, s6, s7, s8)
categorical_col.RoofStyle.value_counts()
s1 = categorical_col.SalePrice[categorical_col.RoofStyle=="Gable"]

s2 = categorical_col.SalePrice[categorical_col.RoofStyle=="Hip"]

s3 = categorical_col.SalePrice[categorical_col.RoofStyle=="Flat"]

s4 = categorical_col.SalePrice[categorical_col.RoofStyle=="Gambrel"]

s5 = categorical_col.SalePrice[categorical_col.RoofStyle=="Mansard"]

s6 = categorical_col.SalePrice[categorical_col.RoofStyle=="Shed"]
stats.f_oneway(s1, s2, s3, s4, s5, s6)
categorical_col.RoofMatl.value_counts()
s1 = categorical_col.SalePrice[categorical_col.RoofMatl=="CompShg"]

s2 = categorical_col.SalePrice[categorical_col.RoofMatl=="Tar&Grv"]

s3 = categorical_col.SalePrice[categorical_col.RoofMatl=="WdShngl"]

s4 = categorical_col.SalePrice[categorical_col.RoofMatl=="WdShake"]

s5 = categorical_col.SalePrice[categorical_col.RoofMatl=="Membran"]

s6 = categorical_col.SalePrice[categorical_col.RoofMatl=="Metal"]

s7 = categorical_col.SalePrice[categorical_col.RoofMatl=="ClyTile"]

s8 = categorical_col.SalePrice[categorical_col.RoofMatl=="Roll"]
stats.f_oneway(s1, s2, s3, s4, s5)
categorical_col.BsmtExposure.value_counts()
s1 = categorical_col.SalePrice[categorical_col.BsmtExposure=="No"]

s2 = categorical_col.SalePrice[categorical_col.BsmtExposure=="Av"]

s3 = categorical_col.SalePrice[categorical_col.BsmtExposure=="Gd"]

s4 = categorical_col.SalePrice[categorical_col.BsmtExposure=="Mn"]
stats.f_oneway(s1, s2, s3, s4)
categorical_col.GarageFinish.value_counts()
s1 = categorical_col.SalePrice[categorical_col.GarageFinish=="Unf"]

s2 = categorical_col.SalePrice[categorical_col.GarageFinish=="RFn"]

s3 = categorical_col.SalePrice[categorical_col.GarageFinish=="Fin"]
stats.f_oneway(s1, s2, s3)
categorical_col.GarageQual.value_counts()
s1 = categorical_col.SalePrice[categorical_col.GarageQual=="TA"]

s2 = categorical_col.SalePrice[categorical_col.GarageQual=="Fa"]

s3 = categorical_col.SalePrice[categorical_col.GarageQual=="Gd"]

s4 = categorical_col.SalePrice[categorical_col.GarageQual=="Ex"]

s5 = categorical_col.SalePrice[categorical_col.GarageQual=="Po"]
stats.f_oneway(s1, s2, s3, s4, s5)
categorical_col.SaleCondition.value_counts()
s1 = categorical_col.SalePrice[categorical_col.SaleCondition=="Normal"]

s2 = categorical_col.SalePrice[categorical_col.SaleCondition=="Partial"]

s3 = categorical_col.SalePrice[categorical_col.SaleCondition=="Abnorml"]

s4 = categorical_col.SalePrice[categorical_col.SaleCondition=="Family"]

s5 = categorical_col.SalePrice[categorical_col.SaleCondition=="Alloca"]

s6 = categorical_col.SalePrice[categorical_col.SaleCondition=="AdjLand"]
stats.f_oneway(s1, s2, s3, s4, s5,s6)
categorical_col.columns
finalcategorical_cols = categorical_col.drop([ 'Street', 'LandContour', 'Utilities',

        'LandSlope', 'Condition1', 'Condition2',

          'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterCond', 'Foundation',

        'BsmtCond',  'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'Electrical',

       'Functional', 'GarageType',

        'PavedDrive', 'SaleType', 'SalePrice'],axis =1)

finalcategorical_cols.columns
test.columns
testcategorical_cols = test.drop(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',

       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',

       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',

       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',

       'MoSold', 'YrSold', 'Street', 'LandContour', 'Utilities', 'LandSlope',

       'Condition1', 'Condition2', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'ExterCond', 'Foundation', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'Electrical', 'Functional', 'GarageType', 'PavedDrive',

       'SaleType'],axis = 1)
testcategorical_cols.columns
cat_concat = pd.concat([finalcategorical_cols,testcategorical_cols],axis = 0)
cat_concat.info()
dummies_concat =  pd.get_dummies(cat_concat, columns =['MSZoning', 'LotShape', 'LotConfig', 'Neighborhood', 'BldgType',

       'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 'BsmtQual',

       'BsmtExposure', 'HeatingQC', 'CentralAir', 'KitchenQual', 'FireplaceQu',

       'GarageFinish', 'GarageQual', 'GarageCond', 'SaleCondition'],drop_first =True)
dummies_concat.info()
traincat_cols = dummies_concat[0:1460:]

testcat_cols = dummies_concat[1461::]
traincat_cols.isnull().sum().sum()
final = pd.concat([finalnum_cols,traincat_cols],axis =1)

final.isnull().sum().sum()
Final = final.sample(n = 730, random_state = 123)

Final.head(4)
Final1x = Final.drop(['SalePrice'], axis= 1)

Final1y = Final.SalePrice
Final2 = final.drop(Final.index)

Final2.info()
Final2x = Final2.drop(['SalePrice'], axis= 1)

Final2y = Final2.SalePrice
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(

        Final1x,

        Final1y,

        test_size=0.20,

        random_state=123)
print (len(X_train), len(X_test))
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()

linreg.fit(X_train, Y_train)
X_train, X_test, Y_train, Y_test = train_test_split(

        Final2x,

        Final2y,

        test_size=0.20,

        random_state=123)
y_pred = linreg.predict(X_test)
print(y_pred.mean())
X_test.columns
from sklearn import metrics
metrics.r2_score(Y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(Y_test, y_pred))

rmse