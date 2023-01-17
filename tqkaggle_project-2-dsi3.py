# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))
# imports

import numpy as np

import pandas as pd

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
# load data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_train.shape, df_test.shape
df_train.head()
df_test.head()
# check dtypes and nulls

df_train.info()
# check if Id column is a unique identifier and if so set it as index

len(df_train.Id.value_counts()), len(df_test.Id.value_counts())
df_train = df_train.set_index('Id')

df_test = df_test.set_index('Id')
# plot grlivarea vs saleprice, remove orange dots

sns.scatterplot(df_train.GrLivArea, df_train.SalePrice)

sns.scatterplot(df_train.GrLivArea[df_train.GrLivArea > 4000], df_train.SalePrice);
df_train = df_train[df_train.GrLivArea < 4000]

df_train.shape
# add all data together to start eda and feature selection & engineering

data = pd.concat((df_train, df_test), sort=False)

data.shape
# check null values

null_totals = data.drop('SalePrice', axis=1).isnull().sum().sort_values(ascending=False)

null_totals = null_totals[null_totals != 0]

plt.figure(figsize=(15,8))

sns.barplot(null_totals.index, null_totals)

plt.xticks(rotation=90)

plt.gca().set_ylabel('Number of NaN values');
# lets investigate missing values and try to fill them

# will define a function to plot distribution and probplot

def dist_and_prob(x):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,7))

    sns.distplot(x, fit=stats.norm, ax=ax[0])

    stats.probplot(x, plot=ax[1])

    ax[0].legend(['Skew = ' + str(x.skew().round(2)) + ' mean = ' + str(round(x.mean(), 2)) + ' std = ' + str(round(x.std(), 2))],

                 loc='best', handlelength=0, handletextpad=0, fancybox=True)
dist_and_prob(data.GarageCars.dropna())
garage_columns = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageCars']

data[data.GarageCars.isnull()][garage_columns]
# garagecars has one missing entry we'll assume to be 0 because most columns point us to no garage and we'll change type to None

data.loc[2577, 'GarageType'] = 'None'

data.GarageCars.fillna(0, inplace=True)

data.GarageCars.isnull().sum()
# same entry as above

data[data.GarageArea.isnull()][garage_columns]
data.GarageArea.fillna(0, inplace=True)

data.GarageArea.isnull().sum()
# kitchenQual

data[data.KitchenQual.isnull()]
data.KitchenQual.value_counts()
# fill with mode TA which means typical average

data.KitchenQual.fillna(data.KitchenQual.mode()[0], inplace=True)

data.KitchenQual.isnull().sum()
# Electrical

data[data.Electrical.isnull()]
data.Electrical.value_counts()
# fill with mode

data.Electrical.fillna(data.Electrical.mode()[0], inplace=True)

data.Electrical.isnull().sum()
# SaleType

data[data.SaleType.isnull()]
data.SaleType.value_counts()
# fill with mode

data.SaleType.fillna(data.SaleType.mode()[0], inplace=True)

data.SaleType.isnull().sum()
# Exterior1st

data[data.Exterior1st.isnull()]
data.Exterior1st.value_counts()
# fill with mode

data.Exterior1st.fillna(data.Exterior1st.mode()[0], inplace=True)

data.Exterior1st.isnull().sum()
bsmt_columns = data.columns[data.columns.str.contains('Bsmt')]

bsmt_columns
# totalbsmtsf

data[data.TotalBsmtSF.isnull()][bsmt_columns]
# since columns NaN, we'll assume 0 bsmt

data.TotalBsmtSF.fillna(0, inplace=True)

data.TotalBsmtSF.isnull().sum()
# fill other bsmt sf values with 0

data.BsmtUnfSF.fillna(0, inplace=True)

data.BsmtFinSF1.fillna(0, inplace=True)

data.BsmtFinSF2.fillna(0, inplace=True)
# exterior2nd

data[data.Exterior2nd.isnull()]
data.Exterior2nd.value_counts()
# fill with mode

data.Exterior2nd.fillna(data.Exterior2nd.mode()[0], inplace=True)
# Bsmtfullbath, Bsmthalfbath

data[data.BsmtFullBath.isnull()][bsmt_columns]
data.BsmtFullBath.value_counts()
# fill with 0

data.BsmtFullBath.fillna(0, inplace=True)

data.BsmtHalfBath.fillna(0, inplace=True)
# Functional

data[data.Functional.isnull()]
data.Functional.value_counts()
# fill with mode

data.Functional.fillna(data.Functional.mode()[0], inplace=True)
# Utilities

data[data.Utilities.isnull()]
data.Utilities.value_counts()
# since all observations are the same type, will drop the whole column

data.drop('Utilities', axis=1, inplace=True)
# mszoning

data[data.MSZoning.isnull()]['Neighborhood']
data.groupby('Neighborhood').MSZoning.value_counts()
data.MSZoning.value_counts()
# will fill the IDOTRR neighborhood properties with RM because no RL in that neighborhood. RL for the mitchel property

nullzones = data[data.MSZoning.isnull()]['Neighborhood']

indexed = nullzones[nullzones == 'IDOTRR'].index

data.loc[indexed, 'MSZoning'] = data.loc[indexed, 'MSZoning'].fillna('RM')

data.MSZoning.fillna('RL', inplace=True)
# MassVnrArea, MassVnrType

data[data.MasVnrArea.isnull()][['MasVnrArea', 'MasVnrType']]
data.MasVnrType.value_counts()
# fill type with None and area with 0

data.MasVnrType.fillna('None', inplace=True)
data.MasVnrArea.fillna(0, inplace=True)
data.MasVnrType.isnull().sum()
# BsmtFinType1, BsmtFinType2

data[data.BsmtFinType1.isnull()][bsmt_columns]
# fill both with None since no basement

data.BsmtFinType1.fillna('None', inplace=True)

data.BsmtFinType2.fillna('None', inplace=True)
data.BsmtFinType2.isnull().sum()
# BsmtQual, BsmtCond, BsmtExposure

data[data.BsmtQual.isnull()][bsmt_columns]
# fill with None except 2218, 2219 mode

data.loc[[2218,2219],'BsmtQual'] = data.loc[[2218,2219],'BsmtQual'].transform(lambda x: data.BsmtQual.mode()[0])

data.BsmtQual.fillna('None', inplace=True)

data.BsmtQual.value_counts()
null_bsmt_cond = data[data.BsmtCond.isnull()][bsmt_columns]

indexes = null_bsmt_cond[null_bsmt_cond.TotalBsmtSF != 0].index

null_bsmt_cond[null_bsmt_cond.TotalBsmtSF != 0]
# will fill these 3 cases with mode and the rest with None

data.loc[indexes, 'BsmtCond'] = data.loc[indexes, 'BsmtCond'].fillna(data.BsmtCond.mode()[0])

data.BsmtCond.fillna('None', inplace=True)

data.BsmtCond.value_counts()
data[data.BsmtExposure.isnull()][bsmt_columns]
null_bsmt_exp = data[data.BsmtExposure.isnull()][bsmt_columns]

indexes = null_bsmt_exp[null_bsmt_exp.TotalBsmtSF != 0].index

null_bsmt_exp[null_bsmt_exp.TotalBsmtSF != 0]
# fill 3 with mode and rest None

data.loc[indexes, 'BsmtExposure'] = data.loc[indexes, 'BsmtExposure'].fillna(data.BsmtExposure.mode()[0])

data.BsmtExposure.fillna('None', inplace=True)

data.BsmtExposure.value_counts()
# garage columns

data[data.GarageType.isnull()][garage_columns]
# fill with none

data.GarageType.fillna('None', inplace=True)

data.GarageType.value_counts()
null_garage = data[data.GarageFinish.isnull()]

indexes = null_garage[null_garage.GarageType != 'None'].index

null_garage[null_garage.GarageType != 'None'][garage_columns]
# fill with mode for the 2 above

data.loc[indexes, 'GarageFinish'] = data.loc[indexes, 'GarageFinish'].fillna(data.GarageFinish.mode()[0])

data.loc[indexes, 'GarageQual'] = data.loc[indexes, 'GarageQual'].fillna(data.GarageQual.mode()[0])

data.loc[indexes, 'GarageCond'] = data.loc[indexes, 'GarageCond'].fillna(data.GarageCond.mode()[0])

data.GarageFinish.fillna('None', inplace=True)

data.GarageFinish.value_counts()
data[data.GarageQual.isnull()][garage_columns]
data.GarageQual.fillna('None', inplace=True)

data.GarageQual.value_counts()
data[data.GarageCond.isnull()][garage_columns]
data.GarageCond.fillna('None', inplace=True)

data.GarageCond.value_counts()
data[data.GarageYrBlt.isnull()][garage_columns + ['GarageYrBlt', 'YearBuilt', 'Neighborhood']][data[data.GarageYrBlt.isnull()].GarageType != 'None']
# lets check when most similar houses with garages had them built

data[(data.YearBuilt == 1910) & (data.Neighborhood == 'OldTown')].GarageYrBlt.value_counts()
# fill with 1910

data.loc[2127, 'GarageYrBlt'] = 1910
# not sure if I should fill the rest of NaNs with 0 or the yearbuilt house value, lets check distribution

dist_and_prob(data.GarageYrBlt.dropna())
# theres this very weird outlier showing a garage built in the future, lets fix this

data[data.GarageYrBlt > 2018][garage_columns + ["GarageYrBlt" , 'YearBuilt']]
# it seems its a typo, should be 2007

data.loc[2593, 'GarageYrBlt'] = 2007
dist_and_prob(data.GarageYrBlt.dropna())
# since 0 will ruin the distribution I'll just fill in the house yearbuilt value for the rest

indexes = data[data.GarageYrBlt.isnull()].index

house_year = data[data.GarageYrBlt.isnull()]['YearBuilt']

data.loc[indexes, 'GarageYrBlt'] = house_year

data.loc[indexes, ['GarageYrBlt', 'YearBuilt']]
# LotFrontage

lot_columns = list(data.columns[data.columns.str.contains('Lot')])

data[data.LotFrontage.isnull()][lot_columns]
dist_and_prob(data.LotFrontage.dropna())
# since its impossible for a lot to have 0 frontage lets see by neighborhood

data.groupby('Neighborhood').LotFrontage.median()
# fill with neighborhood median

data['LotFrontage'] = data.groupby('Neighborhood').LotFrontage.transform(lambda x: x.fillna(x.median()))
dist_and_prob(data.LotFrontage)
# lets check if the outlier makes sense

data[data.LotFrontage > 220][lot_columns + garage_columns + ['YearBuilt', 'Neighborhood']]
data[(data.Neighborhood == 'NAmes') & (data.LotArea > 17000)]
# Cannot tell from the data so will leave it as is
# FirePlaceQu

data[data.FireplaceQu.isnull()]['Fireplaces'].value_counts()
# fill with none

data.FireplaceQu.fillna('None', inplace=True)

data.FireplaceQu.isnull().sum()
# Fence

data.Fence.isnull().sum()
# per data descrip fill with none

data.Fence.fillna('None', inplace=True)

data.Fence.value_counts()
# Alley

data.Alley.isnull().sum()
# per data descrip fill with none

data.Alley.fillna('None', inplace=True)

data.Alley.value_counts()
# miscfeature

data.MiscFeature.isnull().sum()
# per data descrip fill with none

data.MiscFeature.fillna('None', inplace=True)

data.MiscFeature.value_counts()
# PoolQC

data.PoolQC.isnull().sum()
# per data descrip fill with none

data.PoolQC.fillna('None', inplace=True)

data.PoolQC.value_counts()
data.info()
# change mssubclass to str

data.MSSubClass = data.MSSubClass.astype(str)
def scatter_box(x):

    plt.figure(figsize=(10,8))

    if str(x.dtype) != 'object':

        sns.regplot(x, data.SalePrice)

    else:

        sns.boxplot(x, data.SalePrice)

        plt.show()

        print(x.value_counts())
# Lets go through features and see their scatter/box plot and distribution

for column in data.drop('SalePrice', axis=1).columns:

    scatter_box(data[column])
# will create new column totalbaths

data['TotalBaths'] = (data.HalfBath/2) + (data.BsmtHalfBath/2) + (data.FullBath) + (data.BsmtFullBath)

scatter_box(data.TotalBaths)
# change orderly categories of interest into nums

data.ExterQual = data.ExterQual.map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
data.BsmtQual = data.BsmtQual.map({'None':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
data.BsmtExposure = data.BsmtExposure.map({'None':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4})
data.HeatingQC = data.HeatingQC.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
data.KitchenQual = data.KitchenQual.map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
# make dummies for features of interest

data['FireplaceQu_isNone'] = data.FireplaceQu.transform(lambda x: 1 if x == 'None' else 0)

data.FireplaceQu_isNone.value_counts()
data['PavedDrive_isY'] = data.PavedDrive.transform(lambda x: 1 if x == 'Y' else 0)

data.PavedDrive_isY.value_counts()
data['SaleType_isNew'] = data.SaleType.transform(lambda x: 1 if x == 'New' else 0)

data.SaleType_isNew.value_counts()
data['SaleCondition_isPartial'] = data.SaleCondition.transform(lambda x: 1 if x == 'Partial' else 0)

data.SaleCondition_isPartial.value_counts()
# getting ideas on what features to keep, lets try a heatmap

interest_columns = ['SalePrice', 'LotArea', 'OverallQual', 'YearBuilt', 'MasVnrArea', 'TotalBsmtSF', 'GrLivArea', 'TotRmsAbvGrd',

                    'Fireplaces', 'GarageCars', 'GarageArea', 'TotalBaths', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'HeatingQC',

                    'KitchenQual', 'FireplaceQu_isNone', 'PavedDrive_isY', 'SaleType_isNew', 'SaleCondition_isPartial']

plt.figure(figsize=(16,16))

sns.heatmap(data[interest_columns].corr(), center=0, annot=True, cmap='coolwarm');
# dummify features of interest



def dummify_column(data, column):

    return pd.concat((data, pd.get_dummies(data[column.name], prefix=column.name, prefix_sep='_is')), axis=1)

    

data = dummify_column(data, data.MSSubClass)

data = dummify_column(data, data.MSZoning)

data = dummify_column(data, data.LotShape)

data = dummify_column(data, data.Neighborhood)

data = dummify_column(data, data.HouseStyle)

data = dummify_column(data, data.RoofStyle)

data = dummify_column(data, data.MasVnrType)

data = dummify_column(data, data.Foundation)

data = dummify_column(data, data.GarageType)

data = dummify_column(data, data.CentralAir).drop('CentralAir_isN', axis=1)

data = dummify_column(data, data.GarageFinish).drop('GarageFinish_isNone', axis=1)
data.shape
# keep only numeric features of interest

columns_to_drop = [data[column].name for column in list(data.columns) if data[column].dtype == 'object'] + ['SalePrice', 'LotArea', 'LotFrontage', 'TotRmsAbvGrd', 'Fireplaces',

                                                                                                            'GarageCars', 'SaleType_isNew', 'OverallCond', 'YearRemodAdd',

                                                                                                            'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF',

                                                                                                            'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

                                                                                                           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'GarageYrBlt',

                                                                                                            'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',

                                                                                                           'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
# Get X and Y

X = data.drop(columns_to_drop, axis=1)

y = data.SalePrice



X.shape, y.shape
y_train = y[y.notnull()]

dist_and_prob(y_train)
# lets try to get it to a normal distribution using log1p

y_train = np.log1p(y_train)
dist_and_prob(y_train)
# now lets see distribution of X features

for column in list(X.loc[:,:'FireplaceQu_isNone'].columns):

    dist_and_prob(X[column])
# lets try log1p for GrLivArea

X.GrLivArea = np.log1p(X.GrLivArea)

dist_and_prob(X.GrLivArea)
# split back to train and test

X_train = X[y.notnull()]

X_test = X[y.isnull()]



X_train.shape, X_test.shape, y_train.shape
# sklearn imports

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

from sklearn.metrics import mean_squared_error, mean_squared_log_error

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyRegressor
# scale the data

ss = StandardScaler()

X_train_ss = ss.fit_transform(X_train)

X_test_ss = ss.transform(X_test)
# lets use the dummy regressor to get a baseline RMSE

dummy = DummyRegressor()

dummy.fit(X_train_ss, y_train)

dummy.score(X_train_ss, y_train)
# as expected score is 0 because its the baseline and equal to mean

np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(dummy.predict(X_train_ss))))
np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(dummy.predict(X_train_ss))))
# lets do a linear regression first

lin_reg = LinearRegression()

lin_reg.fit(X_train_ss, y_train);
# lets use crossval in combo with pipeline to get an R2 score

scaler = StandardScaler()

lr = LinearRegression()

pipeline = Pipeline([('transformer', scaler), ('estimator', lr)])

cv = KFold(n_splits=10, shuffle=True, random_state=42)



lr_scores = cross_val_score(pipeline, X_train, y_train, cv=cv);
lr_scores.mean()
# now lets see RMSE

np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(lin_reg.predict(X_train_ss))))
np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(lin_reg.predict(X_train_ss))))
# Ridge

ridge_reg = Ridge(alpha=.5)

ridge_reg.fit(X_train_ss, y_train);
scaler = StandardScaler()

rr = Ridge(alpha=.5)

pipeline = Pipeline([('transformer', scaler), ('estimator', rr)])

cv = KFold(n_splits=10, shuffle=True)



rr_scores = cross_val_score(pipeline, X_train, y_train, cv=cv);
rr_scores.mean()
np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(ridge_reg.predict(X_train_ss))))
np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(ridge_reg.predict(X_train_ss))))
# Lasso

lasso_reg = Lasso(alpha=.00005)

lasso_reg.fit(X_train_ss, y_train);
scaler = StandardScaler()

lass = Lasso(alpha=.00005)

pipeline = Pipeline([('transformer', scaler), ('estimator', lass)])

cv = KFold(n_splits=10, shuffle=True)



lass_scores = cross_val_score(pipeline, X_train, y_train, cv=cv);
lass_scores.mean()
np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(lasso_reg.predict(X_train_ss))))
np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(lasso_reg.predict(X_train_ss))))
# ElasticNet

elastic_reg = ElasticNet(alpha=.0001, l1_ratio=.5)

elastic_reg.fit(X_train_ss, y_train);
scaler = StandardScaler()

en = ElasticNet(alpha=.01, l1_ratio=.3)

pipeline = Pipeline([('transformer', scaler), ('estimator', en)])

cv = KFold(n_splits=10, shuffle=True)



en_scores = cross_val_score(pipeline, X_train, y_train, cv=cv);
en_scores.mean()
np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(elastic_reg.predict(X_train_ss))))
# lets hope gridsearch will give us an improvement

elastic_params = {'alpha':[.00001, .00005, .0001, .0005, .001, .01, .05, .1, 1, 5],

                 'l1_ratio':[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]}



elastic_grid = GridSearchCV(ElasticNet(), elastic_params, n_jobs=-1, cv=KFold(n_splits=10, shuffle=True), verbose=2)



results = elastic_grid.fit(X_train_ss, y_train)
results.best_score_, results.best_params_
np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(results.best_estimator_.predict(X_train_ss))))
# slight improvement, lets try zooming in on the range and doing a new gridsearch

elastic_params = {'alpha':[.005, .006, .007, .008, .009, .01, .015, .025, .035, .04],

                 'l1_ratio':[.25, .275, .285, .295, .3, .315, .325, .335, .35, .365]}



elastic_grid = GridSearchCV(ElasticNet(), elastic_params, n_jobs=-1, cv=KFold(n_splits=10, shuffle=True), verbose=2)



results = elastic_grid.fit(X_train_ss, y_train)
results.best_score_, results.best_params_
np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(results.best_estimator_.predict(X_train_ss))))
np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(results.best_estimator_.predict(X_train_ss))))
# Ridge seems to be giving better results than elastic, lets try a gridsearch with it

ridge_params = {'alpha':[90, 92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 200, 500]}



ridge_grid = GridSearchCV(Ridge(), ridge_params, n_jobs=-1, cv=KFold(n_splits=10, shuffle=True), verbose=2)



results = ridge_grid.fit(X_train_ss, y_train)
results.best_score_, results.best_params_
np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(results.best_estimator_.predict(X_train_ss))))
np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(results.best_estimator_.predict(X_train_ss))))
# it seems we're not getting any better results than these, lets export the csv and get our score on kaggle

best_model = results.best_estimator_

y_test_preds = np.expm1(best_model.predict(X_test_ss))

scoring_df = pd.concat((pd.Series(X_test.index, name='Id'), pd.Series(y_test_preds, name='SalePrice')), axis=1)

scoring_df.head()
scoring_df.to_csv('score.csv', index=False)