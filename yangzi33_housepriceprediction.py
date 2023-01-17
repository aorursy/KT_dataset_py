# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import plotly.express as px

import matplotlib.pyplot as plt



# Filter out warnings

import warnings

warnings.filterwarnings('ignore')



# To style plots

plt.style.use('fivethirtyeight')



from itertools import cycle

color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import Dataset

X_sample = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

X_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

X_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# X_sample

X_train.head()

# X_train.shape

# (1460, 81)
# Configuring plot dimension

plt.figure(figsize=(15, 5))



# Plotting and labelling

plt.plot(X_train.SalePrice, linewidth=1, color=next(color_cycle))

plt.title('Distribution for Sale Prices')

plt.ylabel('Sale Prices')
plt.figure(figsize=(15,5))

plt.plot(X_train.SalePrice.sort_values().reset_index(drop=True), color=next(color_cycle))

plt.title('Distribution for Sorted Sale Prices')

plt.ylabel('Sale Price')
# Visualize locations of null values using the heatmap trick

sns.heatmap(X_train.isnull(), yticklabels=False, cmap='plasma')
# Check out number of null values in the training set

X_train.isnull().sum().sort_values(ascending=False)[0:19]
# Check out number of null values in the testing set

X_test.isnull().sum().sort_values(ascending=False)[0:33]
X_train.LotFrontage.head()
# Number of missing values in training set

X_train.LotFrontage.isnull().sum()
# Number of missing values in testing set

X_test.LotFrontage.isnull().sum()
# Filling null values with mean

X_train['LotFrontage'] = X_train['LotFrontage'].fillna(X_train.LotFrontage.mean())

X_test['LotFrontage'] = X_test['LotFrontage'].fillna(X_test.LotFrontage.mean())
X_train.Alley.value_counts(dropna=False)
X_test.Alley.value_counts(dropna=False)
# Dropping columns

if 'Alley' in X_train:

    X_train.drop(columns=['Alley'], inplace=True)



if 'Alley' in X_test:

    X_test.drop(columns=['Alley'], inplace=True)
# Performing same actions for other columns

col_replace = ['BsmtCond', 'BsmtQual', 'FireplaceQu', 'GarageType', 

               'GarageCond', 'GarageFinish', 'GarageQual', 'MasVnrType', 

               'MasVnrArea', 'BsmtExposure', 'BsmtFinType2']

col_drop = ['PoolQC', 'Fence', 'MiscFeature', 'GarageYrBlt']



for col in col_replace:

    X_train[col] = X_train[col].fillna(X_train[col].mode()[0])

    X_test[col] = X_test[col].fillna(X_test[col].mode()[0])

    

for col in col_drop:

    if col in X_train:

        X_train.drop(columns=col, inplace=True)

    if col in X_test:

        X_test.drop(columns=col, inplace=True)
X_train.isnull().sum().sort_values(ascending=False)
# Handling remaining null values

X_train.dropna(inplace=True)



if 'Id' in X_train:

    X_train.drop(columns=['Id'], inplace=True)
X_train.shape
X_test.isnull().sum().sort_values(ascending=False)[0:17]
X_test['MSZoning'] = X_test['MSZoning'].fillna(X_test['MSZoning'].mode()[0])
columns_1 = ['BsmtFinType1', 'Utilities','BsmtFullBath', 'BsmtHalfBath', 'Functional', 'SaleType', 'Exterior2nd', 

           'Exterior1st', 'KitchenQual']

columns_2 = ['GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',  'TotalBsmtSF', 'GarageArea']



for col in columns_1:

    X_test[col] = X_test[col].fillna(X_test[col].mode()[0])

for col in columns_2:

    X_test[col] = X_test[col].fillna(X_test[col].mean())
if 'Id' in X_test:

    X_test.drop(columns=['Id'], inplace=True)
X_test.shape
X_train.isnull().any().any()
X_test.isnull().any().any()
mszoning_dist = px.scatter(X_train, x=X_train.index, y='SalePrice', labels={'x': 'Index'}, 

                           color=X_train.MSZoning, template="seaborn", 

                           title="Sale Price Distribution based on MSZoning")

mszoning_dist.show()
street_dist = px.scatter(X_train, x=X_train.index, y='SalePrice', labels={'x': 'Index'}, 

                           color=X_train.Street, template="seaborn", 

                           title="Sale Price Distribution based on Street")

street_dist.show()
X_train.LotConfig.unique()
# Configuring plot size for compact view

plt.figure(figsize=(20, 10))



# Plots with respect to each value of LotConfig

plt.subplot(2, 3, 1)

plt.scatter(x=X_train[X_train.LotConfig == 'Inside'].index,

           y=X_train[X_train.LotConfig == 'Inside'].SalePrice,

            color=next(color_cycle))

plt.title('Sale Price Distribution based on Inside Value')



plt.subplot(2, 3, 2)

plt.scatter(x=X_train[X_train.LotConfig == 'FR2'].index,

           y=X_train[X_train.LotConfig == 'FR2'].SalePrice,

            color=next(color_cycle))

plt.title('Sale Price Distribution based on FR2 Value')



plt.subplot(2, 3, 3)

plt.scatter(x=X_train[X_train.LotConfig == 'Corner'].index,

           y=X_train[X_train.LotConfig == 'Corner'].SalePrice,

            color=next(color_cycle))

plt.title('Sale Price Distribution based on Corner Value')



plt.subplot(2, 3, 4)

plt.scatter(x=X_train[X_train.LotConfig == 'CulDSac'].index,

           y=X_train[X_train.LotConfig == 'CulDSac'].SalePrice,

            color=next(color_cycle))

plt.title('Sale Price Distribution based on CulDSac Value')



plt.subplot(2, 3, 5)

plt.scatter(x=X_train[X_train.LotConfig == 'FR3'].index,

           y=X_train[X_train.LotConfig == 'FR3'].SalePrice,

            color=next(color_cycle))

plt.title('Sale Price Distribution based on FR3 Value')

cols = ['MSZoning', 'Street',

       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 

       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',

       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical',

       'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',

       'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
len(cols)
# Clean dataframe

final_df = pd.concat([X_train, X_test], axis=0)
final_df.shape
def HotEncoding(cols):

    df = final_df

    i = 0

    for col in cols:

        dummy = pd.get_dummies(final_df[col], drop_first=True)

        final_df.drop([col], axis=1, inplace=True)

        

        if i >= 1:

            df = pd.concat([df, dummy], axis=1)

        else:

            df = dummy.copy()

        

        i += 1

    

    df = pd.concat([final_df, df], axis=1)

    

    return df
final_df = HotEncoding(cols)
final_df.shape
final_df = final_df.loc[:, ~final_df.columns.duplicated()]
final_df.shape
# Splitting data as the original one

df_train = final_df.iloc[:1422, :]

df_test = final_df.iloc[1422:, :]
df_test.drop(['SalePrice'], axis=1, inplace=True)
X_train_final = df_train.drop(['SalePrice'], axis=1)

y_train_final = df_train['SalePrice']
# Importing required libraries

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
X_std = StandardScaler().fit_transform(X_train_final)



my_columns = X_train_final.columns

new_df = pd.DataFrame(X_std, columns=my_columns)
pca = PCA(n_components=2)

df_pca = pca.fit_transform(new_df)
# Plotting

plt.figure(figsize = (10, 8))

plt.scatter(df_pca[:, 0], df_pca[:, 1], c=y_train_final, cmap='plasma')



# Labeling

plt.xlabel('First Principle Component')

plt.ylabel('Second Principle Component')
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(X_train_final, y_train_final)
# Fitting the Linear Regression model

linreg = LinearRegression()

linreg.fit(X_train_linear, y_train_linear)
# Checking model accuracy

print("R_Squared Value for Training Set: {}".format(linreg.score(X_train_linear, y_train_linear)))

print("R_Squared Value for Testing Set: {}".format(linreg.score(X_test_linear, y_test_linear)))
from sklearn.model_selection import RandomizedSearchCV

import xgboost
xgb = xgboost.XGBRegressor()
# Hyperparameter tuning

n_estimators = [100, 500, 900, 1100, 1500, 2000]

max_depth = [2, 3, 5, 10, 15]

booster = ['gbtree', 'gblinear']

learning_rate = [0.05, 0.1, 0.15, 0.20]

min_child_weight = [1, 2, 3, 4]

base_score = [0.25, 0.5, 0.75, 1]



# Define the grid of hyperparamers for searching purposes

hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth': max_depth,

    'booster': booster,

    'learning_rate': learning_rate,

    'min_child_weight': min_child_weight,

    'base_score': base_score,

}



random_cv = RandomizedSearchCV(estimator=xgb, param_distributions=hyperparameter_grid,

                              cv=5, n_iter=50,

                              scoring='neg_mean_absolute_error', n_jobs=4,

                              verbose=5, return_train_score=True,

                              random_state=42)
# Fitting randomcv model

random_cv.fit(X_train_final, y_train_final)
# Finding best estimator

random_cv.best_estimator_
# Initialize model using best estimators

xgb = xgboost.XGBRegressor(base_score=0.75, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints='',

             learning_rate=0.1, max_delta_step=0, max_depth=3,

             min_child_weight=1, missing=None, monotone_constraints='()',

             n_estimators=900, n_jobs=0, num_parallel_tree=1, random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,

             tree_method='exact', validate_parameters=1, verbosity=None)
# Fitting model

xgb.fit(X_train_final, y_train_final)
y_pred = xgb.predict(df_test)
y_pred
# Finalize data to submit

pred = pd.DataFrame(y_pred)

samp = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

submission = pd.concat([samp['Id'], pred], axis=1)

submission.columns = ['Id', 'SalePrice']
submission
# Submitting

submission.to_csv('submission-1.csv', index=False)