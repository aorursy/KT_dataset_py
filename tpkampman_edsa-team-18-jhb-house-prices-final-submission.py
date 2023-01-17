import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import skew

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()
print(train_df.info())

print(test_df.info())
ids_test = test_df['Id']
train_df['SalePrice'].describe()
f, ax = plt.subplots(figsize=(12, 9))

sns.distplot(train_df['SalePrice']).set_title('Distribution of SalePrice')
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])
f, ax = plt.subplots(figsize=(12, 9))

sns.distplot(train_df['SalePrice']).set_title('Distribution of log[SalePrice]')
corrmatrix = train_df.corr() #Create correlation matrix

f, ax = plt.subplots(figsize=(14, 10))

sns.heatmap(corrmatrix, vmax=.8, square=True, cmap='BuPu')
pd.DataFrame(corrmatrix['SalePrice'].abs().sort_values(ascending=False))
#GrLivArea - high correlation

with sns.axes_style('white'):

    sns.jointplot(x=train_df['GrLivArea'],y=train_df['SalePrice'], color='firebrick')
#GarageArea - high correlation

with sns.axes_style('white'):

    sns.jointplot(y=train_df['SalePrice'], x=train_df['GarageArea'], color='firebrick')
#TotalBsmtSF - high correlation

with sns.axes_style('white'):

    sns.jointplot(y=train_df['SalePrice'], x=train_df['TotalBsmtSF'], color='firebrick')
#PoolArea - low correlation

with sns.axes_style('white'):

    sns.jointplot(x=train_df['SalePrice'], y=train_df['PoolArea'], color='teal')
highcorr = pd.DataFrame(corrmatrix.abs().unstack().transpose().sort_values(ascending=False).drop_duplicates())

highcorr.head(15)
#Neighborhood - high correlation

ax=sns.catplot(x='Neighborhood', y='SalePrice', kind='boxen',data=train_df.sort_values('Neighborhood'),height=12,aspect=2)

ax.set_xticklabels(size=15,rotation=30)

ax.set_yticklabels(size=15,rotation=30)

plt.xlabel('Neighborhood',size=25)

plt.ylabel('SalePrice',size=25)

plt.show()
#ExterQual - high correlation

ax=sns.catplot(x='ExterQual', y='SalePrice',kind='boxen',data=train_df.sort_values('BldgType'),height=12,aspect=2)

ax.set_xticklabels(size=15,rotation=30)

ax.set_yticklabels(size=15,rotation=30)

plt.xlabel('ExterQual',size=25)

plt.ylabel('SalePrice',size=25)

plt.show()
#BsmtQual - high correlation

ax=sns.catplot(x='BsmtQual', y='SalePrice', kind='boxen',data=train_df.sort_values('BsmtQual'),height=12,aspect=2)

ax.set_xticklabels(size=15,rotation=30)

ax.set_yticklabels(size=15,rotation=30)

plt.xlabel('BsmtQual',size=25)

plt.ylabel('SalePrice',size=25)
#Heating - low correlation

ax=sns.catplot(x='Heating', y='SalePrice', kind='boxen',data=train_df.sort_values('Heating'),height=12,aspect=2)

ax.set_xticklabels(size=15,rotation=30)

ax.set_yticklabels(size=15,rotation=30)

plt.xlabel('Heating',size=25)

plt.ylabel('SalePrice',size=25)
#GrLivArea vs SalePrice - before

with sns.axes_style('white'):

    sns.jointplot(x=train_df['GrLivArea'],y=train_df['SalePrice'], color='firebrick')
train_df = train_df[train_df['GrLivArea'] < 4000]
#GrLivArea vs SalePrice - after

with sns.axes_style('white'):

    sns.jointplot(x=train_df['GrLivArea'],y=train_df['SalePrice'], color='firebrick')
df = pd.concat([train_df, test_df])
percent_null = pd.DataFrame((df.isnull().sum()/df.isnull().count()).sort_values(ascending=False))

percent_null.head(20)
to_drop = ['Id', 'PoolQC', 'MiscVal', 'MiscFeature', 'Alley', 'LandContour', 'Utilities', 'FireplaceQu', 'GarageCond', 'Fence']

df.drop(to_drop, axis=1, inplace=True)
#Categorical features where null indicates the feature is NOT present - will be replaced with the str 'None'

cat_none = ['MasVnrType', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'GarageType', 'GarageFinish']

for col in cat_none:

    df[col].fillna('None', inplace=True)

    

#Categorical features which must be replaced by mode (nulls don't indicate feature is absent)

cat_mode = ['MSZoning', 'Exterior1st', 'Exterior2nd', 'Electrical', 'KitchenQual', 'SaleType', 'Functional', 'GarageQual']

for col in cat_mode:

    df[col].fillna(df[col].mode()[0], inplace=True)

    

#Numerical features where null indicates feature is not present - will be replaced by 0

num_zero = ['MasVnrArea', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'TotalBsmtSF', 'BsmtUnfSF', 'GarageYrBlt', 'GarageArea', 'GarageCars']

for col in num_zero:

    df[col].fillna(0, inplace=True)

    

#Continuous numerical feature - nulls to be replaced by median (Grouped by neighborhood as the LotFrontage is expected to be similar in a particular neighborhood)

df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

    
#Convert discrete numerical features into categorical

numerical_to_cat=['BedroomAbvGr', 'Fireplaces', 'FullBath', 'GarageCars', 'HalfBath', 'KitchenAbvGr', 'OverallQual']



df[numerical_to_cat] = df[numerical_to_cat].apply(lambda x: x.astype("str"))
#Correct feature skewness - only log those features which show a decrease in skewness after the transformation

numeric_features = df.dtypes[df.dtypes != 'object'].index.drop('SalePrice')

skew_b = df[numeric_features].apply(lambda x: skew(x.dropna()))

log = np.log1p(df[numeric_features])

skew_a = log[numeric_features].apply(lambda x: skew(x.dropna()))

skew_diff = (abs(skew_b)-abs(skew_a)).sort_values(ascending=False)

df[skew_diff[skew_diff > 0].index] = np.log1p(df[skew_diff[skew_diff > 0].index])
#Encode categorical variables

df = pd.get_dummies(df, drop_first=True)
X_train = df.iloc[:1456].drop('SalePrice', axis=1)

X_test = df.iloc[1456:].drop('SalePrice', axis=1)

y_train = df['SalePrice'].dropna().values
print(X_train.info())

print(X_test.info())
#Define function to extract best train RMSE from GridSearchCV

def best_rmse(grid):

    

    best_score = np.sqrt(-grid.best_score_)

    print(best_score)    

    print(grid.best_estimator_)

    

    return best_score
#Linear Regression



from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression



lm = LinearRegression()

parameters_lm = {'fit_intercept':[True,False]}

grid_lm = GridSearchCV(lm, parameters_lm, cv=5, verbose=1 , scoring ='neg_mean_squared_error')

grid_lm.fit(X_train, y_train)

score_lm = best_rmse(grid_lm)
#Ridge



from sklearn.linear_model import Ridge



ridge = Ridge()

parameters_ridge = {'alpha': [4, 4.1, 4.2, 4.3, 4.4, 4.5], 'tol': [0.001, 0.01, 0.1]}

grid_ridge = GridSearchCV(ridge, parameters_ridge, cv=5, verbose=1, scoring='neg_mean_squared_error')

grid_ridge.fit(X_train, y_train)

score_ridge = best_rmse(grid_ridge)
#Lasso



from sklearn.linear_model import Lasso



lasso = Lasso()

parameters_lasso = {'alpha': [1e-4, 0.001, 0.01, 0.1, 0.5, 1], 'tol':[1e-06, 1e-05, 1e-04, 1e-03, 0.01, 0.1]}

grid_lasso = GridSearchCV(lasso, parameters_lasso, cv=5, verbose=1, scoring='neg_mean_squared_error')

grid_lasso.fit(X_train, y_train)

score_lasso = best_rmse(grid_lasso)
#RandomForest



from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor()

parameters_rf = {'min_samples_split' : [2, 3, 4], 'n_estimators' : [50, 100, 500]}

grid_rf = GridSearchCV(rf, parameters_rf, cv=5, verbose=1, scoring='neg_mean_squared_error')

grid_rf.fit(X_train, y_train)

score_rf = best_rmse(grid_rf)
rmse_x = ['linear', 'ridge', 'lasso', 'randomforest']

rmse_y = [score_lm, score_ridge, score_lasso, score_rf]

sns.set(style='whitegrid')

sns.barplot(x=rmse_x, y=rmse_y)
pred_lasso = np.expm1(grid_lasso.predict(X_test))

pred_ridge = np.expm1(grid_ridge.predict(X_test))



pred_combined = (pred_lasso + pred_ridge)/2
#Creating dataframe of submission data

submission = pd.DataFrame()

submission['Id'] = ids_test.values

submission['SalePrice'] = pred_combined
#Save the output as a csv

submission.to_csv('submission_final_edsateam18.csv', index=False)