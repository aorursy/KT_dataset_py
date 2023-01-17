import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import warnings



warnings.filterwarnings('ignore')



pd.set_option('display.max_rows', 10)

pd.set_option('display.max_columns', 100)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train.head(10)
print("train: ", len(train), "\n test: ", len(test))
train['SalePrice'].describe()
sns.distplot(train['SalePrice'], color='green')
train.drop(['Id', 'SalePrice'], axis=1).info()
for col in train.columns:

    if train[col].nunique()==1:

        print(col)
categorical_columns = [col for col in train.columns if train[col].dtype.name == 'object']

numerical_columns   = [col for col in train.columns if train[col].dtype.name != 'object']
cols_with_many_nans = []

for col in train.columns:

    if len(train[train[col].isna()]) > 500 or len(train[train[col].isnull()]) > 500:

        cols_with_many_nans.append(col)

cols_with_many_nans
train = train.drop(cols_with_many_nans, axis=1)
for num_col in numerical_columns:

    train[num_col].fillna(train[num_col].mean(), inplace=True)
train.head()
correlations = train.corr().abs()

sns.heatmap(correlations, square=True, cmap='RdYlGn_r', linewidths=0.5)
correlations80 = []

for i in correlations:

    for j in correlations.index[correlations[i] > 0.80]:

        if i != j and j not in correlations80 and i not in correlations80:

            correlations80.append(j)

correlations80
train = train.drop(correlations80, axis=1)
corr_df = train.corr().abs()

corr_df['SalePrice'].sort_values(ascending=False).head(5)
# OverallQual / SalePrice

sns.catplot(x='OverallQual', y='SalePrice', data=train)
# GrLivArea/SalePrice

sns.relplot(x='GrLivArea', y='SalePrice', color='green',data=train)
# GrLivArea/SalePrice

sns.relplot(x='TotalBsmtSF', y='SalePrice', color='orange',data=train)
important_num_cols = ['SalePrice', 'GrLivArea', 'TotalBsmtSF']

for col in important_num_cols:

    train = train[np.abs(train[col]-train[col].mean())<=(3*train[col].std())]

len(train)
sns.distplot(train['SalePrice'], color='green')
sns.distplot(np.log(train["SalePrice"]), color='green')
np.log(train["SalePrice"]).skew()
sns.distplot(train['GrLivArea'], color='orange')

print("Skewness:" , train['GrLivArea'].skew())
train["GrLivArea"] = np.log(train["GrLivArea"])

sns.distplot(train['GrLivArea'], color='orange')

print("Skewness:" , train['GrLivArea'].skew())
train = pd.get_dummies(train.drop(['Id'], axis=1)).reset_index(drop=True)
train.head()
needed_cols = train.drop(['SalePrice'], axis=1).columns

test = pd.get_dummies(test)

test_columns = test.columns



for needed_col in needed_cols:

    if needed_col not in test_columns:

        test[needed_col] = 0



Ids = test['Id']

test = test[needed_cols]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop(["SalePrice"], axis=1), train["SalePrice"], test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()

linear_regressor.fit(X_train, y_train)
y_pred = linear_regressor.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score

print('mean_squared_error: ',mean_squared_error(y_test, y_pred),

     '\nr2_score: ',r2_score(y_test, y_pred))
from xgboost import XGBRegressor



xgb_regressor = XGBRegressor(nthread=-1, learning_rate=0.01, max_depth=5, n_estimators=6000, gamma=0.05, subsample=0.7, objective='reg:linear')

xgb_regressor.fit(train.drop(["SalePrice"], axis=1), train["SalePrice"], verbose=False)
from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(max_depth=10, n_estimators=2000)

rf_regressor.fit(train.drop(["SalePrice"], axis=1), train["SalePrice"])
prediction_xgb = xgb_regressor.predict(test)
prediction_rf = rf_regressor.predict(test.fillna(test.mean()))
submit = pd.DataFrame()

submit['Id'] = Ids

submit['SalePrice'] = pd.Series(prediction_xgb)

submit.to_csv('submission_house_prices_xgb.csv', index=False)
submit = pd.DataFrame()

submit['Id'] = Ids

submit['SalePrice'] = pd.Series(prediction_rf)

submit.to_csv('submission_house_prices_rf.csv', index=False)