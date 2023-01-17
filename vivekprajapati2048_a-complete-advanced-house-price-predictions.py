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
import  matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

pd.pandas.set_option('display.max_columns', None)

import warnings
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
print('Training set shape:', df_train.shape)

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print('Testing set shape:', df_test.shape)
df_train.head()
def df_characteristics(df):
    print('Shape of the dataset: {}'.format(df.shape))
    
    df_numerical = df.select_dtypes(include = [np.number])
    print('Number of Numerical Features: {}'.format(df_numerical.shape[1]))
    df_categorical = df.select_dtypes(exclude = [np.number])
    print('Number of Categorical Features: {}'.format(df_categorical.shape[1]))
df_characteristics(df_train)
df_characteristics(df_test)
df_numerical = df_train.select_dtypes(include = [np.number])
numerical_features = df_numerical.columns
numerical_features
df_categorical = df_train.select_dtypes(exclude = [np.number])
categorical_features = df_categorical.columns
categorical_features
def check_null(df):
    null_percent = (df.isnull().sum() / len(df)) * 100
    
    try:
        null_percent = (null_percent.drop(null_percent[null_percent == 0].index)).sort_values(ascending=False)
        
    except:
        print('There is No null values in the dataset')
        print('Returning the dataset...')
        return df
    
    return null_percent
train_nan = check_null(df_train)
test_nan = check_null(df_test)

nan = pd.DataFrame({'Train(%)': train_nan, 'Test(%)': test_nan})
nan.sort_values(by='Train(%)', ascending=False)
columns_drop = ['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence']

df_train.drop(columns = columns_drop, axis=1, inplace=True)
df_test.drop(columns = columns_drop, axis=1, inplace=True)
df_train.FireplaceQu.value_counts()
sns.countplot(df_train['FireplaceQu'])
sns.boxplot(data=df_train, x='SalePrice', y='FireplaceQu')
df_train['FireplaceQu'] = df_train['FireplaceQu'].fillna(0)
df_test['FireplaceQu'] = df_test['FireplaceQu'].fillna(0)

sns.boxplot(data=df_train, x='SalePrice', y='FireplaceQu')
sns.distplot(df_train.LotFrontage)
def LotFrontage_Stats(df):
    print('Mean: {}, Medain: {}'.format(df.LotFrontage.mean(), 
                                        df.LotFrontage.median()))
    
LotFrontage_Stats(df_train)  # training set
LotFrontage_Stats(df_test)  # testing set
sns.regplot(data=df_train, x='SalePrice', y='LotFrontage')
df_train['LotFrontage'] = df_train['LotFrontage'].fillna(df_train['LotFrontage'].median())
df_test['LotFrontage'] = df_test['LotFrontage'].fillna(df_test['LotFrontage'].median())
df_train['GarageQual'].value_counts()
sns.boxplot(data=df_train, x='SalePrice', y='GarageQual')
sns.distplot(df_train[df_train['GarageQual'] == 'TA'].SalePrice)
sns.distplot(df_train[df_train['GarageQual'] == 'Fa'].SalePrice)
df_train.drop('GarageQual', axis=1, inplace=True)
df_test.drop('GarageQual', axis=1, inplace=True)
df_train.GarageFinish.value_counts()
sns.boxplot(data=df_train, x='SalePrice', y='GarageFinish')
df_train['GarageFinish'] = df_train['GarageFinish'].fillna('Nog')
df_test['GarageFinish'] = df_test['GarageFinish'].fillna('Nog')
sns.boxplot(data=df_train, x='SalePrice', y='GarageFinish')
df_train.GarageCond.value_counts()
df_train['GarageCond'] = df_train['GarageCond'].fillna('Nog')

sns.boxplot(data=df_train, x='SalePrice', y='GarageCond')
df_train.drop('GarageCond', axis=1, inplace=True)
df_test.drop('GarageCond', axis=1, inplace=True)
sns.distplot(df_train.GarageYrBlt)
print('Maximum value: {}'.format(df_train.GarageYrBlt.max()))
print('Minimun value: {}'.format(df_train.GarageYrBlt.min()))
sns.regplot(data=df_train, x='SalePrice', y='GarageYrBlt')
df_train['GarageYrBlt'] = df_train['GarageYrBlt'].fillna(df_train.GarageYrBlt.min())
df_test['GarageYrBlt'] = df_test['GarageYrBlt'].fillna(df_test.GarageYrBlt.min())
df_train.GarageType.value_counts()
sns.boxplot(data=df_train, x='SalePrice', y='GarageType')
df_train['GarageType'] = df_train['GarageType'].fillna('Nog')
df_test['GarageType'] = df_test['GarageType'].fillna('Nog')

sns.boxplot(data=df_train, x='SalePrice', y='GarageType')
df_train.BsmtQual.value_counts()
sns.boxplot(data=df_train, x='SalePrice', y='BsmtQual')
df_train['BsmtQual'] = df_train['BsmtQual'].fillna('NoBsmt')
df_test['BsmtQual'] = df_test['BsmtQual'].fillna('NoBsmt')

sns.boxplot(data=df_train, x='SalePrice', y='BsmtQual')
df_train.BsmtCond.value_counts()
sns.boxplot(data=df_train, x='SalePrice', y='BsmtCond')
df_train.drop('BsmtCond', axis=1, inplace=True)
df_test.drop('BsmtCond', axis=1, inplace=True)
df_train.BsmtExposure.value_counts()
sns.boxplot(data=df_train, x='SalePrice', y='BsmtExposure')
df_train['BsmtExposure'] = df_train['BsmtExposure'].fillna('NoBsmt')
df_test['BsmtExposure'] = df_test['BsmtExposure'].fillna('NoBsmt')

sns.boxplot(data=df_train, x='SalePrice', y='BsmtExposure')
df_train.BsmtFinType1.value_counts()
sns.boxplot(data=df_train, x='SalePrice', y='BsmtFinType1')
df_train['BsmtFinType1'] = df_train['BsmtFinType1'].fillna('NoBsmt')
df_test['BsmtFinType1'] = df_test['BsmtFinType1'].fillna('NoBsmt')

sns.boxplot(data=df_train, x='SalePrice', y='BsmtFinType1')
df_train.BsmtFinType2.value_counts()
sns.boxplot(data=df_train, x='SalePrice', y='BsmtFinType2')
df_train.drop('BsmtFinType2', axis=1, inplace=True)
df_test.drop('BsmtFinType2', axis=1, inplace=True)
df_train.MasVnrType.value_counts()
sns.boxplot(data=df_train, x='SalePrice', y='MasVnrType')
df_train['MasVnrType'] = df_train['MasVnrType'].fillna('None')
df_test['MasVnrType'] = df_test['MasVnrType'].fillna('None')

sns.boxplot(data=df_train, x='SalePrice', y='MasVnrType')
sns.distplot(df_train.MasVnrArea)
print('Maximum value: {}'.format(df_train.MasVnrArea.max()))
print('Minimun value: {}'.format(df_train.MasVnrArea.min()))
df_train['MasVnrArea'] = df_train['MasVnrArea'].fillna(df_train.MasVnrArea.min())
df_test['MasVnrArea'] = df_test['MasVnrArea'].fillna(df_test.MasVnrArea.min())
df_train.Electrical.value_counts()
sns.boxplot(data=df_train, x='SalePrice', y='Electrical')
df_train['Electrical'] = df_train['Electrical'].fillna('SBrkr')

sns.boxplot(data=df_train, x='SalePrice', y='Electrical')
check_null(df_train)
df_test.MSZoning.value_counts()
sns.boxplot(data=df_train, x='SalePrice', y='MSZoning')
df_test['MSZoning'] = df_test['MSZoning'].fillna('RL')
df_test.Functional.value_counts() 
df_train.Functional.value_counts()
df_test['Functional'] = df_test['Functional'].fillna('Typ')
df_test.BsmtFullBath.value_counts()
df_train.BsmtFullBath.value_counts()
df_test['BsmtFullBath'] = df_test['BsmtFullBath'].fillna('0')
df_test.BsmtHalfBath.value_counts()
df_train.BsmtHalfBath.value_counts()
df_test['BsmtHalfBath'] = df_test['BsmtHalfBath'].fillna('0')
df_test.Utilities.value_counts()
df_train.Utilities.value_counts()
df_train.drop('Utilities', axis=1, inplace=True)
df_test.drop('Utilities', axis=1, inplace=True)
df_test.SaleType.value_counts()
df_train.SaleType.value_counts()
sns.boxplot(data=df_train, x='SalePrice', y='SaleType')
df_test['SaleType'] = df_test['SaleType'].fillna('WD')
sns.distplot(df_test.GarageArea)
df_test['GarageArea'] = df_test['GarageArea'].fillna(df_test.GarageArea.min())
sns.distplot(df_test.GarageCars)
df_test['GarageCars'] = df_test['GarageCars'].fillna(df_test.GarageCars.min())
df_test.KitchenQual.value_counts()
df_test['KitchenQual'] = df_test['KitchenQual'].fillna('TA')
sns.distplot(df_test.TotalBsmtSF)
df_test['TotalBsmtSF'] = df_test['TotalBsmtSF'].fillna(df_test.TotalBsmtSF.min())
df_test['BsmtUnfSF'] = df_test['BsmtUnfSF'].fillna(df_test.BsmtUnfSF.min())
df_test['BsmtFinSF1'] = df_test['BsmtFinSF1'].fillna(df_test.BsmtFinSF1.min())
df_test['BsmtFinSF2'] = df_test['BsmtFinSF2'].fillna(df_test.BsmtFinSF2.min())
df_test.Exterior1st.value_counts()
df_test['Exterior1st'] = df_test['Exterior1st'].fillna('VinylSd')
df_test.Exterior2nd.value_counts()
df_test['Exterior2nd'] = df_test['Exterior2nd'].fillna('VinylSd')
check_null(df_train)
check_null(df_test)
year_features = [feature for feature in df_numerical
                 if 'Yr' in feature or 'Year' in feature]
year_features
for feature in year_features:
    print('\n', feature, '\n', df_train[feature].unique())
df_train.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title('House Prive vs YearSold')
from sklearn.preprocessing import OneHotEncoder
def concat_df(train, test):
    return pd.concat((train, test), sort=True).reset_index(drop=True)
df_all = concat_df(df_train, df_test)
df_all = df_all.drop(['SalePrice'], axis=1)
df_cat = df_all.select_dtypes(exclude = [np.number])
df_cat_dummies = pd.get_dummies(df_cat)

df_cat_dummies.head()
print(df_all.shape)
print(df_cat.shape)
print(df_cat_dummies.shape)
df_all_features = df_all.join(df_cat_dummies)  # combined all features
df_all_features = df_all_features.drop(df_cat, axis=1)  # dropped original categorical features
df_all_features.head()
df_all_features.shape
def divide_df(df):
    return df.iloc[:1460], df.iloc[1460:]
X_train, X_test = divide_df(df_all_features)
y_train = df_train['SalePrice']
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score,mean_squared_error,make_scorer
from sklearn.ensemble import RandomForestRegressor
rf = make_pipeline(StandardScaler(), 
                   RandomForestRegressor(max_samples=1460, 
                                         n_estimators=5000, 
                                         min_samples_leaf=1, 
                                         random_state=14))
rf.fit(X_train, y_train)
check_null(X_train)
check_null(X_test)
pred = rf.predict(X_test)
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
submission = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': pred })
submission.to_csv('Submission.csv', index=False)
submission.head()