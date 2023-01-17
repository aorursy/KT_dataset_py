import sys

!cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import cupy as cp# linear algebra

import cudf # data processing, CSV file I/O (e.g. pd.read_csv)

import cuml



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = cudf.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_df = cudf.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



df = [train_df, test_df]
for dataset in df:

    dataset.drop(['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
train_df['BsmtQual'].value_counts().index[0]
for dataset in df:

    dataset['LotFrontage'] = dataset['LotFrontage'].fillna(dataset['LotFrontage'].mean())#float

    dataset['BsmtQual'] = dataset['BsmtQual'].fillna(dataset['BsmtQual'].value_counts().index[0])

    dataset['BsmtCond'] = dataset['BsmtCond'].fillna(dataset['BsmtCond'].value_counts().index[0])

    dataset['BsmtExposure'] = dataset['BsmtExposure'].fillna(dataset['BsmtExposure'].value_counts().index[0])

    dataset['BsmtFinType1'] = dataset['BsmtFinType1'].fillna(dataset['BsmtFinType1'].value_counts().index[0])

    dataset['MasVnrType'] = dataset['MasVnrType'].fillna(dataset['MasVnrType'].value_counts().index[0])

    dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].mean())#float

    dataset['BsmtFinType2'] = dataset['BsmtFinType2'].fillna(dataset['BsmtFinType2'].value_counts().index[0])

    dataset['Electrical'] = dataset['Electrical'].fillna(dataset['Electrical'].value_counts().index[0])

    dataset['FireplaceQu'] = dataset['FireplaceQu'].fillna(dataset['FireplaceQu'].value_counts().index[0])

    dataset['GarageType'] = dataset['GarageType'].fillna(dataset['GarageType'].value_counts().index[0])

    dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(dataset['GarageYrBlt'].mean())#float

    dataset['GarageFinish'] = dataset['GarageFinish'].fillna(dataset['GarageFinish'].value_counts().index[0])

    dataset['GarageQual'] = dataset['GarageQual'].fillna(dataset['GarageQual'].value_counts().index[0])

    dataset['GarageCond'] = dataset['GarageCond'].fillna(dataset['GarageCond'].value_counts().index[0])

    #In test case

    dataset['GarageCars'] = dataset['GarageCars'].fillna(dataset['GarageCars'].mean().astype(int))

    dataset['GarageArea'] = dataset['GarageArea'].fillna(dataset['GarageArea'].mean().astype(int))

    dataset['SaleType'] = dataset['SaleType'].fillna(dataset['SaleType'].value_counts().index[0])

    dataset['Functional'] = dataset['Functional'].fillna(dataset['Functional'].value_counts().index[0])

    dataset['KitchenQual'] = dataset['KitchenQual'].fillna(dataset['KitchenQual'].value_counts().index[0])

    dataset['BsmtHalfBath'] = dataset['BsmtHalfBath'].fillna(dataset['BsmtHalfBath'].mean().astype(int))

    dataset['BsmtFullBath'] = dataset['BsmtFullBath'].fillna(dataset['BsmtFullBath'].mean().astype(int))

    dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(dataset['TotalBsmtSF'].mean().astype(int))

    dataset['BsmtUnfSF'] = dataset['BsmtUnfSF'].fillna(dataset['BsmtUnfSF'].mean().astype(int))

    dataset['BsmtFinSF2'] = dataset['BsmtFinSF2'].fillna(dataset['BsmtFinSF2'].mean().astype(int))

    dataset['BsmtFinSF1'] = dataset['BsmtFinSF1'].fillna(dataset['BsmtFinSF1'].mean().astype(int))

    dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna(dataset['Exterior2nd'].value_counts().index[0])

    dataset['Exterior1st'] = dataset['Exterior1st'].fillna(dataset['Exterior1st'].value_counts().index[0])

    dataset['Utilities'] = dataset['Utilities'].fillna(dataset['Utilities'].value_counts().index[0])

    dataset['MSZoning'] = dataset['MSZoning'].fillna(dataset['MSZoning'].value_counts().index[0])

dataframe = cudf.concat([train_df, test_df])
dataframe.shape
dataframe.columns
dataframe['BsmtQual'].dtype == 'O'
dataframe2 = cudf.get_dummies(dataframe)
dataframe2.shape
dataframe2.head()
for column in dataframe2.columns:

    if dataframe2[column].dtype == 'O':

        dataframe2.drop([column], axis=1, inplace=True)

        

dataframe2.shape
train_df = dataframe2.iloc[: 1460, :]

test_df = dataframe2.iloc[1460: , :]
X = train_df.drop('SalePrice', axis=1)

y = train_df['SalePrice']

test_df = test_df.drop('SalePrice', axis=1)
from cuml.ensemble import RandomForestRegressor
forest_regressor = RandomForestRegressor(n_estimators = 150)

forest_regressor.fit(X.values.astype('float32'), y.astype('float32'))

y_pred = forest_regressor.predict(test_df)
y_pred
sample_sub = cudf.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

sample_sub.head()
sample_sub['SalePrice'] = y_pred

sample_sub.to_csv('rf_submission.csv', index=False)