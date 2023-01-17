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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_train.head(5)
df_train.describe()
df_test.head()
df_test.describe()
df_train.info()
df_test.info()
sns.distplot(df_train['SalePrice'])
df_train['SalePrice'].skew()
df_train['SalePrice_log'] = np.log(df_train['SalePrice'])

df_train.drop('SalePrice', axis=1, inplace=True)
sns.distplot(df_train['SalePrice_log'])

print("Skewness: %f" % df_train['SalePrice_log'].skew())
plt.figure(figsize=(15,9))

sns.heatmap(df_train.isnull())
total = df_train.isnull().sum().sort_values(ascending = False)

percentage = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percentage], axis=1, keys=['total', 'percent'])

missing_data.head(10)
df_train = df_train.drop((missing_data[missing_data['total']>81]).index,1)

df_train.head()
df_train.isnull().sum().sort_values(ascending=False)
total_test = df_test.isnull().sum().sort_values(ascending = False)

percentage_test = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)

missing_data_test = pd.concat([total_test, percentage_test], axis=1, keys=['total', 'percent'])

missing_data_test.head(10)
df_test = df_test.drop((missing_data_test[missing_data_test['total']>78]).index,1)

df_test.head()
df_test.isnull().sum().sort_values(ascending = False)
train_num_cols = df_train.select_dtypes(exclude='object').columns

train_cat_cols = df_train.select_dtypes(include='object').columns
train_num_cols
train_cat_cols
test_num_cols = df_test.select_dtypes(exclude='object').columns

test_cat_cols = df_test.select_dtypes(include='object').columns
for i in range(0, len(train_num_cols)):

    df_train[train_num_cols[i]] = df_train[train_num_cols[i]].fillna(df_train[train_num_cols[i]].mean())



for i in range(0, len(test_num_cols)):

    df_test[test_num_cols[i]] = df_test[test_num_cols[i]].fillna(df_test[test_num_cols[i]].mean()) 
for i in range(0, len(train_cat_cols)):

    df_train[train_cat_cols[i]] = df_train[train_cat_cols[i]].fillna(df_train[train_cat_cols[i]].mode()[0])



for i in range(0, len(test_cat_cols)):

    df_test[test_cat_cols[i]] = df_test[test_cat_cols[i]].fillna(df_test[test_cat_cols[i]].mode()[0])
df_train.isnull().sum().max()
df_test.isnull().sum().max()
df_train.corr()['SalePrice_log']
plt.figure(figsize=(20,9))

sns.heatmap(df_train.corr())
df_train.corr()['SalePrice_log'].sort_values(ascending=False)
df_train.drop(['MasVnrArea', 'GarageYrBlt','Fireplaces','BsmtFinSF1','WoodDeckSF','2ndFlrSF','OpenPorchSF','HalfBath','LotArea',

         'BsmtFullBath','BsmtUnfSF','BedroomAbvGr','ScreenPorch','PoolArea','MoSold','3SsnPorch','BsmtFinSF2','BsmtHalfBath',

         'MiscVal','Id','LowQualFinSF','YrSold','OverallCond','MSSubClass','EnclosedPorch','KitchenAbvGr'], axis=1, inplace=True)

df_train.head()
df_train.corr()['SalePrice_log']
train_num_cols1 = df_train.select_dtypes(exclude='object').columns

for col in train_num_cols1:

    print('{:15}'.format(col),'skewness: {}'.format(df_train[col].skew()))
sns.distplot(df_train['GrLivArea'])
df_train['GrLivArea_log'] = np.log(df_train['GrLivArea'])

df_train.drop('GrLivArea', axis=1, inplace=True)
sns.distplot(df_train['GrLivArea_log'])
df_test['GrLivArea_log'] = np.log(df_test['GrLivArea'])

df_test.drop('GrLivArea', axis=1, inplace=True)
df_test.drop(['MasVnrArea', 'GarageYrBlt','Fireplaces','BsmtFinSF1','WoodDeckSF','2ndFlrSF','OpenPorchSF','HalfBath','LotArea',

         'BsmtFullBath','BsmtUnfSF','BedroomAbvGr','ScreenPorch','PoolArea','MoSold','3SsnPorch','BsmtFinSF2','BsmtHalfBath',

         'MiscVal','Id','LowQualFinSF','YrSold','OverallCond','MSSubClass','EnclosedPorch','KitchenAbvGr'], axis=1, inplace=True)
df_test.head(5)
sns.scatterplot('GrLivArea_log', 'SalePrice_log', data=df_train)
train_cat_cols1 = df_train.select_dtypes(include='object').columns

train_cat_cols1
nrows = 13

ncols = 3



fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*4,nrows*3))



for r in range(0, nrows):

    for c in range(0, ncols):

        i = r*ncols + c

        if i < len(train_cat_cols1):

            sns.boxplot(x=train_cat_cols1[i], y= 'SalePrice_log', data= df_train, ax= ax[r][c])



plt.tight_layout()

plt.show()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_train[train_cat_cols1] = df_train[train_cat_cols1].apply(lambda col:le.fit_transform(col.astype(str)))
test_cat_cols1 = df_test.select_dtypes(include='object').columns

test_cat_cols1
df_test[test_cat_cols1] = df_test[test_cat_cols1].apply(lambda col:le.fit_transform(col.astype(str)))
X_train = df_train.drop('SalePrice_log', axis=1)

y_train = df_train['SalePrice_log']
X_test = df_test
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
lm.score(X_train, y_train)
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=3000, max_depth=4, learning_rate=0.05)
gbr.fit(X_train, y_train)
gbr.score(X_train, y_train)
predictions_lm = lm.predict(X_test)

predictions_lm = np.exp(predictions_lm)

predictions_lm
predictions_gbr = gbr.predict(X_test)

predictions_gbr = np.exp(predictions_gbr)

predictions_gbr
final_predict = (predictions_lm + predictions_gbr)*0.5

final_predict
sub_file = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

sub_file.head()
sub_file['SalePrice'] = final_predict
sub_file.head()
sub_file.to_csv('final_submission1.csv', index=False)