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

import matplotlib.pylab as plt

plt.style.use('bmh')

%matplotlib inline

import seaborn as sns



from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import scale



from sklearn.linear_model import LinearRegression, Ridge, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor



pd.set_option('display.max_columns', None)
import pandas as pd



# Read the data

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

print('train shape:', train.shape, '\n', 'test shape:', test.shape)

train.head()
# Check Null Value

train_null = train.isnull().sum().to_frame().reset_index().rename(columns={"index": "Col_Name", 0:"train"})

test_null = test.isnull().sum().to_frame().reset_index().rename(columns={"index": "Col_Name", 0:"test"})



# Null Value Percentage

train_null_percentage = (train.isnull().sum() * 100 / len(train)).to_frame().reset_index().rename(columns={"index": "Col_Name", 0:"train_percentage"})

test_null_percentage = (test.isnull().sum() * 100 / len(test)).to_frame().reset_index().rename(columns={"index": "Col_Name", 0:"test_percentage"})
df_concat = pd.concat([train_null['Col_Name'], train_null['train'], train_null_percentage['train_percentage'], test_null['test'], test_null_percentage['test_percentage']], axis=1)

df_concat[(df_concat['train_percentage']>15)].sort_values(['train','test'], ascending=False)
# train and test dataset in an array



datasets = [train, test]
# Drop the features which have high no of null values and having timestamp in them.



feature_drop = ['PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageYrBlt', 'MoSold', 'YrSold', 

                'LowQualFinSF', 'MiscVal', 'PoolArea']



for df in datasets:

    df.drop(feature_drop, axis=1, inplace=True)
# If the house has no Alley, it will have missing value , so just fill NaNs.



for df in datasets:

    df.loc[df['Alley'].isnull(), 'Alley'] = 'NoAlley'
# If a house has no garage, it will have missing value on the 'Garage related' features, so just fill NaNs with 'NoGarage'.



for df in datasets:

    df.loc[df['GarageCond'].isnull(), 'GarageCond'] = 'NoGarage'

    df.loc[df['GarageQual'].isnull(), 'GarageQual'] = 'NoGarage'

    df.loc[df['GarageType'].isnull(), 'GarageType'] = 'NoGarage'

    df.loc[df['GarageFinish'].isnull(), 'GarageFinish'] = 'NoGarage'
# If a house has no basement, it will have missing value on the 'basement related' features, so just fill NaNs with 'NoBsmt'. 



for df in datasets:   

    df.loc[df['BsmtExposure'].isnull(), 'BsmtExposure'] = 'NoBsmt'

    df.loc[df['BsmtFinType2'].isnull(), 'BsmtFinType2'] = 'NoBsmt'

    df.loc[df['BsmtCond'].isnull(), 'BsmtCond'] = 'NoBsmt'

    df.loc[df['BsmtQual'].isnull(), 'BsmtQual'] = 'NoBsmt'

    df.loc[df['BsmtFinType1'].isnull(), 'BsmtFinType1'] = 'NoBsmt'
# Masonry veneer feature: just fill with 'None' if there is no Masonry veneer.  



for df in datasets:  

    df.loc[df['MasVnrType'].isnull(), 'MasVnrType'] = 'None'

    df.loc[df['MasVnrArea'].isnull(), 'MasVnrArea'] = 0
# Electrical is the categorical column hence fill it with mode



train['Electrical'].fillna(train['Electrical'].mode()[0], inplace=True)
# other numerical and categorical missing value columns

# fill numrical with 0

# fill categorcial with Mode value.



test_numeric_missing = ['BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageArea', 'GarageCars', 'TotalBsmtSF']

test_categorical_missing = ['MSZoning', 'Functional', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType']



for i in test_numeric_missing:

    test[i].fillna(0, inplace=True)

for j in test_categorical_missing:

    test[j].fillna(test[j].mode()[0], inplace=True)
# After imputing all missing values

# Check the missing values again for datasets



missing_numeric = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['train', 'test'])

missing_numeric = missing_numeric[(missing_numeric['train']>0) | (missing_numeric['test']>0)]

missing_numeric.sort_values(by=['train', 'test'], ascending=False)
train.select_dtypes(exclude=[object]).describe()
print(train['SalePrice'].describe(), '\n')

print('Before Transformation Skew: ', train['SalePrice'].skew())



target = np.log1p(train['SalePrice'])

print('Log Transformation Skew: ', target.skew())
plt.rcParams['figure.figsize'] = (12, 5)

target_log_tran = pd.DataFrame({'before transformation':train['SalePrice'], 'log transformation': target})

target_log_tran.hist()
## Checking the skewness for the numerical values

## Pointing out the features whose Skewness is greater than 0.8



skewness = pd.DataFrame({'Skewness':train.select_dtypes(exclude=[object]).skew()})



print(skewness[skewness['Skewness']>0.8].sort_values(by='Skewness'), '\n')  

print(skewness[skewness['Skewness']>0.8].sort_values(by='Skewness').index.tolist())
skews = ['2ndFlrSF', 'BsmtUnfSF', 'GrLivArea', '1stFlrSF', 'MSSubClass', 'TotalBsmtSF', 'WoodDeckSF', 'BsmtFinSF1', 'OpenPorchSF', 

         'MasVnrArea', 'EnclosedPorch', 'BsmtHalfBath', 'ScreenPorch', 'BsmtFinSF2', 'KitchenAbvGr', '3SsnPorch', 'LotArea']

for df in datasets:

    for s in skews:

        df[s] = np.log1p(df[s])
numeric_data_select = train[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF', 

                             'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'Fireplaces', 'BsmtFinSF1',

                            'WoodDeckSF', '2ndFlrSF', 'OpenPorchSF', 'HalfBath', 'LotArea', 'BsmtFullBath', 'BsmtUnfSF']]



corr = numeric_data_select.corr()

plt.figure(figsize=(12, 12))

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, vmax=1, square=True, annot=True, mask=mask, cbar=False, linewidths=0.1)

plt.xticks(rotation=45)
cust_corr = corr[(corr>=.9) | (corr <= .1)]

mask = np.zeros_like(cust_corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(25, 10))

    ax = sns.heatmap(cust_corr, mask=mask, cmap='Greens', vmax=.3, square=True, annot=True,linewidths=0.1, annot_kws={'size':10})
# Ascending Correaltion



corr = train.select_dtypes(exclude=[object]).corr()

print(corr['SalePrice'].sort_values(ascending=False).head(8))
# Descending Correlation 



print(corr['SalePrice'].sort_values(ascending=False).tail(8))
plt.figure(figsize=(18,5))

train.drop(['SalePrice', 'Id'], axis=1).boxplot()

plt.xticks(rotation=70)
print('Train Shape Before :',train.shape )

Q1_train = train.quantile(0.25)

Q3_train = train.quantile(0.75)

IQR_train = Q3_train - Q1_train

train = train[~((train < (Q1_train - 1.5 * IQR_train)) |(train > (Q3_train + 1.5 * IQR_train))).any(axis=1)]

print('Train Shape After :',train.shape )
print('Test Shape Before :',test.shape )

Q1_test = test.quantile(0.25)

Q3_test = test.quantile(0.75)

IQR_test = Q3_test - Q1_test

test = test[~((test < (Q1_test - 1.5 * IQR_test)) |(test > (Q3_test + 1.5 * IQR_test))).any(axis=1)]

print('Test Shape After :',test.shape )
#sns.pairplot(numeric_data_select, size=2)
plt.rcParams['figure.figsize'] = (12, 4)

plt.subplot(121)

sns.boxplot(train['OverallQual'], target)
plt.subplot(121)

plt.scatter(train['GrLivArea'], train['SalePrice'])

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')

plt.subplot(122)

plt.scatter(train['GarageArea'], train['SalePrice'])

plt.xlabel('GarageArea')

plt.ylabel('SalePrice')
plt.subplot(121)

plt.scatter(train['TotalBsmtSF'], train['SalePrice'])

plt.xlabel('TotalBsmtSF')

plt.ylabel('SalePrice')



plt.subplot(122)

plt.scatter(train['MasVnrArea'], train['SalePrice'])

plt.xlabel('MasVnrArea')

plt.ylabel('SalePrice')
plt.subplot(121)

sns.boxplot(train['Fireplaces'], train['SalePrice'])



plt.subplot(122)

plt.scatter(train['YearBuilt'], train['SalePrice'])

plt.xlabel('YearBuilt')
plt.subplot(121)

plt.scatter(train['BsmtFinSF1'], train['SalePrice'])

plt.xlabel('BsmtFinSF1')

plt.ylabel('SalePrice')



plt.subplot(122)

plt.scatter(train['WoodDeckSF'], train['SalePrice'])

plt.xlabel('WoodDeckSF')

plt.ylabel('SalePrice')
categorical_data = train.select_dtypes(include=[object])

categorical_data.describe()
plt.rcParams['figure.figsize'] = (12, 7)

plt.subplot(221)

sns.boxplot(train['ExterQual'], target)

plt.subplot(222)

sns.boxplot(train['BsmtQual'], target)

plt.subplot(223)

sns.boxplot(train['BsmtExposure'], target)

plt.subplot(224)

sns.boxplot(train['GarageFinish'], target)
plt.subplot(221)

sns.boxplot(train['CentralAir'], target)

plt.subplot(222)

sns.boxplot(train['KitchenQual'], target)
train_ExterQual_dummy = pd.get_dummies(train['ExterQual'], prefix='ExterQual')

test_ExterQual_dummy = pd.get_dummies(test['ExterQual'], prefix='ExterQual')



train_BsmtQual_dummy = pd.get_dummies(train['BsmtQual'], prefix='BsmtQual')

test_BsmtQual_dummy = pd.get_dummies(test['BsmtQual'], prefix='BsmtQual')



train_BsmtExposure_dummy = pd.get_dummies(train['BsmtExposure'], prefix='BsmtExposure')

test_BsmtExposure_dummy = pd.get_dummies(test['BsmtExposure'], prefix='BsmtExposure')



train_GarageFinish_dummy = pd.get_dummies(train['GarageFinish'], prefix='GarageFinish')

test_GarageFinish_dummy = pd.get_dummies(test['GarageFinish'], prefix='GarageFinish')



train_SaleCondition_dummy = pd.get_dummies(train['SaleCondition'], prefix='SaleCondition')

test_SaleCondition_dummy = pd.get_dummies(test['SaleCondition'], prefix='SaleCondition')



train_CentralAir_dummy = pd.get_dummies(train['CentralAir'], prefix='CentralAir')

test_CentralAir_dummy = pd.get_dummies(test['CentralAir'], prefix='CentralAir')



train_KitchenQual_dummy = pd.get_dummies(train['KitchenQual'], prefix='KitchenQual')

test_KitchenQual_dummy = pd.get_dummies(test['KitchenQual'], prefix='KitchenQual')
# Define a model evaluation function by outputing R2 score and mean squared error. (using 10-fold cross validation)

    

def model_eval(model):

    model_fit = model.fit(X, y)

    R2 = cross_val_score(model_fit, X, y, cv=10 , scoring='r2').mean()

    MSE = -cross_val_score(lr, X, y, cv=10 , scoring='neg_mean_squared_error').mean()

    print('For', model,'\n','- R2 Score:', R2, '|', 'MSE:', MSE,'\n')
data = train.select_dtypes(exclude=[object])

y = np.log1p(data['SalePrice'])

X = data.drop(['Id', 'SalePrice'], axis=1)

X = pd.concat([X, train_ExterQual_dummy, train_BsmtQual_dummy, train_GarageFinish_dummy, train_BsmtExposure_dummy,

              train_SaleCondition_dummy, train_CentralAir_dummy, train_KitchenQual_dummy], axis=1)
lr = LinearRegression()

ri = Ridge(alpha=0.1, normalize=False)

ricv = RidgeCV(cv=5)

gdb = GradientBoostingRegressor(n_estimators=200)
for model in [lr, ri, ricv, gdb]:

    model_eval(model)
test_id = test['Id']

test = test.select_dtypes(exclude=[object])#.drop('Id', axis=1)

test = pd.concat([test, test_ExterQual_dummy, test_BsmtQual_dummy, test_GarageFinish_dummy, test_BsmtExposure_dummy,

              test_SaleCondition_dummy, test_CentralAir_dummy, test_KitchenQual_dummy], axis=1)
pred = ri.predict(test)



pred = np.expm1(pred)

prediction = pd.DataFrame({'Id':test_id, 'SalePrice':pred})

prediction.to_csv('Prediction1.csv', index=False)

prediction.head()
pred_2 = lr.predict(test)



pred_2 = np.expm1(pred_2)

prediction = pd.DataFrame({'Id':test_id, 'SalePrice':pred_2})

prediction.to_csv('Prediction1.csv', index=False)

prediction.head()
pred_3 = gdb.predict(test)



pred_3 = np.expm1(pred_3)

prediction = pd.DataFrame({'Id':test_id, 'SalePrice':pred_3})

prediction.to_csv('Prediction1.csv', index=False)

prediction.head()