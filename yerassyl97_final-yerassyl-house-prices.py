import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as stats

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.preprocessing import normalize

from sklearn.ensemble import GradientBoostingRegressor



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train_df.shape
test_df.shape
train_df.info()
train_df.head()
train_df.describe(include = 'all')
train_df.count(0) / train_df.shape[0] * 100
train_df.columns
cat_cols = list(train_df.select_dtypes(include='object'))

num_cols = list(train_df.select_dtypes(exclude='object'))
cat_cols
num_cols
for col in cat_cols:

    print(col+" : ", train_df[col].unique(), len(train_df[col].unique()))
corr_matrix = train_df.corr()

plt.subplots(figsize = (12,10))

sns.heatmap(corr_matrix, vmax=1, square = True)
train_df.corr()
# Let's derive the age property at the time of sale, YearSold-YearBuilt and drop these two attributes



train_df['age'] = train_df['YrSold'] - train_df['YearBuilt']

train_df.drop(['YrSold','YearBuilt'], axis = 1, inplace = True)
test_df['age'] = test_df['YrSold'] - test_df['YearBuilt']

test_df.drop(['YrSold', 'YearBuilt'], axis = 1, inplace = True )
# Let's drop the least correlated attributes with saleprice



train_df.drop(['MSSubClass','OverallCond','3SsnPorch','PoolArea','MiscVal','MoSold'], axis = 1, inplace = True)
test_df.drop(['MSSubClass','OverallCond','3SsnPorch','PoolArea','MiscVal','MoSold'], axis = 1, inplace = True)
pd.set_option('display.max_columns',85)

pd.set_option('display.max_rows',85)
train_df.corr()
# Let's drop the least correlated attributes with saleprice

train_df.drop(['BsmtFinSF2','LowQualFinSF','BsmtHalfBath'],axis = 1, inplace = True)
test_df.drop(['BsmtFinSF2','LowQualFinSF','BsmtHalfBath'],axis = 1, inplace = True)
train_df.corr()
corr_matrix = train_df.corr()

plt.subplots(figsize = (12,10))

sns.heatmap(corr_matrix, vmax=1, square = True)
train_df.drop(['1stFlrSF','2ndFlrSF','TotRmsAbvGrd','GarageYrBlt','GarageArea'], axis = 1, inplace = True)



test_df.drop(['1stFlrSF','2ndFlrSF','TotRmsAbvGrd','GarageYrBlt','GarageArea'], axis = 1, inplace = True)
corr_matrix = train_df.corr()

plt.subplots(figsize = (12,10))

sns.heatmap(corr_matrix, vmax=1, square = True)
train_df.shape
test_df.shape
cat_cols = list(train_df.select_dtypes(include='object'))

num_cols = list(train_df.select_dtypes(exclude='object'))
cat_cols
num_cols
for col in cat_cols:

    print(col+" : ", train_df[col].unique(), len(train_df[col].unique()))
train_df['Street'] = train_df['Street'].replace({'Grvl':0, 'Pave': 1})
test_df['Street'] = test_df['Street'].replace({'Grvl':0, 'Pave': 1})




train_df['Alley'] = train_df['Alley'].replace( {np.nan:0,'Grvl':1, 'Pave':1 })
test_df['Alley'] = test_df['Alley'].replace( {np.nan:0,'Grvl':1, 'Pave':1 })
train_df['LotShape'] = train_df['LotShape'].replace( {'Reg':'Reg', 'IR1':'Irr', 'IR2':'Irr', 'IR3':'Irr'} )

train_df = pd.get_dummies(data = train_df, columns = ['LotShape'])
test_df['LotShape'] = test_df['LotShape'].replace( {'Reg':'Reg', 'IR1':'Irr', 'IR2':'Irr', 'IR3':'Irr'} )

test_df = pd.get_dummies(data = test_df, columns = ['LotShape'])
train_df = train_df.rename(columns={'LandContour':'IsLevel'})

train_df['IsLevel'] = train_df['IsLevel'].replace( {'Lvl':1, 'Bnk':0, 'Low':0, 'HLS':0} )
test_df = test_df.rename(columns={'LandContour':'IsLevel'})

test_df['IsLevel'] = test_df['IsLevel'].replace( {'Lvl':1, 'Bnk':0, 'Low':0, 'HLS':0} )


train_df = train_df.rename(columns={'Utilities': 'AllUtilities'})

train_df['AllUtilities'] = train_df['AllUtilities'].replace( {'AllPub':1, 'NoSeWa':0} )
test_df = test_df.rename(columns={'Utilities': 'AllUtilities'})

test_df['AllUtilities'] = test_df['AllUtilities'].replace( {'AllPub':1, 'NoSeWa':0} )
train_df = pd.get_dummies(data = train_df, columns= ['LotConfig'])
test_df = pd.get_dummies(data = test_df, columns= ['LotConfig'])


train_df.drop('LandSlope',axis = 1, inplace = True)
test_df.drop('LandSlope',axis = 1, inplace = True)
train_df = pd.get_dummies(data = train_df, columns= ['Neighborhood'])
test_df = pd.get_dummies(data = test_df, columns= ['Neighborhood'])


train_df.drop( ['Condition1','Condition2'] ,axis = 1, inplace = True)
test_df.drop( ['Condition1','Condition2'] ,axis = 1, inplace = True)
train_df = pd.get_dummies(data = train_df, columns=['BldgType','HouseStyle','RoofStyle','RoofMatl'])
test_df = pd.get_dummies(data = test_df, columns=['BldgType','HouseStyle','RoofStyle','RoofMatl'])


train_df.drop( ['Exterior1st','Exterior2nd'] ,axis = 1, inplace = True)

train_df = pd.get_dummies(data = train_df, columns=['ExterQual'])
test_df.drop( ['Exterior1st','Exterior2nd'] ,axis = 1, inplace = True)

test_df = pd.get_dummies(data = test_df, columns=['ExterQual'])
train_df['MasVnrType'] = train_df['MasVnrType'].replace( {np.nan:'None'} )

train_df = pd.get_dummies(data = train_df, columns=['MasVnrType'])





train_df = pd.get_dummies(data = train_df, columns=['ExterCond'])



train_df = pd.get_dummies(data = train_df, columns=['Foundation'])



train_df['BsmtQual'] = train_df['BsmtQual'].replace( {np.nan:'None'} )

train_df = pd.get_dummies(data = train_df, columns=['BsmtQual'])



train_df['BsmtCond'] = train_df['BsmtCond'].replace( {np.nan:'None'} )

train_df = pd.get_dummies(data = train_df, columns=['BsmtCond'])



train_df['BsmtExposure'] = train_df['BsmtExposure'].replace( {np.nan:'None'} )

train_df = pd.get_dummies(data = train_df, columns=['BsmtExposure'])



train_df.drop( ['BsmtFinType1','BsmtFinType2'] ,axis = 1, inplace = True)



test_df['MasVnrType'] = test_df['MasVnrType'].replace( {np.nan:'None'} )

test_df = pd.get_dummies(data = test_df, columns=['MasVnrType'])



test_df = pd.get_dummies(data = test_df, columns=['ExterCond'])



test_df = pd.get_dummies(data = test_df, columns=['Foundation'])



test_df['BsmtQual'] = test_df['BsmtQual'].replace( {np.nan:'None'} )

test_df = pd.get_dummies(data = test_df, columns=['BsmtQual'])



test_df['BsmtCond'] = test_df['BsmtCond'].replace( {np.nan:'None'} )

test_df = pd.get_dummies(data = test_df, columns=['BsmtCond'])



test_df['BsmtExposure'] = test_df['BsmtExposure'].replace( {np.nan:'None'} )

test_df = pd.get_dummies(data = test_df, columns=['BsmtExposure'])



test_df.drop( ['BsmtFinType1','BsmtFinType2'] ,axis = 1, inplace = True)

train_df = pd.get_dummies(data = train_df, columns=['Heating'])

train_df = pd.get_dummies(data = train_df, columns=['HeatingQC'])

train_df['CentralAir'] = train_df['CentralAir'].replace( {'Y':1, 'N':0} )
test_df = pd.get_dummies(data = test_df, columns=['Heating'])

test_df = pd.get_dummies(data = test_df, columns=['HeatingQC'])

test_df['CentralAir'] = test_df['CentralAir'].replace( {'Y':1, 'N':0} )
train_df.shape
test_df.shape
# Let's check null values for Electrical attribute:

train_df['Electrical'].isnull().sum()
test_df['Electrical'].isnull().sum()
train_df['Electrical'].fillna(train_df['Electrical'].mode()[0],inplace = True)
train_df['Electrical'].isnull().sum()
train_df = pd.get_dummies(data = train_df, columns=['Electrical'])
test_df = pd.get_dummies(data = test_df, columns=['Electrical'])
train_df.drop(['KitchenQual'], axis = 1, inplace = True)
test_df.drop(['KitchenQual'], axis = 1, inplace = True)
train_df.shape
test_df.shape
train_df.drop(['Functional'], axis = 1, inplace = True)
test_df.drop(['Functional'], axis = 1, inplace = True)
train_df['FireplaceQu'].isnull().sum()
test_df['FireplaceQu'].isnull().sum()
train_df['FireplaceQu'].fillna('None',inplace = True)
test_df['FireplaceQu'].fillna('None',inplace = True)
train_df['FireplaceQu'].isnull().sum()
test_df['FireplaceQu'].isnull().sum()
train_df = pd.get_dummies(data = train_df, columns=['FireplaceQu'])
test_df = pd.get_dummies(data = test_df, columns=['FireplaceQu'])
train_df.drop(['GarageType','GarageFinish'], axis = 1, inplace = True)
test_df.drop(['GarageType','GarageFinish'], axis = 1, inplace = True)
print(train_df['GarageQual'].isnull().sum())

print(train_df['GarageCond'].isnull().sum())
print(test_df['GarageQual'].isnull().sum())

print(test_df['GarageCond'].isnull().sum())
print(train_df['GarageQual'].value_counts())

print(train_df['GarageCond'].value_counts())
train_df[['GarageQual','GarageCond', 'SalePrice']].groupby(['GarageQual','GarageCond'])['SalePrice'].mean()
train_df['SalePrice'].where( (train_df['GarageCond'].isnull()) & (train_df['GarageQual'].isnull()) ).mean()
train_df['GarageQual'].fillna('Po',inplace = True)

train_df['GarageCond'].fillna('Po', inplace = True)
test_df['GarageQual'].fillna('Po',inplace = True)

test_df['GarageCond'].fillna('Po', inplace = True)
print(train_df['GarageQual'].isnull().sum())

print(train_df['GarageCond'].isnull().sum())
print(test_df['GarageQual'].isnull().sum())

print(test_df['GarageCond'].isnull().sum())
# Now that we have handled Null values for GarageCond and GarageQual attributes, lets use get_dummies

train_df = pd.get_dummies(data = train_df, columns=['GarageQual','GarageCond'])
test_df = pd.get_dummies(data = test_df, columns=['GarageQual','GarageCond'])
train_df.shape
test_df.shape
train_df['PavedDrive'] = train_df['PavedDrive'].replace( {'Y':1,'N':0,'P':0} )
test_df['PavedDrive'] = test_df['PavedDrive'].replace( {'Y':1,'N':0,'P':0} )


train_df.drop('PoolQC', axis = 1, inplace = True)



train_df['Fence'] = train_df['Fence'].replace( {'GdPrv':'CP', 'MnPrv':'PP', 'GdWo':'PP', 'MnWw':'NP', np.nan:'NP'} )

train_df = pd.get_dummies(data = train_df, columns=['Fence'])



# We already have dropped MiscValue column, so let's drop the MiscFeature

train_df.drop('MiscFeature', axis = 1, inplace = True)

test_df.drop('PoolQC', axis = 1, inplace = True)



test_df['Fence'] = test_df['Fence'].replace( {'GdPrv':'CP', 'MnPrv':'PP', 'GdWo':'PP', 'MnWw':'NP', np.nan:'NP'} )

test_df = pd.get_dummies(data = test_df, columns=['Fence'])



test_df.drop('MiscFeature', axis = 1, inplace = True)


train_df.drop('SaleCondition',axis = 1, inplace = True)

train_df.drop('SaleType',axis = 1, inplace = True)

test_df.drop('SaleCondition',axis = 1, inplace = True)

test_df.drop('SaleType',axis = 1, inplace = True)
train_df.shape
test_df.shape
plt.bar(train_df['MSZoning'],train_df['SalePrice'])

plt.show()
train_df = pd.get_dummies(data = train_df, columns=['MSZoning'])
test_df = pd.get_dummies(data = test_df, columns=['MSZoning'])
train_df.shape
test_df.shape
train_df.isnull().sum()
test_df.isnull().sum()
train_df[['LotArea','Alley','LotShape_Irr','LotShape_Reg','IsLevel','LotConfig_Corner','LotConfig_CulDSac','LotConfig_FR2', 'LotConfig_FR3', 'LotConfig_Inside','LotFrontage']].corr()
train_df['LotFrontage'].describe()
train_df['LotFrontage'].hist()
train_df['LotFrontage'].fillna(train_df['LotFrontage'].mean(), inplace = True)
test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean(), inplace = True)
pd.set_option('display.max_rows',170)
train_df.isnull().sum()
train_df['MasVnrArea'].describe()
plt.scatter(train_df['MasVnrArea'],train_df['SalePrice'])
train_df[train_df.columns[1:]].corr()['SalePrice'][:]
sns.scatterplot(x = 'OverallQual', y = 'SalePrice', data=train_df)
sns.scatterplot(x = 'age', y = 'SalePrice', data=train_df)
sns.scatterplot(x = 'MasVnrArea', y = 'SalePrice', data=train_df)
sns.scatterplot(x = 'BsmtFinSF1', y = 'SalePrice', data=train_df)
sns.scatterplot(x = 'TotalBsmtSF', y = 'SalePrice', data=train_df)
sns.scatterplot(x = 'GrLivArea', y = 'SalePrice', data=train_df)
sns.scatterplot(x = 'FullBath', y = 'SalePrice', data=train_df)
sns.scatterplot(x = 'GarageCars', y = 'SalePrice', data=train_df)
sns.scatterplot(x = 'LotFrontage', y = 'SalePrice', data=train_df)
sns.scatterplot(x = 'LotArea', y = 'SalePrice', data=train_df)
train_df.drop( train_df[ (train_df['OverallQual'] > 9) & (train_df['SalePrice'] < 200000) ].index, inplace = True)
train_df.drop( train_df[ (train_df['age'] > 100) & (train_df['SalePrice'] > 300000) ].index, inplace = True)
train_df.drop( train_df[ (train_df['MasVnrArea'] > 1200) & (train_df['SalePrice'] < 700000) ].index, inplace = True)
train_df.drop( train_df[ (train_df['BsmtFinSF1'] > 2000) & (train_df['SalePrice'] < 200000) ].index, inplace = True)
train_df.drop( train_df[ (train_df['TotalBsmtSF'] > 3000) & (train_df['SalePrice'] < 300000) ].index, inplace = True)
train_df.drop( train_df[ (train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000) ].index, inplace = True)
train_df.drop( train_df[ (train_df['FullBath'] < 1) & (train_df['SalePrice'] > 300000) ].index, inplace = True)
train_df.drop( train_df[ (train_df['LotFrontage'] > 200) & (train_df['SalePrice'] < 300000) ].index, inplace = True)
train_df.shape
train_df.isnull().sum()
train_df['MasVnrArea'].fillna(train_df['MasVnrArea'].mean(), inplace = True)
test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].mean(), inplace = True)
train_df_pred = pd.DataFrame(normalize(train_df.drop(['Id','SalePrice'], axis = 1)), columns = train_df.drop(['Id','SalePrice'],axis = 1).columns)
test_df.isnull().sum()
test_df['AllUtilities'].fillna(test_df['AllUtilities'].mean(), inplace = True)
test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].mean(), inplace = True)
test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].mean(), inplace = True)
test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].mean(), inplace = True)
test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mean(), inplace = True)
test_df['GarageCars'].fillna(test_df['GarageCars'].mean(), inplace = True)
test_df_pred = pd.DataFrame(normalize(test_df.drop('Id', axis = 1)), columns = test_df.drop('Id',axis = 1).columns)
train_df_pred.head()
test_df_pred.head()
train_df_pred.reset_index(inplace = True)
train_df.reset_index(inplace = True)
train_df_pred['SalePrice'] = train_df['SalePrice']
train_df_pred.shape
train_df_pred.drop('index', axis = 1, inplace = True)
train_df_pred.shape
train_df_pred.columns.difference(test_df_pred.columns)   
train_df_pred.drop(['Electrical_Mix', 'GarageQual_Ex', 'Heating_Floor', 'Heating_OthW','HouseStyle_2.5Fin', 'RoofMatl_ClyTile', 'RoofMatl_Membran','RoofMatl_Metal', 'RoofMatl_Roll'], axis = 1, inplace = True)
print(train_df_pred.shape)

print(test_df_pred.shape)
train_df_pred.columns.difference(test_df_pred.columns)
train_df_pred.head()
test_df_pred.head()
sns.distplot(train_df_pred['SalePrice']).set_title("Distribution of SalePrice")
# probability plot

fig = plt.figure()

res = stats.probplot(train_df_pred['SalePrice'], plot=plt)
train_df_pred["SalePrice"] = np.log1p(train_df_pred["SalePrice"])
sns.distplot(train_df_pred['SalePrice'] )
(mu, sigma) = stats.norm.fit(train_df_pred['SalePrice'])

print( '\n mean = {:.2f} and std dev = {:.2f}\n'.format(mu, sigma))
fig = plt.figure()

res = stats.probplot(train_df_pred['SalePrice'], plot=plt)

plt.show()
train_df_pred.isnull().sum()
X_train, X_val, y_train, y_val = train_test_split(train_df_pred.drop('SalePrice', axis = 1), train_df_pred['SalePrice'], test_size = 0.33, shuffle = True, random_state = 42)
X_train.shape
X_val.shape
y_train.shape
y_val.shape
LRM = LinearRegression()
LRM.fit(X_train,y_train)
LRM.score(X_val,y_val)
predictions = LRM.predict(X_val)
print(mean_squared_error(y_val,predictions))
GBR = GradientBoostingRegressor(max_depth = 3, n_estimators = 1500, verbose = 1, random_state = 42)
GBR.fit(X_train,y_train)
GBR.score(X_val,y_val)
predictions_GBR = GBR.predict(X_val)



print(mean_squared_error(y_val,predictions_GBR))



print(test_df.shape)

print(test_df_pred.shape)
predictions_test = GBR.predict(test_df_pred)
test_df.head()
len(predictions_test)
predictions_test.max()
predictions_test = np.expm1(predictions_test)
submission = test_df[['Id']].copy()
submission['SalePrice'] = predictions_test
submission.head()
submission.to_csv('submission_final.csv', index=False, header=True)