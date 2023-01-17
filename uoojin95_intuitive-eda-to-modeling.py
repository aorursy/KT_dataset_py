# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from scipy import stats
DATA_DIR = '/kaggle/input/house-prices-advanced-regression-techniques/'
train_df = pd.read_csv(DATA_DIR + 'train.csv')

test_df = pd.read_csv(DATA_DIR + 'test.csv')
train_df
test_df
sns.distplot(train_df['SalePrice'])
fig = plt.figure()

res = stats.probplot(train_df['SalePrice'], plot=plt)
train_df['SalePrice'].describe()
lowerbound, upperbound = np.percentile(train_df['SalePrice'], [0.5, 99.5])

print(lowerbound, upperbound)

train_df = train_df.drop(train_df[(train_df['SalePrice']<lowerbound) | (train_df['SalePrice']>upperbound)].index)
train_df['SalePrice'].describe()
# SCALE TARGET VARIABLE

train_df['SalePrice'] = np.log1p(train_df['SalePrice'])
corr_matrix = train_df.corr()

corr_matrix.sort_values(by='SalePrice', inplace=True, axis=1, ascending=False)

plt.figure(figsize=(25,25))

sns.heatmap(corr_matrix, square=True, annot=True, fmt='0.2f')
sns.scatterplot(x=train_df['OverallQual'], y=train_df['SalePrice'])
train_df = train_df.drop(train_df[train_df['OverallQual']<=2].index)

train_df = train_df.drop(train_df[(train_df['OverallQual']==10) & (train_df['SalePrice']<12.5)].index)

train_df = train_df.drop(train_df[(train_df['OverallQual']==4) & (train_df['SalePrice']>12.3)].index)

train_df = train_df.drop(train_df[(train_df['OverallQual']==7) & (train_df['SalePrice']<11.5)].index)
sns.scatterplot(x=train_df['OverallQual'], y=train_df['SalePrice'])
sns.scatterplot(x=train_df['GrLivArea'], y=train_df['SalePrice'])
train_df = train_df.drop(train_df[(train_df['GrLivArea']>3300) & (train_df['SalePrice']<12.5)].index)
sns.scatterplot(x=train_df['GarageCars'], y=train_df['SalePrice'])
sns.scatterplot(x=train_df['GarageArea'], y=train_df['SalePrice'])
train_df = train_df.drop(train_df[train_df['GarageArea']>1230].index)
sns.scatterplot(x=train_df['GarageYrBlt'], y=train_df['SalePrice'], alpha=0.6)
train_df['GarageTotal'] = train_df['GarageArea']*train_df['GarageCars']

test_df['GarageTotal'] = test_df['GarageArea']*test_df['GarageCars']
sns.scatterplot(x=train_df['GarageTotal'], y=train_df['SalePrice'])
train_df = train_df.drop(train_df[train_df['GarageTotal']>3750].index)

train_df = train_df.drop(train_df[(train_df['SalePrice']<11.7) & (train_df['GarageTotal']>2000)].index)
train_df.drop(['GarageArea', 'GarageCars'], axis=1, inplace=True)

test_df.drop(['GarageArea', 'GarageCars'], axis=1, inplace=True)
sns.scatterplot(x=train_df['TotalBsmtSF'], y=train_df['SalePrice'], alpha=0.6)
train_df = train_df.drop(train_df[train_df['TotalBsmtSF']>3000].index)

train_df = train_df.drop(train_df[(train_df['SalePrice']<11.1) & (train_df['TotalBsmtSF']>1000)].index)
sns.scatterplot(x=train_df['YearBuilt'], y=train_df['SalePrice'], alpha=0.6)
sns.scatterplot(x=train_df['FullBath'], y=train_df['SalePrice'], alpha=0.6)
train_df = train_df.drop(train_df[(train_df['SalePrice']<11.2) & (train_df['FullBath']==2)].index)

train_df = train_df.drop(train_df[(train_df['SalePrice']>12.8) & (train_df['FullBath']<=1)].index)
sns.scatterplot(x=train_df['YearRemodAdd'], y=train_df['SalePrice'], alpha=0.6)
train_df['RemodToSold'] = train_df['YrSold']-train_df['YearRemodAdd']

test_df['RemodToSold'] = test_df['YrSold']-test_df['YearRemodAdd']
sns.scatterplot(x=train_df['RemodToSold'], y=train_df['SalePrice'], alpha=0.6)
train_df.drop(['YrSold', 'YearRemodAdd'], axis=1, inplace=True)

test_df.drop(['YrSold', 'YearRemodAdd'], axis=1, inplace=True)
sns.scatterplot(x=train_df['Fireplaces'], y=train_df['SalePrice'], alpha=0.6)
sns.scatterplot(x=train_df['MasVnrArea'], y=train_df['SalePrice'], alpha=0.6)
train_df = train_df.drop(train_df[(train_df['SalePrice']<12.5) & (train_df['MasVnrArea']>1000)].index)

train_df = train_df.drop(train_df[(train_df['SalePrice']<11.5) & (train_df['MasVnrArea']>500)].index)
cat_features = [f for f in train_df.columns if train_df[f].dtype == 'object']

cat_features
def analyzeCategoricalFeature(x, y):

    f, axes = plt.subplots(1,3, figsize=(20,5))

    f.suptitle(x)



    axes[0].set_title("box plot")

    axes[0].tick_params(axis='x', labelrotation=45)

    sns.boxplot(x=train_df[x], y=train_df[y], ax=axes[0])



    axes[1].set_title("stirp plot")

    axes[1].tick_params(axis='x',labelrotation=45)

    sns.stripplot(x=train_df[x], y=train_df[y], jitter=0.4, alpha=0.5, marker="D", size=5, ax=axes[1])



    axes[2].set_title("frequency plot")

    axes[2].tick_params(axis='x',labelrotation=45)

    sns.countplot(x=train_df[x], ax=axes[2])
cat_to_drop = ['Street', 'Alley', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2',

               'BldgType', 'RoofStyle', 'RoofMatl', 'Exterior2nd', 'BsmtFinType2', 'Heating', 'Functional',

               'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
for f in cat_features:

    analyzeCategoricalFeature(f, 'SalePrice')
train_df.drop(cat_to_drop, axis=1, inplace=True)

test_df.drop(cat_to_drop, axis=1, inplace=True)
# missing data cleaning
not_missing_cols = [f for f in train_df.columns if train_df[f].isna().sum() == 0]

not_missing_cols
missing_cols = [f for f in train_df.columns if train_df[f].isna().sum() > 0]

train_df[missing_cols].isna().sum().sort_values()
test_missing_cols = [f for f in test_df.columns if test_df[f].isna().sum() > 0]

test_df[test_missing_cols].isna().sum().sort_values()
# missing value imputing
combined = train_df.drop('SalePrice', axis=1).append(test_df)



missing = combined.isna().sum() > 0

missing_features = missing[missing==True].index

print(missing_features)



for feature in missing_features:

    if combined[feature].dtype == 'object':

        combined[feature] = combined.groupby(['Neighborhood', 'OverallQual'])[feature].transform(lambda x: x.fillna(x.value_counts().index[0]) if (len(x.value_counts().index) > 0) else None)

    else:

        combined[feature] = combined.groupby(['Neighborhood', 'OverallQual'])[feature].transform(lambda x: x.fillna(x.mean()))
missing = combined.isna().sum() > 0

missing_features = missing[missing==True].index

print(missing_features)



for feature in missing_features:

    if combined[feature].dtype == 'object':

        combined[feature] = combined.groupby(['Neighborhood'])[feature].transform(lambda x: x.fillna(x.value_counts().index[0]) if (len(x.value_counts().index) > 0) else None)

    else:

        combined[feature] = combined.groupby(['Neighborhood'])[feature].transform(lambda x: x.fillna(x.mean()))
combined.isna().sum().any()
features_to_encode = [f for f in train_df.columns if train_df[f].dtype == 'object']

features_to_encode
#Categorical Feature Encoding



def getObjectColumnsList(df):

    return [cname for cname in df.columns if df[cname].dtype == "object"]



def PerformOneHotEncoding(df,columnsToEncode):

    return pd.get_dummies(df,columns = columnsToEncode)





cat_cols = getObjectColumnsList(combined)

combined = PerformOneHotEncoding(combined, features_to_encode)
combined
# split again

train_df_final = combined.iloc[0:train_df.shape[0]].copy()



# df_train.loc[:, "SalePrice"] = np.log(train.SalePrice)

test_df_final = combined.iloc[train_df.shape[0]::].copy()
# BUILDING MODEL
import xgboost as xgb

from sklearn.linear_model import Lasso



# models

model_xgb = xgb.XGBRegressor(n_estimators=800, learning_rate=0.25)

model_lasso = Lasso(alpha=0.01)
# predictors = ['OverallQual', 'GrLivArea', 'GarageTotal', 'GarageYrBlt', 'TotalBsmtSF', 'FullBath', 'RemodToSold', 'Fireplaces', 'MasVnrArea']
model_xgb.fit(pd.DataFrame(train_df_final), train_df['SalePrice'])

model_lasso.fit(pd.DataFrame(train_df_final), train_df['SalePrice'])
prediction1 = np.expm1(model_xgb.predict(pd.DataFrame(test_df_final)))

prediction1
prediction2 = np.expm1(model_lasso.predict(pd.DataFrame(test_df_final)))

prediction2
prediction = (prediction1*0.85) + (prediction2*0.15)

prediction
submission = pd.DataFrame({

    'Id': test_df['Id'],

    'SalePrice': prediction

})

submission.to_csv('submission.csv', index=False)