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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

df_test= pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

df= pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df.info()
corrmat=df.corr()

f, ax=plt.subplots(figsize=(15,10))

sns.heatmap(corrmat, vmax=1, square=True)
total=df.isnull().sum().sort_values(ascending=False)

percent=(total/df.isnull().count()).sort_values(ascending=False)

missing_data=pd.concat([total,percent], axis=1,keys=['Total','percent']).sort_values(by=['Total'], ascending=False)

missing_data
#find outliers

var='GrLivArea'

data=pd.concat([df['SalePrice'],df[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice')
#GrLivarea above 4500 has weird low saleprice, delete.

df.sort_values(by='GrLivArea', ascending=False)[:2]
df=df.drop(index=df[df['Id']==1299].index)

df=df.drop(index=df[df['Id']==524].index)
var='GrLivArea'

data=pd.concat([df['SalePrice'],df[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice')
#Normality to get homoscedasticity

print('skewness: %f'% df['SalePrice'].skew())

print('kurtosis: %f' % df['SalePrice'].kurt())
from scipy.stats import norm

sns.distplot(df['SalePrice'], fit=norm)

import scipy.stats as stats

fig = plt.figure() #must open a new figure, otherwise two figures would draw together.

res = stats.probplot(df['SalePrice'], plot=plt)
#From not normal to normal. when positive skewness, log transformations usually works well.

df['SalePrice']=np.log(df['SalePrice'])
sns.distplot(df['SalePrice'], fit=norm)

fig=plt.figure()

res=stats.probplot(df['SalePrice'],plot=plt)
df['GrLivArea']=np.log(df['GrLivArea'])

sns.distplot(df['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df['GrLivArea'], plot=plt)
plt.scatter(df['GrLivArea'], df['SalePrice']);
df['1stFlrSF']=np.log(df['1stFlrSF'])

sns.distplot(df['1stFlrSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df['1stFlrSF'], plot=plt)
#dealing with missing data

#too much missing data will be deleted

df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1, inplace=True)
#delect not important features from analysis of heatmap

df.drop(['Id', 'MSSubClass', '1stFlrSF', '2ndFlrSF', 'BsmtFullBath', 'OverallCond', 'BsmtFinSF2', 'BsmtUnfSF', 'LowQualFinSF', 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'LotArea', 'GarageYrBlt'], axis=1, inplace=True)
df.info()
#fill missing data

df['MasVnrType'].unique()
df['MasVnrType'].fillna('None', inplace=True)
df['MasVnrType'].unique()
df['MasVnrArea'].fillna(0, inplace=True)
df['BsmtCond'].fillna('None', inplace=True)

df['BsmtQual'].fillna('None', inplace=True)

df['BsmtExposure'].fillna('None', inplace=True)

df['BsmtFinType1'].fillna('None', inplace=True)

df['BsmtFinType2'].fillna('None', inplace=True)

df.dropna(subset=['Electrical'], inplace=True)

df['GarageType'].fillna('None', inplace=True)

df['GarageFinish'].fillna('None', inplace=True)

df['GarageQual'].fillna('None', inplace=True)

df['GarageCond'].fillna('None', inplace=True)
df.info()
#now we have three data types: categorical(get dummies later), numerical and year. Year will transferred.

df['YearBuiltBand']=pd.cut(df['YearBuilt'],13, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13]).astype('int64')

df['YearRemodAddBand']=pd.cut(df['YearRemodAdd'],6,labels=[1,2,3,4,5,6]).astype('int64')

df.drop(['YearBuilt', 'YearRemodAdd'], axis=1, inplace=True)
df.info()
#deal with test dataset

Id=df_test['Id'].values
df_test.head()
df_test.info()
df_test['GrLivArea']=np.log(df_test['GrLivArea'])

df_test['1stFlrSF']=np.log(df_test['1stFlrSF'])
df_test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1, inplace=True)

df_test.drop(['Id', 'MSSubClass', '1stFlrSF', '2ndFlrSF', 'BsmtFullBath', 'OverallCond', 'BsmtFinSF2', 'BsmtUnfSF', 'LowQualFinSF', 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'LotArea', 'GarageYrBlt'], axis=1, inplace=True)
df_test.info()
df_test['MSZoning'].fillna('None', inplace=True)

df_test['Utilities'].fillna('None', inplace=True)

df_test['MasVnrType'].fillna('None', inplace=True)

df_test['MasVnrArea'].fillna(0, inplace=True)

df_test['BsmtCond'].fillna('None', inplace=True)

df_test['BsmtQual'].fillna('None', inplace=True)

df_test['BsmtExposure'].fillna('None', inplace=True)

df_test['BsmtFinType1'].fillna('None', inplace=True)

df_test['BsmtFinType2'].fillna('None', inplace=True)

df_test['GarageCond'].fillna('None', inplace=True)

df_test['GarageType'].fillna('None', inplace=True)

df_test['GarageFinish'].fillna('None', inplace=True)

df_test['GarageQual'].fillna('None', inplace=True)

df_test['Exterior1st'].fillna('None', inplace=True)

df_test['Exterior2nd'].fillna('None', inplace=True)
df_test.info()
df_test['BsmtFinSF1'].fillna(0, inplace=True)

df_test['TotalBsmtSF'].fillna(0, inplace=True)

df_test['GarageCars'].fillna(0, inplace=True)

df_test['SaleType'].fillna('Oth', inplace=True)

df_test['Functional'].fillna('Typ', inplace=True)

df_test['KitchenQual'].fillna('Typ', inplace=True)
df_test['YearBuiltBand']=pd.cut(df_test['YearBuilt'],13, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13]).astype('int64')

df_test['YearRemodAddBand']=pd.cut(df_test['YearRemodAdd'],6,labels=[1,2,3,4,5,6]).astype('int64')

df_test.drop(['YearBuilt', 'YearRemodAdd'], axis=1, inplace=True)
df_test.info()
all_data=pd.concat((df, df_test))

for column in all_data.select_dtypes(include=[np.object]).columns:

    print(column, all_data[column].unique())
from pandas.api.types import CategoricalDtype

all_data=pd.concat((df, df_test))

for column in all_data.select_dtypes(include=[np.object]).columns:

    df[column]=df[column].astype(CategoricalDtype(categories=all_data[column].unique()))

    df_test[column]=df_test[column].astype(CategoricalDtype(categories=all_data[column].unique()))
df=pd.get_dummies(df)

df_test=pd.get_dummies(df_test)
df.shape
df_test.shape
#we use feature importance with random forests

X=df[df.loc[:,df.columns!='SalePrice'].columns].values

y=df['SalePrice'].values

from sklearn.ensemble import RandomForestRegressor

forest=RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)

forest.fit(X, y)

importances=forest.feature_importances_
feat_labels=df.loc[:,df.columns!='SalePrice'].columns

indices=np.argsort(importances)[::-1]

for f in range(X.shape[1]):

    print("%d) %-s (%f)" % (f + 1, feat_labels[indices[f]], importances[indices[f]]))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=1)
#now we use PCA

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline

pipe_lr=Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=30)), ('clf', LinearRegression())])

pipe_lr.fit(X_train, y_train)

print('Test Accuracy: %.3f' %pipe_lr.score(X_test, y_test))
#model evaluation

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores=learning_curve(estimator=pipe_lr, X=X_train, y=y_train, train_sizes=np.linspace(0.1,1.0,10), cv=10, n_jobs=-1)

train_means=np.mean(train_scores, axis=1)

test_means=np.mean(test_scores, axis=1)

ylim=(0, 1.0)

plt.plot(train_sizes, train_means, color='red')

plt.plot(train_sizes, test_means, color='blue')
#validation curve to see find how many features to use

from sklearn.model_selection import validation_curve

param_range=[20, 30, 40,50, 60, 70, 80, 90, 100]

pipe_lr=Pipeline([('scl', StandardScaler()), ('pca', PCA()), ('clf', LinearRegression())])

train_scores, test_scores=validation_curve(estimator=pipe_lr, X=X_train, y=y_train, param_name='pca__n_components', param_range=param_range, cv=10)

train_means=np.mean(train_scores, axis=1)

test_means=np.mean(test_scores, axis=1)

plt.plot(param_range, train_means, color='red')

plt.plot(param_range, test_means, color='blue')
pipe_lr=Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=30)), ('clf', LinearRegression())])

pipe_lr.fit(X, y)

predictions=pipe_lr.predict(df_test)

predictions
predictions=np.exp(predictions)

predictions.round(1)
submission=pd.DataFrame({'Id': Id, 'SalePrice': predictions})

submission.to_csv('submission.csv', index=False)