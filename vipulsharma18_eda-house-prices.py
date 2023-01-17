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

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_train.head()
df_train.columns
df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice']);
print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
def plot_custom(var):

    data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

    data.plot.scatter(x= var, y = 'SalePrice', ylim = (0, 800000));
plot_custom('GrLivArea')
plot_custom('TotalBsmtSF')
def plot_custom_cat(var, figsizeCustom):

    data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

    f, ax = plt.subplots(figsize= figsizeCustom)

    fig = sns.boxplot(x = var, y = "SalePrice", data = data)

    fig.axis(ymin = 0, ymax = 800000);
plot_custom_cat('OverallQual', figsizeCustom = (8,6))
var = 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

f, ax = plt.subplots(figsize = (16,8))

fig = sns.boxplot(x = var, y = 'SalePrice', data = data)

fig.axis(ymin = 0, ymax = 800000);

plt.xticks(rotation = 90);
#correlation matrix or heatmap

corrmat = df_train.corr()
f, ax = plt.subplots(figsize = (12,9))

sns.heatmap(corrmat, vmax = 0.8, square = True);
# Taking variables most correlated with SalePrice and measuring correlation between them

print(corrmat.nlargest(10, 'SalePrice').index)

print(corrmat.nlargest(10, 'SalePrice')['SalePrice'].index)
#heatmap of top 10 variables with highest correlation

k = 10

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

print(cm)
sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot = True, square = True, fmt='.2f', annot_kws = {'size':10}, yticklabels = cols.values, xticklabels=cols.values)

plt.show()
#scatterPlots between all variables

sns.set()

cols = ['SalePrice', 'OverallQual', "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]

sns.pairplot(df_train[cols], size = 2.5)

plt.show()
total = df_train.isnull().sum().sort_values(ascending=False)

percent = ((df_train.isnull().sum()/df_train.isnull().count())*100).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis = 1 ,keys = ['Total', 'Percent'])

missing_data.head(20)
cols = ['SalePrice','PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage',

       'GarageCond', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual',

       'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual',

       'MasVnrArea', 'MasVnrType', 'Electrical', 'Utilities']

sns.set()

sns.pairplot(df_train[cols], size = 2.5)

plt.show()
df_train = df_train.drop((missing_data[missing_data['Total']>1]).index, 1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max()

#to check if any missing values left
#standardizing data

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][: ,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[: ,0].argsort()][:10]

high_range = saleprice_scaled[saleprice_scaled[: ,0].argsort()][-10:]

print('outer range(low) of the distribution: ')

print(low_range)

print('\nouter range (high) of the distribution: ')

print(high_range)
#bivariate analysis saleprice/grlivarea

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#deleting points

df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)

df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
#bivariate analysis saleprice/grlivarea

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#histogram and normal probability plot

sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
#applying log transformation

df_train['SalePrice'] = np.log(df_train['SalePrice'])
#transformed histogram and normal probability plot

sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
df_train['SalePrice'].describe()
#histogram and normal probability plot

sns.distplot(df_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
#data transformation

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
#transformed histogram and normal probability plot

sns.distplot(df_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
#histogram and normal probability plot

sns.distplot(df_train['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
#create column for new variable (one is enough because it's a binary categorical feature)

#if area>0 it gets 1, for area==0 it gets 0

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)

df_train['HasBsmt'] = 0 

df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data

df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
#histogram and normal probability plot

sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit = norm);

fig = plt.figure()

res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot = plt)
plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);

#due to normalization the heterscedastic data has been converted into homoscedastic
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);

#here also the variance of TotalBsmtSf accross SalePrice variable is uniform
df_train.head()
#convert categorical variable into dummy

df_train = pd.get_dummies(df_train)
df_train.head()
df_train.to_csv('TrainingDATABasicCleaning.csv', index = False)