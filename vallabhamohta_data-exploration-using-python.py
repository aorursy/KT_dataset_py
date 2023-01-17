#Importing the libraries

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

#Reading the data

df_train = pd.read_csv('../input/housetrain.csv')

#View the variables

df_train.columns
#Descriptive statistics summary

df_train['SalePrice'].describe()
#Histogram for visualising the spread

sns.distplot(df_train['SalePrice']);
#To check the skewness and kurtosis of the data

print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#Scatterplot 1

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000));
#Scatterplot 2

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000));
#Boxplot 1

var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

f, ax = plt.subplots(figsize = (8, 6))

fig = sns.boxplot(x = var, y = "SalePrice", data = data)

fig.axis(ymin = 0, ymax = 800000);
#Boxplot 2

var = 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

f, ax = plt.subplots(figsize = (16, 8))

fig = sns.boxplot(x = var, y = "SalePrice", data = data)

fig.axis(ymin = 0, ymax = 800000);

plt.xticks(rotation = 90);
#Correlation matrix - Heatmap style

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
#Correlation matrix with SalePrice

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#Scatterplots

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], size = 2.5)

plt.show();
#Missing values

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#Dealing with missing values

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max()
#Data standardisation

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#Checking outliers

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000));
#Removing outliers

df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]

df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)

df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
#Bivariate Analysis

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000));
#Histogram and normal probability plot

sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
#Applying log transformation

df_train['SalePrice'] = np.log(df_train['SalePrice'])
#Transformed histogram and normal probability plot

sns.distplot(df_train['SalePrice'], fit = norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot = plt)
#Histogram and normal probability plot

sns.distplot(df_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
#Data transformation

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
#Transformed histogram and normal probability plot

sns.distplot(df_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
#Histogram and normal probability plot

sns.distplot(df_train['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
#Create column for new variable (one is enough because it's a binary categorical feature)

#if area>0 it gets 1, for area==0 it gets 0 (as significant number of data points below 0)

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)

df_train['HasBsmt'] = 0 

df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#Transform data

df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
#Histogram and normal probability plot

sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
#Scatterplot for homoscedasticity

plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);
#Scatterplot

plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);
#Convert categorical variable into dummy variable

df_train = pd.get_dummies(df_train)