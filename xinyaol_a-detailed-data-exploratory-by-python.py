# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import math

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

import xgboost as xgb

import seaborn as sns
# EDA purposes

## 1. check data types for variables

## 2. summarize missing values for variables

## 3. plot ditributions for variables

## 4. get the correlations between dependent variable and independent variables

## 5. transfrom the target variables



## got insights and used several functions from Angela & Alexandru Papiu

## https://www.kaggle.com/xchmiao/house-prices-advanced-regression-techniques/detailed-data-exploration-in-python/comments

## https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models/discussion
data = pd.read_csv("../input/train.csv")

data.head()
# 1. check data types for variables

print(data.select_dtypes(include = ['float64']).dtypes)

print(data.select_dtypes(include = ['int64']).dtypes)

print(data.select_dtypes(include = ['object']).dtypes)
# 2. summarize missing values for variables

miss_df = pd.DataFrame({'Column': data.isnull().sum().index, 

                        'Num of Missing': data.isnull().sum()}, index = None)

miss_df.set_index('Column', inplace = True)

miss_df = miss_df.rename_axis(None)

miss_df.sort_values(by = ['Num of Missing'], ascending = [False])
# 3. plot ditributions for numeric variables

num_df = data.select_dtypes(include = ['float64', 'int64']).iloc[:, 0:21]

m = len(num_df.columns)



fig = plt.figure(dpi = 200, figsize = (20, 25))



for i in range(1, m):

    col = num_df.iloc[:, i]

    ax = fig.add_subplot(math.ceil(m / 4), 4, i)

    ax.plot = sns.distplot(col[~ np.isnan(col)])

    plt.xlabel(num_df.columns[i], fontsize = 10)

    plt.ylabel("Probability")
# 3. plot ditributions for numeric variables (cont.)

num_df = data.select_dtypes(include = ['float64', 'int64']).iloc[:, 21:]

m = len(num_df.columns)



fig = plt.figure(dpi = 200, figsize = (18, 20))



for i in range(1, m):

    col = num_df.iloc[:, i]

    ax = fig.add_subplot(math.ceil(m / 4), 4, i)

    ax.plot = sns.distplot(col[~ np.isnan(col)])

    plt.xlabel(num_df.columns[i], fontsize = 8)

    plt.ylabel("Probability")
# 3. plot ditributions for categorical variables

cat_df = data.select_dtypes(include = ['object']).iloc[:, 0:16]

cat_list = cat_df.columns

m = len(cat_list)



fig = plt.figure(dpi = 200, figsize = (15, 20))



for i in range(0, m):

    col = cat_df.iloc[:, i]

    ax = fig.add_subplot(4, 4, i + 1)

    ax.plot = sns.countplot(x = cat_list[i], data = cat_df, palette = "Greens_d");

    plt.xlabel(cat_list[i], fontsize = 8)

    plt.ylabel("Probability")
# 3. plot ditributions for categorical variables (cont.)

cat_df = data.select_dtypes(include = ['object']).iloc[:, 16:32]

cat_list = cat_df.columns

m = len(cat_list)



fig = plt.figure(dpi = 200, figsize = (15, 20))



for i in range(0, m):

    col = cat_df.iloc[:, i]

    ax = fig.add_subplot(4, 4, i + 1)

    ax.plot = sns.countplot(x = cat_list[i], data = cat_df, palette="Greens_d");

    plt.xlabel(cat_list[i], fontsize = 8)

    plt.ylabel("Probability")
# 3. plot ditributions for categorical variables (cont.)

cat_df = data.select_dtypes(include = ['object']).iloc[:, 32:]

cat_list = cat_df.columns

m = len(cat_list)



fig = plt.figure(dpi = 200, figsize = (15, 15))



for i in range(0, m):

    col = cat_df.iloc[:, i]

    ax = fig.add_subplot(3, 4, i + 1)

    ax.plot = sns.countplot(x = cat_list[i], data = cat_df, palette="Greens_d");

    plt.xlabel(cat_list[i], fontsize = 8)

    plt.ylabel("Probability")
# 4. get the correlations between dependent variable and independent variables

## correlation matrix

corr = data.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()

plt.figure(figsize=(12, 12))

sns.heatmap(corr, vmax = 1, square = True)
## rank the correlation

corr_price = corr['SalePrice']

corr_df = pd.DataFrame({'Column': corr_price.index, 

                        'Correlation': corr_price,

                        'Abs Value': abs(corr_price)}, index = None)

corr_df = corr_df.drop(['SalePrice'], axis = 0)

corr_df.set_index('Column', inplace = True)

corr_df = corr_df.rename_axis(None)

corr_df = corr_df.sort_values(by = ['Abs Value'], ascending = [False])

corr_df.iloc[:, 1:2]
## plot high correlated features

fig = plt.figure(dpi = 100, figsize = (5, 5))

ax = sns.regplot(x = 'OverallQual', y = 'SalePrice', data = data) 

plt.xlabel('OverallQual', fontsize = 10)

plt.ylabel("Sale Price", fontsize = 10)
## plot high correlated features (cont.)

fig = plt.figure(dpi = 100, figsize = (18, 3))



ax1 = fig.add_subplot(1, 3, 1)

ax1.plot = sns.barplot(x = 'GarageCars', y= 'SalePrice', data = data, order = np.sort(data.GarageCars.unique()))

plt.xlabel('GarageCars')



ax2 = fig.add_subplot(1, 3, 2)

ax2.plot = sns.barplot(x = 'FullBath', y= 'SalePrice', data = data, order = np.sort(data.FullBath.unique()))

plt.xlabel('FullBath')



ax3 = fig.add_subplot(1, 3, 3)

ax3.plot = sns.barplot(x = 'TotRmsAbvGrd', y= 'SalePrice', data = data, order = np.sort(data.TotRmsAbvGrd.unique()))

plt.xlabel('TotRmsAbvGrd')
## plot high correlated features (cont.)

plt.figure(1)

f, axarr = plt.subplots(2, 3, figsize=(12, 8))

price = data.SalePrice.values

axarr[0, 0].scatter(data.GrLivArea.values, price)

axarr[0, 0].set_title('GrLiveArea')

axarr[0, 1].scatter(data.GarageArea.values, price)

axarr[0, 1].set_title('GarageArea')

axarr[0, 2].scatter(data.TotalBsmtSF.values, price)

axarr[0, 2].set_title('TotalBsmtSF')

axarr[1, 0].scatter(data['1stFlrSF'].values, price)

axarr[1, 0].set_title('1stFlrSF')

axarr[1, 1].scatter(data.YearBuilt.values, price)

axarr[1, 1].set_title('YearBuilt')

axarr[1, 2].scatter(data.YearRemodAdd.values, price)

axarr[1, 2].set_title('YearRemodAdd')

f.text(-0.01, 0.5, 'Sale Price', va='center', rotation='vertical', fontsize = 12)

plt.tight_layout()

plt.show()
## plot several categorical features (cont.)

fig = plt.figure(dpi = 100, figsize = (18, 18))



ax1 = fig.add_subplot(3, 3, 1)

ax1.plot = sns.barplot(x = 'LotShape', y= 'SalePrice', data = data, order = np.sort(data.LotShape.unique()))

plt.xlabel('LotShape')

plt.ylabel('SalePrice')



ax2 = fig.add_subplot(3, 3, 2)

ax2.plot = sns.barplot(x = 'Neighborhood', y= 'SalePrice', data = data, order = np.sort(data.Neighborhood.unique()))

plt.xlabel('Neighborhood')

plt.ylabel('SalePrice')



ax3 = fig.add_subplot(3, 3, 3)

ax3.plot = sns.barplot(x = 'HouseStyle', y= 'SalePrice', data = data, order = np.sort(data.HouseStyle.unique()))

plt.xlabel('HouseStyle')

plt.ylabel('SalePrice')

           

ax4 = fig.add_subplot(3, 3, 4)

ax4.plot = sns.barplot(x = 'ExterQual', y= 'SalePrice', data = data, order = np.sort(data.ExterQual.unique()))

plt.xlabel('ExterQual')

plt.ylabel('SalePrice')



ax5 = fig.add_subplot(3, 3, 5)

ax5.plot = sns.barplot(x = 'ExterCond', y= 'SalePrice', data = data, order = np.sort(data.ExterCond.unique()))

plt.xlabel('ExterCond')

plt.ylabel('SalePrice')



ax6 = fig.add_subplot(3, 3, 6)

ax6.plot = sns.barplot(x = 'Foundation', y= 'SalePrice', data = data, order = np.sort(data.Foundation.unique()))

plt.xlabel('Foundation')

plt.ylabel('SalePrice')

           

ax7 = fig.add_subplot(3, 3, 7)

ax7.plot = sns.barplot(x = 'BsmtQual', y= 'SalePrice', data = data)

plt.xlabel('BsmtQual')

plt.ylabel('SalePrice')



ax8 = fig.add_subplot(3, 3, 8)

ax8.plot = sns.barplot(x = 'BsmtCond', y= 'SalePrice', data = data)

plt.xlabel('BsmtCond')

plt.ylabel('SalePrice')



ax9 = fig.add_subplot(3, 3, 9)

ax9.plot = sns.barplot(x = 'BsmtFinType1', y= 'SalePrice', data = data)

plt.xlabel('BsmtFinType1')

plt.ylabel('SalePrice')
## plot several categorical features (cont.)

fig = plt.figure(dpi = 100, figsize = (18, 18))



ax1 = fig.add_subplot(3, 3, 1)

ax1.plot = sns.barplot(x = 'MSZoning', y= 'SalePrice', data = data, order = np.sort(data.MSZoning.unique()))

plt.xlabel('MSZoning')

plt.ylabel('SalePrice')



ax2 = fig.add_subplot(3, 3, 2)

ax2.plot = sns.barplot(x = 'HeatingQC', y= 'SalePrice', data = data, order = np.sort(data.HeatingQC.unique()))

plt.xlabel('HeatingQC')

plt.ylabel('SalePrice')



ax3 = fig.add_subplot(3, 3, 3)

ax3.plot = sns.barplot(x = 'KitchenQual', y= 'SalePrice', data = data, order = np.sort(data.KitchenQual.unique()))

plt.xlabel('KitchenQual')

plt.ylabel('SalePrice')

           

ax4 = fig.add_subplot(3, 3, 4)

ax4.plot = sns.barplot(x = 'FireplaceQu', y= 'SalePrice', data = data)

plt.xlabel('FireplaceQu')

plt.ylabel('SalePrice')



ax5 = fig.add_subplot(3, 3, 5)

ax5.plot = sns.barplot(x = 'GarageQual', y= 'SalePrice', data = data)

plt.xlabel('GarageQual')

plt.ylabel('SalePrice')



ax6 = fig.add_subplot(3, 3, 6)

ax6.plot = sns.barplot(x = 'GarageCond', y= 'SalePrice', data = data)

plt.xlabel('GarageCond')

plt.ylabel('SalePrice')

           

ax7 = fig.add_subplot(3, 3, 7)

ax7.plot = sns.barplot(x = 'PoolQC', y= 'SalePrice', data = data)

plt.xlabel('PoolQC')

plt.ylabel('SalePrice')



ax8 = fig.add_subplot(3, 3, 8)

ax8.plot = sns.barplot(x = 'Fence', y= 'SalePrice', data = data)

plt.xlabel('Fence')

plt.ylabel('SalePrice')



ax9 = fig.add_subplot(3, 3, 9)

ax9.plot = sns.barplot(x = 'SaleCondition', y= 'SalePrice', data = data, order = np.sort(data.SaleCondition.unique()))

plt.xlabel('SaleCondition')

plt.ylabel('SalePrice')
# 5. transfrom the target variables

## ECDF tricks

from statsmodels.distributions.empirical_distribution import ECDF

from scipy.stats import norm

ecdf = ECDF(data.SalePrice)

new_dis = ecdf(data.SalePrice) * 0.98 + 0.01

nom_dis = norm.ppf(new_dis)



## plot the transformed data

matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"price":data["SalePrice"], "log(price + 1)":np.log1p(data["SalePrice"]), "ECDF(price)":nom_dis})

prices.hist()