#invite people for the Kaggle party

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#bring in the six packs

df_train = pd.read_csv('../input/train.csv')
#check the decoration

df_train.columns
#descriptive statistics summary

df_train['SalePrice'].describe()


#histogram

sns.distplot(df_train['SalePrice']);
#skewness and kurtosis

print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1) # (Yuqing) axis = 1: concatenate along column

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#box plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
#correlation matrix

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(18, 8.5))  

# plt.subplots() is a function that returns a tuple containing a figure and axes object(s). 

# Thus when using fig, ax = plt.subplots() you unpack this tuple into the variables fig and ax

sns.heatmap(corrmat, vmax=.8, square=True); # (Yuqing) vmax: max correllation. Why set to 0.8 though?
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()



## Yuqing: 9 predictors that has the highest correlations with SalePrice

## My segmentation dissection, see below
# corrmat = df_train.corr(); corrmat  # set above

corrmat.nlargest(10, 'SalePrice')  # Highest 10 (including SalePrice itself, so 9 predictors) variables correlated to the SalePrice, ordered in the leftmost column.

# see the last/rightmost column for coreelation values

## Yuqing: more precisely, should be the abs(corr)

# select column SalePrice (like select in tidyverse):

corrmat.nlargest(10, 'SalePrice')['SalePrice']
corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
#scatterplot

sns.set() # seaborn.set, Set aesthetic parameters in one step.

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], size = 3)

plt.show();
### Yuqing ###

# For categorical data, information from scattorplots is very limited. 

# Contigenvy tables can be much more appropriate and useful



print(df_train['OverallQual'].value_counts())

print(df_train['OverallCond'].value_counts())

pd.crosstab(df_train['OverallQual'], df_train['OverallCond']) 
# correlation vetween qual and cond: heatmap

qual_cond = np.corrcoef(df_train[['OverallQual', 'OverallCond']].values.T);

sns.set(font_scale=1.25)

hm = sns.heatmap(qual_cond, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10})

plt.show()
#missing data

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20) # 
df_train.isnull().sum().sort_values(ascending=False).head(5)
#dealing with missing data

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max() #just checking that there's no missing data missing...
#standardizing data

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#saleprice_scaled = 

StandardScaler(). fit_transform(df_train['SalePrice'][:,np.newaxis])
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

### from scipy.stats import norm

sns.distplot(df_train['SalePrice'], fit=norm); # top figure

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
# sns.distplot(df_train['SalePrice'], fit=norm);



# res = stats.probplot(df_train['SalePrice']) # default plot = None, no plot createdfig = plt.figure()



fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)

#applying log transformation

df_train['SalePrice'] = np.log(df_train['SalePrice'])
#transformed histogram and normal probability plot

sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
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

sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
#scatter plot

plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);
#scatter plot

plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);
#convert categorical variable into dummy

df_train = pd.get_dummies(df_train)