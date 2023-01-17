# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
sns.distplot(train['SalePrice'])
print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
var = 'GrLivArea' #Above grade (ground) living area square feet

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF' #Total square feet of basement area

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#box plot overallqual/saleprice

var = 'OverallQual'# Overall material and finish quality

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
corrmat = train.corr()



f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);

corrmat['SalePrice'].sort_values(ascending=False)
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 2.5)

plt.show()
#missing data

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
missing_data[missing_data['Total'] > 1].index
train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)

train = train.drop(train.loc[train['Electrical'].isnull()].index)

train.isnull().sum().max()
test = test.drop((missing_data[missing_data['Total'] > 1]).index,1)

test = test.drop(train.loc[train['Electrical'].isnull()].index)

test.isnull().sum().max()
saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#bivariate analysis saleprice/grlivarea

var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
train.sort_values(by = 'GrLivArea', ascending = False)[:2]

train = train.drop(train[train['Id'] == 1299].index)

train = train.drop(train[train['Id'] == 524].index)
#bivariate analysis saleprice/totalbsmtsf

var = 'TotalBsmtSF'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#histogram and normal probability plot

sns.distplot(train['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['TotalBsmtSF'], plot=plt)

#create column for new variable (one is enough because it's a binary categorical feature)

#if area>0 it gets 1, for area==0 it gets 0

train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)

train['HasBsmt'] = 0 

train.loc[train['TotalBsmtSF']>0,'HasBsmt'] = 1

#transform data

train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])

#histogram and normal probability plot

sns.distplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

test['TotalBsmtSF'].fillna(0, inplace=True) 
#histogram and normal probability plot

sns.distplot(test['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(test['TotalBsmtSF'], plot=plt)

#create column for new variable (one is enough because it's a binary categorical feature)

#if area>0 it gets 1, for area==0 it gets 0

test['HasBsmt'] = pd.Series(len(test['TotalBsmtSF']), index=test.index)

test['HasBsmt'] = 0 

test.loc[test['TotalBsmtSF']>0,'HasBsmt'] = 1

#transform data

test.loc[test['HasBsmt']==1,'TotalBsmtSF'] = np.log(test['TotalBsmtSF'])

#histogram and normal probability plot

sns.distplot(test[test['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(test[test['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
plt.scatter(train['GrLivArea'], train['SalePrice'])
plt.scatter(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], train[train['TotalBsmtSF']>0]['SalePrice'])
from sklearn.impute import SimpleImputer

imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')



for column_name in test.columns[test.isna().any()].tolist():

    imp_most_frequent.fit(test[column_name].values.reshape(-1, 1))

    test[column_name] = imp_most_frequent.transform(test[column_name].values.reshape(-1, 1))
combine = pd.concat([train, test], sort=False)

combine['HasBsmt'] = combine['HasBsmt'].fillna(0)
combine.isnull().sum()
train.shape, test.shape, combine.shape
combine = pd.concat([train, test], sort=False)



combine=combine.drop(['2ndFlrSF',

 '3SsnPorch',

 'BedroomAbvGr',

 'BsmtFinSF1',

 'BsmtFinSF2',

 'BsmtFullBath',

 'BsmtHalfBath',

 'BsmtUnfSF',

 'EnclosedPorch',

 'Fireplaces',

 'Functional',

 'GarageArea',

 'HalfBath',

 'Heating',

 'Id',

 'KitchenAbvGr',

 'LotArea',

 'LowQualFinSF',

 'MSSubClass',

 'MiscVal',

 'MoSold',

 'OpenPorchSF',

 'OverallCond',

 'PoolArea',

 'RoofMatl',

 'ScreenPorch',

 'Utilities',

 'WoodDeckSF',

 'YrSold'], axis=1)
combine.corr()['SalePrice'].sort_values(ascending=False)
combine.info()
combine_full=pd.get_dummies(combine)
y = combine_full['SalePrice'].dropna()

X = combine_full[:train.shape[0]].drop(['SalePrice'], axis=1)

X_test = combine_full[test.shape[0]-2:].drop(['SalePrice'], axis=1)
X.shape,X_test.shape, y.shape
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=3000, 

                                  learning_rate=0.05, 

                                  max_depth=3, 

                                  max_features='sqrt',

                                  min_samples_leaf=15, 

                                  min_samples_split=10, 

                                  loss='huber')

model.fit(X, y)
from sklearn.model_selection import cross_val_score



scores = cross_val_score(model, X, y, cv=5)

print(scores)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predictions = model.predict(X_test)
submissions=pd.DataFrame({"Id": list(range(len(predictions)+2,len(predictions)+len(predictions)+2)), "SalePrice": predictions})

submissions.to_csv("DR.csv", index=False, header=True)