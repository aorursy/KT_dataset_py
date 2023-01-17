# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #Data Visualization 
import seaborn as sns
import warnings
warnings.simplefilter("ignore")
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Now import and sore the train and test datasets in  pandas dataframe

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#print the Sze of data set
print("train dataset size is : {}".format(train.shape))
print("test dataset size is : {}".format(test.shape))
train.head()
# Getting rid of the IDs but keeping the test IDs in a variable
#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' 
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
#statistics summary
train.SalePrice.describe()
#Distribution 
plt.subplots(figsize=(10,7))
sns.distplot(train['SalePrice'],)

corrmat = train.corr()
plt.subplots(figsize=(18,12))
sns.heatmap(train.corr())
k = 5 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cols
cm = np.corrcoef(train[cols].values.T)
plt.subplots(figsize=(10,7))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
train[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea']].corr()

sns.pairplot(data=train[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea']])
train[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea']].describe()
train.select_dtypes(include='object').describe()
#box plot overallqual/saleprice
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=train['OverallQual'], y=train["SalePrice"],)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=train['YearBuilt'], y=train["SalePrice"],)
plt.xticks(rotation=90)
from scipy import stats
from scipy.stats import norm
#Plot the Normal Distribution of sales price
plt.figure(figsize=(10,6))
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot to see common distribution
fig = plt.figure(figsize=(10,6))
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution 
plt.figure(figsize=(10,6))
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure(figsize=(10,6))
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
#data transformation
train['GrLivArea'] = np.log(train['GrLivArea'])
#transformed histogram and normal probability plot
sns.distplot(train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)
#histogram and normal probability plot
sns.distplot(train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['TotalBsmtSF'], plot=plt)



    
