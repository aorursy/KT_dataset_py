# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df.head()
df.describe()
df.columns
print('Some statistics of the house price ')

print(df['SalePrice'].describe())

print()

print('house price median')

print(df['SalePrice'].median())
fig,ax = plt.subplots(figsize=(8,6))

sns.distplot(df['SalePrice'],kde=False,ax=ax)

plt.title('Sale Price Distribution')

plt.xlabel('Sale Price')

plt.ylabel('Freq')
df.dtypes
fig,ax = plt.subplots(figsize=(8,6))

correlation = df.select_dtypes(include=['float64','int64']).iloc[:,1:].corr()

sns.heatmap(correlation,ax=ax,vmax=1,square=True)

plt.title('Numeric Feature heatmap')
corr_dict = correlation['SalePrice'].to_dict()

del correlation['SalePrice']

## descending 

## correlation with sale price

for key,val in sorted(corr_dict.items(),key=lambda x:-abs(x[1])):

    print('{0} \t : {1}' .format(key,val))
fig,ax = plt.subplots(figsize=(8,6))

sns.regplot(x = 'OverallQual', y='SalePrice' , data=df,ax=ax)
fig,ax = plt.subplots(3,2,figsize=(10,9))

sale_price = df['SalePrice'].values





ax[0,0].scatter(df['GrLivArea'],sale_price)

ax[0,0].set_title('GriLivArea vs SalePrice')

ax[0,1].scatter(df['TotRmsAbvGrd'],sale_price)

ax[0,1].set_title('TotRmsAbvGrd vs SalePrice')

ax[1,0].scatter(df['GarageArea'],sale_price)

ax[1,0].set_title('GarageArea vs SalePrice')

ax[1,1].scatter(df['TotalBsmtSF'],sale_price)

ax[1,1].set_title('TotalBsmtSF vs SalePrice')

ax[2,0].scatter(df['1stFlrSF'],sale_price)

ax[2,0].set_title('1stFlrSF vs SalePrice')

ax[2,1].scatter(df['MasVnrArea'],sale_price)

ax[2,1].set_title('MasVnrArea vs SalePrice')

plt.tight_layout()





fig = plt.figure(2,figsize=(10,6))

plt.subplot(211)

plt.scatter(df['YearBuilt'].values,sale_price)

plt.title('YearBuilt vs SalePrice')

plt.subplot(212)

plt.scatter(df['GarageYrBlt'].values,sale_price)

plt.title('GarageYrBlt vs SalePrice')

plt.tight_layout()
print(df.select_dtypes(include=['object']).columns)
fig,ax = plt.subplots(figsize=(8,6))

sns.boxplot(x = 'Neighborhood', y = 'SalePrice',  data = df,ax=ax)

plt.title('neighborhood vs SalePrice')

ticks = plt.setp(ax.get_xticklabels(),rotation=90)
fig,ax = plt.subplots(figsize=(8,6))

sns.boxplot(data=df,x='Functional',y='SalePrice',ax=ax)

plt.title('Functional vs SalePrice')
fig = plt.figure(2,figsize=(8,6))

plt.subplot(211)

sns.boxplot(data=df,x='SaleType',y='SalePrice')

plt.title('SaleType vs SalePrice')



plt.subplot(212)

sns.boxplot(data=df,x='SaleCondition',y='SalePrice')

plt.title('SaleCondition vs SalePrice')



plt.tight_layout()
g = sns.FacetGrid(df,col='YrSold',col_wrap=3)

g.map(sns.boxplot,'MoSold','SalePrice',palette='Set2',order=range(1,13)).set(ylim=(0,500000))

plt.tight_layout()
fig = plt.figure(2,figsize=(8,8))

plt.subplot(211)

sns.boxplot(data=df,x='HouseStyle',y='SalePrice')

plt.title('HouseStyle vs SalePrice')



plt.subplot(212)

sns.boxplot(data=df,x='BldgType',y='SalePrice')

plt.title('BldgType vs SalePrice')



plt.tight_layout()
fig,ax = plt.subplots(2,2,figsize=(8,6))



sns.boxplot(data=df,x='BsmtQual',y='SalePrice',ax=ax[0,0])

ax[0,0].set_title('BsmtQual vs SalePrice')



sns.boxplot(data=df,x='BsmtCond',y='SalePrice',ax=ax[0,1])

ax[0,1].set_title('BsmtCond vs SalePrice')



sns.boxplot(data=df,x='BsmtExposure',y='SalePrice',ax=ax[1,0])

ax[1,0].set_title('BsmtExposure vs SalePrice')



sns.boxplot(data=df,x='BsmtFinType1',y='SalePrice',ax=ax[1,1])

ax[1,1].set_title('BsmtFinType1 vs SalePrice')



plt.tight_layout()
sns.factorplot(data=df,x='FireplaceQu',y='SalePrice',

               order=['Ex','Gd','TA','Fa','Po'],size=4.5,

              aspect=1.35,estimator=np.median)

plt.title('FireplaceQu vs SalePrice')
pd.crosstab(df['Fireplaces'],df['FireplaceQu'])
g= sns.FacetGrid(data=df,col='FireplaceQu',col_wrap=3,

             col_order=['Ex', 'Gd', 'TA', 'Fa', 'Po'])



g.map(sns.boxplot,'Fireplaces','SalePrice',order=[1,2,3],palette='Set2')
sns.factorplot(data=df,x='HeatingQC',y='SalePrice',

               hue='CentralAir',size=4,aspect=1.35,

               estimator=np.mean)

plt.title('HeatingQC vs SalePrce')
sns.boxplot(data=df,x='Electrical',y='SalePrice')

plt.title('Electrical vs SalePrice')
sns.factorplot(data=df,x='KitchenQual',y='SalePrice',

               size=4.5,aspect=1.35,

               order=['Ex','Gd','TA','Fa'])

plt.title('KitchenQual vs SalePrice')
fig,ax = plt.subplots(1,2,figsize=(8,6))



sns.boxplot(data=df,x='Street',y='SalePrice',ax=ax[0])

ax[0].set_title('Street vs SalePrice')

sns.boxplot(data=df,x='Alley',y='SalePrice',ax=ax[1])

ax[1].set_title('Alley vs SalePrice')