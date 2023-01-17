import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
housetrn = pd.read_csv('../input/train.csv')

housetst = pd.read_csv('../input/test.csv')
# let us check the columsn that are part of the input file

print (housetrn.columns)

# Because the number of columns are large, we can set to display all columns

pd.options.display.max_columns = 999

print (pd.options.display.max_columns)
print (housetrn.head())
# Let us do a scatterplot for the variable selected as HIGH

sns.set_style("whitegrid")

sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(10, 5)

sns.stripplot(x="Neighborhood", y="SalePrice", data=housetrn, jitter=True)

plt.xticks(rotation=45)

plt.show()
sns.set_style('ticks')

sns.factorplot(x="OverallQual", y="SalePrice", col="BldgType", data=housetrn, kind="swarm", col_wrap=3)

fig.set_size_inches(5, 4)

plt.show()
fig, ax = plt.subplots()

fig.set_size_inches(10, 5)

sns.barplot(x="YearBuilt", y="SalePrice", data=housetrn)

plt.xticks(rotation=90)

sns.set_style('ticks')

plt.show()
fig, ax = plt.subplots()

fig.set_size_inches(10, 5)

sns.regplot(x="TotalBsmtSF", y="SalePrice", data=housetrn)

sns.regplot(x="GrLivArea", y="SalePrice", data=housetrn)

plt.show()
sns.factorplot(x="SaleCondition", y="SalePrice", col="Functional", data=housetrn, kind="swarm", col_wrap=3)

plt.show()
# Plotting the Pearson correlation of the different features

corr_matrix = housetrn.corr()

colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(corr_matrix,linewidths=0.1,vmax=0.8, square=True, cmap=colormap, linecolor='white')

plt.show()
# We are now selecting more feature to improve the accuracy of our fit

newhousetrn = housetrn[['Id', 'LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'OverallQual', 'TotalBsmtSF', '1stFlrSF',  'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars','GarageArea', 'Neighborhood', 'YearBuilt', 'YearRemodAdd','Functional', 'SalePrice']]
print (newhousetrn.head())
# There are few missing values that we observe

total = newhousetrn.isnull().sum().sort_values(ascending=False)

percent = (newhousetrn.isnull().sum()/newhousetrn.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print (missing_data)
# We have 2 variables that we should convert from category to numerical

newhousetrn.dtypes
# Let us understand the missing values a little bit before we start filling them

newhousetrn.LotFrontage.hist()

plt.show()

print ('min value is:', newhousetrn.LotFrontage.min())

print ('max value is:', newhousetrn.LotFrontage.max())

print ('mean value is:', newhousetrn.LotFrontage.mean())

print ('standard dev is:', newhousetrn.LotFrontage.std())
# we are going to replace the missing value with a series to rand varaibles distributed with the mean and std

# We are also going to plot to make sure distribution after filling is not affected which is very important

lot_av = newhousetrn.LotFrontage.mean()

lot_sd = newhousetrn.LotFrontage.std()

tot_mislot = newhousetrn.LotFrontage.isnull().sum()

rand_lot= np.random.randint(lot_av - lot_sd, lot_av + lot_sd, size=tot_mislot)

newhousetrn['LotFrontage'][np.isnan(newhousetrn['LotFrontage'])] = rand_lot

newhousetrn['LotFrontage'] = newhousetrn['LotFrontage'].astype(int)

newhousetrn.LotFrontage.hist()

plt.show()
# Next we are going to fill the missing value for MasVnrArea

newhousetrn.MasVnrArea.hist()

plt.show()

print ('min value is:', newhousetrn.MasVnrArea.min())

print ('max value is:', newhousetrn.MasVnrArea.max())

print ('mean value is:', newhousetrn.MasVnrArea.mean())

print ('standard dev is:', newhousetrn.MasVnrArea.std())
# Let us apply the same technique for this too

mva_av = newhousetrn.MasVnrArea.mean()

mva_sd = newhousetrn.MasVnrArea.std()

tot_mismva = newhousetrn.MasVnrArea.isnull().sum()

rand_mva= np.random.randint(mva_av - mva_sd, mva_av + mva_sd, size=tot_mismva)

newhousetrn['MasVnrArea'][np.isnan(newhousetrn['MasVnrArea'])] = rand_mva

newhousetrn['MasVnrArea'] = newhousetrn['MasVnrArea'].astype(int)

newhousetrn.MasVnrArea.hist()

plt.show()
# the total number of missing value in year is about 0.05%, apply the same for this too GarageYrBlt

gyr_av = newhousetrn.GarageYrBlt.mean()

gyr_sd = newhousetrn.GarageYrBlt.std()

tot_misgyr = newhousetrn.GarageYrBlt.isnull().sum()

rand_gyr= np.random.randint(gyr_av - gyr_sd, gyr_av + gyr_sd, size=tot_misgyr)

newhousetrn['GarageYrBlt'][np.isnan(newhousetrn['GarageYrBlt'])] = rand_gyr

newhousetrn['GarageYrBlt'] = newhousetrn['GarageYrBlt'].astype(int)

newhousetrn.GarageYrBlt.hist()

plt.show()
# Let us test out the missing values after the treatment

total = newhousetrn.isnull().sum().sort_values(ascending=False)

percent = (newhousetrn.isnull().sum()/newhousetrn.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print (missing_data)
# Encoding the Categorical variables

cols_to_transform = newhousetrn[['Id', 'Neighborhood', 'Functional']]

newcols = pd.get_dummies(cols_to_transform)
newcols.head()
del newhousetrn['Neighborhood']

del newhousetrn['Functional']
print (newcols.shape)

print (newhousetrn.shape)
fhoustrn = newhousetrn.merge(newcols, how='inner', on='Id' )

fhoustrn.head()
fhoustrn.shape
#Repeating the same for the test data

newhousetst = housetst[['Id', 'LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'OverallQual', 'TotalBsmtSF', '1stFlrSF',  'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars','GarageArea', 'Neighborhood', 'YearBuilt', 'YearRemodAdd','Functional']]
#checking for missing values in the test data

total = newhousetst.isnull().sum().sort_values(ascending=False)

percent = (newhousetst.isnull().sum()/newhousetst.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print (missing_data)
# since the missing value is very less we can do a replace with mean and mode

# Since the column Functional is categorical we can replace it with most frequent value

newhousetst['Functional'] = newhousetst['Functional'].fillna('Typ')

# We can replace both GarageArea and TotalBsmtSF as a mean value

men1 = newhousetst.GarageArea.mean()

newhousetst['GarageArea'] =  newhousetst['GarageArea'].fillna(men1)

men2 = newhousetst.TotalBsmtSF.mean()

newhousetst['TotalBsmtSF'] =  newhousetst['TotalBsmtSF'].fillna(men2)

men3 = newhousetst.BsmtFinSF1.mean()

newhousetst['BsmtFinSF1'] =  newhousetst['BsmtFinSF1'].fillna(men3)

men4 = newhousetst.GarageCars.mean()

newhousetst['GarageCars'] =  newhousetst['GarageCars'].fillna(men4)
# Replace the LotFrontage in the test

lot_av = newhousetst.LotFrontage.mean()

lot_sd = newhousetst.LotFrontage.std()

tot_mislot = newhousetst.LotFrontage.isnull().sum()

rand_lot= np.random.randint(lot_av - lot_sd, lot_av + lot_sd, size=tot_mislot)

newhousetst['LotFrontage'][np.isnan(newhousetst['LotFrontage'])] = rand_lot

newhousetst['LotFrontage'] = newhousetst['LotFrontage'].astype(int)

# Replace the GarageYrBlt in the test

gyr_av = newhousetst.GarageYrBlt.mean()

gyr_sd = newhousetst.GarageYrBlt.std()

tot_misgyr = newhousetst.GarageYrBlt.isnull().sum()

rand_gyr= np.random.randint(gyr_av - gyr_sd, gyr_av + gyr_sd, size=tot_misgyr)

newhousetst['GarageYrBlt'][np.isnan(newhousetst['GarageYrBlt'])] = rand_gyr

newhousetst['GarageYrBlt'] = newhousetst['GarageYrBlt'].astype(int)

# Replace the MasVnrArea in the test

mva_av = newhousetst.MasVnrArea.mean()

mva_sd = newhousetst.MasVnrArea.std()

tot_mismva = newhousetst.MasVnrArea.isnull().sum()

rand_mva= np.random.randint(mva_av - mva_sd, mva_av + mva_sd, size=tot_mismva)

newhousetst['MasVnrArea'][np.isnan(newhousetst['MasVnrArea'])] = rand_mva

newhousetst['MasVnrArea'] = newhousetst['MasVnrArea'].astype(int)
#checking for missing values after replacing them

total = newhousetst.isnull().sum().sort_values(ascending=False)

percent = (newhousetst.isnull().sum()/newhousetst.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print (missing_data)
#Encodinf for Categorical variables

cols1_to_transform = newhousetst[['Id', 'Neighborhood', 'Functional']]

newcols1 = pd.get_dummies(cols1_to_transform)
del newhousetst['Neighborhood']

del newhousetst['Functional']
print (newcols1.shape)

print (newhousetst.shape)
fhoustst = newhousetst.merge(newcols1, how='inner', on='Id' )
fhoustst.shape
fhoustst.head()
# Creating the y for Training the data

y = fhoustrn.SalePrice
y.head()
del fhoustrn['SalePrice']
X_train = fhoustrn[:]

X_test = fhoustst[:]
X_train.head()
print (X_train.shape)

print (X_test.shape)
# Standardizing for the unscaled 15 variables in X_train

from sklearn import preprocessing

std_scale1 = preprocessing.StandardScaler().fit(X_train[['OverallQual']])

X_train[['OverallQual']] = std_scale1.transform(X_train[['OverallQual']])

std_scale2 = preprocessing.StandardScaler().fit(X_train[['TotalBsmtSF']])

X_train[['TotalBsmtSF']] = std_scale2.transform(X_train[['TotalBsmtSF']])

std_scale3 = preprocessing.StandardScaler().fit(X_train[['1stFlrSF']])

X_train[['1stFlrSF']] = std_scale3.transform(X_train[['1stFlrSF']])

std_scale4 = preprocessing.StandardScaler().fit(X_train[['GrLivArea']])

X_train[['GrLivArea']] = std_scale4.transform(X_train[['GrLivArea']])

std_scale5 = preprocessing.StandardScaler().fit(X_train[['FullBath']])

X_train[['FullBath']] = std_scale5.transform(X_train[['FullBath']])

std_scale6 = preprocessing.StandardScaler().fit(X_train[['GarageArea']])

X_train[['GarageArea']] = std_scale6.transform(X_train[['GarageArea']])

std_scale7 = preprocessing.StandardScaler().fit(X_train[['YearBuilt']])

X_train[['YearBuilt']] = std_scale7.transform(X_train[['YearBuilt']])

std_scale8 = preprocessing.StandardScaler().fit(X_train[['LotFrontage']])

X_train[['LotFrontage']] = std_scale8.transform(X_train[['LotFrontage']])

std_scale9 = preprocessing.StandardScaler().fit(X_train[['MasVnrArea']])

X_train[['MasVnrArea']] = std_scale9.transform(X_train[['MasVnrArea']])

std_scale10 = preprocessing.StandardScaler().fit(X_train[['BsmtFinSF1']])

X_train[['BsmtFinSF1']] = std_scale10.transform(X_train[['BsmtFinSF1']])

std_scale11 = preprocessing.StandardScaler().fit(X_train[['TotRmsAbvGrd']])

X_train[['TotRmsAbvGrd']] = std_scale11.transform(X_train[['TotRmsAbvGrd']])

std_scale12 = preprocessing.StandardScaler().fit(X_train[['Fireplaces']])

X_train[['TotRmsAbvGrd']] = std_scale12.transform(X_train[['Fireplaces']])

std_scale13 = preprocessing.StandardScaler().fit(X_train[['GarageYrBlt']])

X_train[['GarageYrBlt']] = std_scale13.transform(X_train[['GarageYrBlt']])

std_scale14 = preprocessing.StandardScaler().fit(X_train[['GarageCars']])

X_train[['GarageCars']] = std_scale14.transform(X_train[['GarageCars']])

std_scale15 = preprocessing.StandardScaler().fit(X_train[['YearRemodAdd']])

X_train[['YearRemodAdd']] = std_scale15.transform(X_train[['YearRemodAdd']])
X_train.head()
# Let us apply the same technique of rescaling for the Test data set too

std_scale16 = preprocessing.StandardScaler().fit(X_test[['OverallQual']])

X_test[['OverallQual']] = std_scale16.transform(X_test[['OverallQual']])

std_scale17 = preprocessing.StandardScaler().fit(X_test[['TotalBsmtSF']])

X_test[['TotalBsmtSF']] = std_scale17.transform(X_test[['TotalBsmtSF']])

std_scale18 = preprocessing.StandardScaler().fit(X_test[['1stFlrSF']])

X_test[['1stFlrSF']] = std_scale18.transform(X_test[['1stFlrSF']])

std_scale19 = preprocessing.StandardScaler().fit(X_test[['GrLivArea']])

X_test[['GrLivArea']] = std_scale19.transform(X_test[['GrLivArea']])

std_scale20 = preprocessing.StandardScaler().fit(X_test[['FullBath']])

X_test[['FullBath']] = std_scale20.transform(X_test[['FullBath']])

std_scale21 = preprocessing.StandardScaler().fit(X_test[['GarageArea']])

X_test[['GarageArea']] = std_scale21.transform(X_test[['GarageArea']])

std_scale22 = preprocessing.StandardScaler().fit(X_test[['YearBuilt']])

X_test[['YearBuilt']] = std_scale22.transform(X_test[['YearBuilt']])

std_scale23 = preprocessing.StandardScaler().fit(X_test[['LotFrontage']])

X_test[['LotFrontage']] = std_scale23.transform(X_test[['LotFrontage']])

std_scale24 = preprocessing.StandardScaler().fit(X_test[['MasVnrArea']])

X_test[['MasVnrArea']] = std_scale24.transform(X_test[['MasVnrArea']])

std_scale25 = preprocessing.StandardScaler().fit(X_test[['BsmtFinSF1']])

X_test[['BsmtFinSF1']] = std_scale25.transform(X_test[['BsmtFinSF1']])

std_scale26 = preprocessing.StandardScaler().fit(X_test[['TotRmsAbvGrd']])

X_test[['TotRmsAbvGrd']] = std_scale26.transform(X_test[['TotRmsAbvGrd']])

std_scale27 = preprocessing.StandardScaler().fit(X_test[['Fireplaces']])

X_test[['TotRmsAbvGrd']] = std_scale27.transform(X_test[['Fireplaces']])

std_scale28 = preprocessing.StandardScaler().fit(X_test[['GarageYrBlt']])

X_test[['GarageYrBlt']] = std_scale28.transform(X_test[['GarageYrBlt']])

std_scale29 = preprocessing.StandardScaler().fit(X_test[['GarageCars']])

X_test[['GarageCars']] = std_scale29.transform(X_test[['GarageCars']])

std_scale30 = preprocessing.StandardScaler().fit(X_test[['YearRemodAdd']])

X_test[['YearRemodAdd']] = std_scale30.transform(X_test[['YearRemodAdd']])
X_test.head()
from sklearn.linear_model import Ridge, Lasso, ElasticNet

from sklearn.model_selection import GridSearchCV

from scipy.stats import uniform as sp_rand

from sklearn.model_selection import RandomizedSearchCV
# Performing Grid Search with specific alpha values for Ridge

alphas = np.array([1,0.1,0.01,0.001,0.0001])

model = Ridge()

grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))

grid.fit(X_train, y)

print(grid)

# summarize the results of the grid search

print(grid.best_score_)

print(grid.best_estimator_.alpha)
# Performing Grid Search with range of alpha values (between 0 to 1) for Ridge

param_grid = {'alpha': sp_rand()}

# create and fit a ridge regression model, testing random alpha values

model = Ridge()

rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)

rsearch.fit(X_train, y)

print(rsearch)

# summarize the results of the random parameter search

print(rsearch.best_score_)

print(rsearch.best_estimator_.alpha)
# Performing Grid Search with specific of alpha values for Lasso

alphas = np.array([1,0.1,0.01,0.001,0.0001])

model = Lasso()

grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))

grid.fit(X_train, y)

print(grid)

# summarize the results of the grid search

print(grid.best_score_)

print(grid.best_estimator_.alpha)
# Performing Grid Search with range of alpha values (between 0 to 1) for Lasso

param_grid = {'alpha': sp_rand()}

# create and fit a ridge regression model, testing random alpha values

model = Lasso()

rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)

rsearch.fit(X_train, y)

print(rsearch)

# summarize the results of the random parameter search

print(rsearch.best_score_)

print(rsearch.best_estimator_.alpha)
# Performing Grid Search specific alpha values for ElasticNet

alphas = np.array([1,0.1,0.01,0.001,0.0001])

model = ElasticNet()

grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))

grid.fit(X_train, y)

print(grid)

# summarize the results of the grid search

print(grid.best_score_)

print(grid.best_estimator_.alpha)
# Performing Grid Search with range of alpha & l1_ration values (between 0 to 1) for ElasticNet

param_grid = {'alpha': sp_rand(), 'l1_ratio': sp_rand()}

# create and fit a ridge regression model, testing random alpha values

model = ElasticNet()

rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)

rsearch.fit(X_train, y)

print(rsearch)

# summarize the results of the random parameter search

print(rsearch.best_score_)

print(rsearch.best_estimator_.alpha)

print(rsearch.best_estimator_.l1_ratio)
# Applying the searched values of Alpha and l1_ratio

model1 = Ridge(alpha=0.923990115671)

model2 = Lasso(alpha=0.968154257543)

model3 = ElasticNet(alpha=0.001, l1_ratio=0.99411629749)

model1.fit(X_train, y)

model2.fit(X_train, y)

model3.fit(X_train, y)

pred1 = model1.predict(X_test)

pred2 = model2.predict(X_test)

pred3 = model3.predict(X_test)
finalpred = 0.33*pred1 + 0.33*pred2 + 0.34*pred3

combsolution = pd.DataFrame({"Id":X_test.Id, "SalePrice":finalpred})

combsolution.to_csv("comb_sol3.csv", index = False)