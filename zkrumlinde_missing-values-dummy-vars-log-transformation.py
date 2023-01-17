
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
training = pd.read_csv('../input/train.csv')
testing = pd.read_csv('../input/test.csv')
all_data = pd.concat([training, testing], ignore_index = True, sort = True)
all_data.head()
#Find how much missind data there is.
mis_val_percent = (100 * all_data.isnull().sum() / len(all_data)).sort_values(ascending = False)
print(mis_val_percent[mis_val_percent > 0])
#With such a large percentage missing from some columns, I am going to drop the top 5 columns.
#Be careful, SalePrice is missing so much due to the fact that the test set does not include the SalePrice
all_data = all_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis = 1)
#LotFrontage
#It seems common that lots are all fairly similar in each neighborhood.  
#Fill all the missing data with the mean of the LotFrontage of the neighborhood the house is in

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
#NA for Garage info means there is no garage, replace 'NA' with 'None'
all_data['GarageFinish'] = all_data['GarageFinish'].fillna('None')
all_data['GarageCond'] = all_data['GarageCond'].fillna('None')
all_data['GarageQual'] = all_data['GarageQual'].fillna('None')
all_data['GarageType'] = all_data['GarageType'].fillna('None')
#Similar to garage info, 'NA' for basement columns, represents 'No Basement'
#For Basement Bath info, filling with 0
all_data['BsmtCond'] = all_data['BsmtCond'].fillna('None')
all_data['BsmtExposure'] = all_data['BsmtExposure'].fillna('None')
all_data['BsmtFinType1'] = all_data['BsmtFinType1'].fillna('None')
all_data['BsmtFinType2'] = all_data['BsmtFinType2'].fillna('None')
all_data['BsmtQual'] = all_data['BsmtQual'].fillna('None')
all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna(0)
all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].fillna(0)
#'NA' for Masonry will be filled with 'None' while the area, since none, will be filled with 0.
all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
#MSZoning will be filled with the most common MSZoning in the neighborhood of the missing values.
all_data['MSZoning'].value_counts().sort_values(ascending = False)
all_data[all_data['MSZoning'].isnull()]['Neighborhood']
all_data[all_data['Neighborhood'] == 'IDOTRR']['MSZoning'].value_counts().sort_values(ascending = False)
all_data[all_data['Neighborhood'] == 'Mitchel']['MSZoning'].value_counts().sort_values(ascending = False)
all_data.loc[all_data['Neighborhood'] == 'IDOTRR', 'MSZoning'] = all_data.loc[all_data['Neighborhood'] == 'IDOTRR', 'MSZoning'].fillna('RM')
all_data.loc[all_data['Neighborhood'] == 'Mitchel', 'MSZoning'] = all_data.loc[all_data['Neighborhood'] == 'Mitchel', 'MSZoning'].fillna('RL')
#That took way longer than it should have for 4 missing values, but the lesson learned was great!
#Utilities
all_data['Utilities'].value_counts().sort_values(ascending = False)
#All but one case are 'AllPub', going to stick with the trend
all_data['Utilities'] = all_data['Utilities'].fillna('AllPub')
#Functional, data description states assume normal unless otherwise stated
all_data['Functional'] = all_data['Functional'].fillna('Typ')
#Electrical, the vast majority are 'SBrkr'
all_data['Electrical'].value_counts().sort_values(ascending = False)
all_data['Electrical'] = all_data['Electrical'].fillna('SBrkr')
#Exterior1st and Exterior2nd
#The null values for Exterior1st and Exterior2nd are the same observation
all_data[all_data['Exterior1st'].isnull()][['Exterior2nd', 'Neighborhood']]
all_data[all_data['Neighborhood'] == 'Edwards']['Exterior1st'].value_counts().sort_values(ascending= False)
all_data[all_data['Neighborhood'] == 'Edwards']['Exterior2nd'].value_counts().sort_values(ascending= False)
#Houses in the same neighbor tend to be built pretty similar.
#The null value comes from the Edwards neighborhood where 'Wd Sdng' is the most popular
all_data['Exterior1st'] = all_data['Exterior1st'].fillna('Wd Sdng')
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna('Wd Sdng')
#KitchenQual is only missing one observation as well, fill with the most popular
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
#GarageCars
#Upon looking at the null observation, the GarageArea is also null
#No garage means no cars
all_data[all_data['GarageCars'].isnull()]['GarageArea']
all_data['GarageCars'] = all_data['GarageCars'].fillna(0)
all_data['GarageArea'] = all_data['GarageArea'].fillna(0)
#Taking a look at the remaining missing basement data
all_data[all_data['BsmtFinSF1'].isnull()][['BsmtUnfSF', 'BsmtFinSF2', 'TotalBsmtSF']]
#With all the basement columns null, there is no basement
all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].fillna(0)
all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].fillna(0)
all_data['BsmtFinSF2'] = all_data['BsmtFinSF2'].fillna(0)
all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(0)
#SaleType
all_data['SaleType'].value_counts().sort_values(ascending = False).head(3)
#WD is by far the most popular 'SaleType'
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
#GarageYrBlt is throwing me for a loop, this most likely means that there isn't a garage.
#Since the data is a year, inputting 0 would skew this data very much and could lead to 
#correlations not being accurate
## Filling null values with same year the house was built.
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(all_data['YearBuilt'])
## Double Check that there is no more missing values
## The SalePrice is due to the test set not having that column
mis_val_percent = (100 * all_data.isnull().sum() / len(all_data)).sort_values(ascending = False)
print(mis_val_percent[mis_val_percent > 0])
all_data = pd.get_dummies(all_data)
print('Size Before Dummy Variables (Just training): ', training.shape)
print('Size After Dummy Variables: ', all_data.shape)
training = all_data.loc[all_data['SalePrice'].notnull()]
testing = all_data.loc[all_data['SalePrice'].isnull()]
# A skew of 0 represents a normal distribution.
# 'SalePrice' has a positive skew, shown by the skew > 0.
training['SalePrice'].skew()
#Visually we can see the positive skew by the tail on the right of the histogram
plt.hist(training['SalePrice'])
plt.show()
#Check normality of the dependent variable
#Base on the p-value (less than 0.05), we can assume the data is not normally distributed
scipy.stats.shapiro(training['SalePrice'])
#A transformation should be done on the 'SalePrice' to attempt to bring it closer to noraml
#Some methods for a positive skew are square root, cube root and log transformations.

SalePrice_sqrt = training['SalePrice'] ** (1/2)
SalePrice_cubert = training['SalePrice'] ** (1/3)
SalePrice_log = np.log(training['SalePrice'])

print('The skew of the square root is', SalePrice_sqrt.skew())
print('The skew of the cube root is', SalePrice_cubert.skew())
print('The skew of the log is', SalePrice_log.skew())
#All the transformations increased the normality, 
#however the log transformation is the closest to 0
training['SalePrice(log)'] = SalePrice_log
#The distribution of the log transformation
plt.hist(training['SalePrice(log)'])
plt.show()

