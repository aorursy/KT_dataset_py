import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, kruskal
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

pd.set_option('display.max_rows', 81)
pd.set_option('display.max_columns', None)

sns.set(style='whitegrid', context='notebook', palette='deep')

from matplotlib.ticker import StrMethodFormatter

def plot_pct_cat_variables(df, col, rotate_xticks=False, sort=True):
    tmp = df.groupby(col)[col].count()
    tmp2 = pd.DataFrame({col: tmp.index,
                    'Perct': tmp})

    tmp2['Perct'] = tmp2['Perct'] / df.shape[0] * 100
    
    if (sort):
        tmp2 = tmp2.sort_values(by=['Perct'], ascending=False)

    plt.figure(figsize=[15,9])

    ax = sns.barplot(x=col, y='Perct', data=tmp2, order=tmp2.index)
    
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x}%'))
    
    if (rotate_xticks):
        plt.xticks(rotation=45)
        
    plt.show()
    
def get_feature_correlation(corr_matrix, feature, threshold=0.3):
    mask = corr_matrix[feature].abs() > 0.3
    return corr_matrix.loc[mask, feature].sort_values(ascending=False)
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
corr = train.corr()
num_train_observations = float(train.shape[0])
num_test_observations = float(test.shape[0])

# Replace NaN values in these columns with "NA" since this value indicates that the house
# has none of this feature
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Fence',
            'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC',
            'MiscFeature', 'Alley'):
    train[col] = train[col].fillna("NA")
    test[col] = test[col].fillna("NA")
tmp = train.isnull().sum()
tmp = tmp[tmp > 0].sort_values(ascending=False)

missing_train_count = pd.DataFrame(data={
                        'Total missing values': tmp,
                        'Perct. of missing values': tmp / num_train_observations * 100
                       })

display(missing_train_count)

print('There are %d features with missing values in training set' % (missing_train_count.shape[0]))
tmp = test.isnull().sum()
tmp = tmp[tmp > 0].sort_values(ascending=False)

missing_test_count = pd.DataFrame(data={
                        'Total missing values': tmp,
                        'Perct. of missing values': tmp / num_test_observations * 100
                       })

display(missing_test_count)

print('There are %d features with missing values in test set' % (missing_test_count.shape[0]))
print('Feature with missing values in training set but not in test set ' + str(set(missing_train_count.index) - set(missing_test_count.index)))
print('Feature with missing values in test set but not in traing set ' + str(set(missing_test_count.index) - set(missing_train_count.index)))
train['LotFrontage'].fillna(train['LotFrontage'].median(), inplace=True)
test['LotFrontage'].fillna(test['LotFrontage'].median(), inplace=True)
print("Top features correlated with GarageYrBlt:")
print(get_feature_correlation(corr, 'GarageYrBlt'))
sns.jointplot('GarageYrBlt', 'YearBuilt', data=train)

plt.show()
train.loc[train['GarageYrBlt'].isnull(), 'GarageYrBlt'] = train[train['GarageYrBlt'].isnull()]['YearBuilt']
test.loc[test['GarageYrBlt'].isnull(), 'GarageYrBlt'] = test[test['GarageYrBlt'].isnull()]['YearBuilt']
mask_train = ((train['MasVnrArea'] == 0) & (train['MasVnrType'] != 'None')) | \
    ((train['MasVnrArea'] != 0) & (train['MasVnrType'] == 'None'))
mask_test = ((test['MasVnrArea'] == 0) & (test['MasVnrType'] != 'None')) | \
    ((test['MasVnrArea'] != 0) & (test['MasVnrType'] == 'None'))
    
print("Number of rows with mismatch MasVnrArea and MasVnrType in the training set: {}".format(
    train[mask_train].shape[0]))
print("Number of rows with mismatch MasVnrArea and MasVnrType in the test set: {}".format(
    test[mask_test].shape[0]))
train.loc[mask_train, ['MasVnrArea', 'MasVnrType']] = [0, 'None']
test.loc[mask_test, ['MasVnrArea', 'MasVnrType']] = [0, 'None']
sns.distplot(train['MasVnrArea'].dropna())

plt.show()
corr = train.corr()
print("Top features correlated with MasVnrArea:")
print(get_feature_correlation(corr, 'MasVnrArea'))
sns.jointplot("MasVnrArea", "SalePrice", data=train)

plt.show()
print("Rows with missing MasVnrArea and MasVnrType in the training set")
display(train.loc[train['MasVnrArea'].isnull(), ['MasVnrArea', 'MasVnrType']])

print("Rows with missing MasVnrArea and MasVnrType in the test set")
display(test.loc[test['MasVnrArea'].isnull(), ['MasVnrArea', 'MasVnrType']])
train['MasVnrType'].fillna('None', inplace=True)
train['MasVnrArea'].fillna(train['MasVnrArea'].median(), inplace=True)

test.loc[test['MasVnrArea'].isnull(), 'MasVnrType'] = 'None'
test['MasVnrArea'].fillna(test['MasVnrArea'].median(), inplace=True)
print("The summary statistics for houses with a masonry veneer in the test set:")
print(test[test['MasVnrType'] != 'None']['MasVnrType'].describe())

test['MasVnrType'].fillna('BrkFace', inplace=True)
print("The summary statistics for electrical systems in the training set:")
print(train['Electrical'].describe())

train['Electrical'].fillna('Sbrkr', inplace=True)
# Count the number of different MSZoning types in each neighborhood
s = test.groupby(['MSZoning', 'Neighborhood'])['MSZoning'].count()
d = s.unstack(level=0)
d.fillna(0, inplace=True)

plt.figure(figsize=[10,15])

sns.heatmap(d, annot=True)

plt.show()
print("The neighborhoods of houses with missing MSZoning in the test set:")
print(test.loc[test['MSZoning'].isnull(), 'Neighborhood'])
test.loc[test['MSZoning'].isnull() & (test['Neighborhood'] == 'IDOTRR'), 'MSZoning'] = 'RM'
test.loc[test['MSZoning'].isnull() & (test['Neighborhood'] == 'Mitchel'), 'MSZoning'] = 'RL'
plot_pct_cat_variables(test, 'Functional')
test['Functional'].fillna("Typ", inplace=True)
train_count = (train['TotalBsmtSF'] == (train['BsmtUnfSF'] + train['BsmtFinSF1'] + train['BsmtFinSF2'])).sum()
# Drop rows with missing values in test set
tmp = test.dropna(subset=['BsmtUnfSF', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF'])
test_count = (tmp['TotalBsmtSF'] == (tmp['BsmtUnfSF'] + tmp['BsmtFinSF1'] + tmp['BsmtFinSF2'])).sum()

print("{} out of {} houses in the training set have the total basement area equals to the sum of the unfinshed area and the finished area".format(train_count, train.shape[0]))
print("{} out of {} houses in the test set (after dropping rows with null values) have the total basement area equals to the sum of the unfinshed area and the finished area".format(test_count, tmp.shape[0]))
print("The house that has missing basement values:")
print(test.loc[test['BsmtFinSF2'].isnull(), ['TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1', 'BsmtFinSF2',
                                        'BsmtFullBath', 'BsmtHalfBath']])
corr = test.corr()
print("Features that are at least somewhat correlated with TotalBsmtSF:")
print(get_feature_correlation(corr, 'TotalBsmtSF'))
sns.jointplot('1stFlrSF', 'TotalBsmtSF', data=test, kind='reg')

plt.show()
# Combine the data from both training and test sets
# Removed datapoints where TotalBsmtSF = 0. Not sure if this is the right choice
# Also removed the datapoint where TotalBsmtSF is null
combined = train.loc[train['TotalBsmtSF'] > 0, ['TotalBsmtSF', '1stFlrSF', 'YearBuilt']]\
                .append(test.loc[(test['TotalBsmtSF'] > 0) & (test['TotalBsmtSF'].notnull()),
                                 ['TotalBsmtSF', '1stFlrSF', 'YearBuilt']])
combined_x = combined.as_matrix(['1stFlrSF', 'YearBuilt'])
combined_y = combined.as_matrix(['TotalBsmtSF'])

# Split the combined data into train and test split
x_train, x_test, y_train, y_test = train_test_split(combined_x, combined_y, test_size=0.3,
                                                    random_state=10)

# Fit and evaluate the model
reg = LinearRegression()
reg.fit(x_train, y_train)
r2_score = reg.score(x_test, y_test)
print("The R2 score is {:.2f}".format(r2_score))

# Re-train the model using all of the data
reg.fit(combined_x, combined_y)

# Predict the null value
null_x = test.loc[test['TotalBsmtSF'].isnull(), ['1stFlrSF', 'YearBuilt']].iloc[0]
null_y = reg.predict(np.array(null_x).reshape(1, -1))[0][0]
print("The model imputed the basement area to be {:.2f} sq ft".format(null_y))

# Fill in the missing value
test['TotalBsmtSF'].fillna(null_y, inplace=True)
corr = test.corr()
print("Features that are at least somewhat correlated with BsmtFinSF1:")
print(get_feature_correlation(corr, 'BsmtFinSF1'))
sns.jointplot('TotalBsmtSF', 'BsmtFinSF1', data=train, stat_func=spearmanr, kind='reg')

plt.show()
test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].median(), inplace=True)
corr = test.corr()
print("Features that are at least somewhat correlated with BsmtUnfSF:")
print(get_feature_correlation(corr, 'BsmtUnfSF'))
# Combine data from both training set and test set
combined = train.loc[train['TotalBsmtSF'] > 0, ['TotalBsmtSF', 'BsmtFinSF1', 'BsmtUnfSF']]\
                .append(test.loc[(test['TotalBsmtSF'] > 0) & (test['BsmtUnfSF'].notnull()),
                                 ['TotalBsmtSF', 'BsmtFinSF1', 'BsmtUnfSF']])
combined_x = combined.as_matrix(['BsmtFinSF1', 'TotalBsmtSF'])
combined_y = combined.as_matrix(['BsmtUnfSF'])

# Split the combined data into train and test split
x_train, x_test, y_train, y_test = train_test_split(combined_x, combined_y, test_size=0.3,
                                                    random_state=10)

# Fit and evaluate the model
reg = LinearRegression()
reg.fit(x_train, y_train)
r2_score = reg.score(x_test, y_test)
print("The R2 score is {:.2f}".format(r2_score))

# Re-train the model using all of the data
reg.fit(combined_x, combined_y)

# Predict the null value
null_x = test.loc[test['BsmtUnfSF'].isnull(), ['BsmtFinSF1', 'TotalBsmtSF']].iloc[0]
null_y = reg.predict(np.array(null_x).reshape(1, -1))[0][0]
print("The model imputed the missing basement unfinished area to be {:.2f} sq ft".format(null_y))

# Fill in the missing value
test['BsmtUnfSF'].fillna(null_y, inplace=True)
row = test.loc[test['BsmtFinSF2'].isnull(), ['TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1']]
test['BsmtFinSF2'].fillna(row['TotalBsmtSF'] - row['BsmtUnfSF'] - row['BsmtFinSF1'], inplace=True)
corr = test.corr()
print("Features that are at least somewhat correlated with BsmtFullBath:")
print(get_feature_correlation(corr, 'BsmtFullBath'))
sns.jointplot('BsmtFullBath', 'BsmtFinSF1', data=test)
# Combine data from both training set and test set
combined = train.loc[train['TotalBsmtSF'] > 0, ['TotalBsmtSF', 'BsmtFinSF1', 'BsmtFullBath']]\
                .append(test.loc[(test['TotalBsmtSF'] > 0) & (test['BsmtFullBath'].notnull()),
                                 ['TotalBsmtSF', 'BsmtFinSF1', 'BsmtFullBath']])
combined_x = combined.as_matrix(['BsmtFinSF1'])
combined_y = combined.as_matrix(['BsmtFullBath'])

# Train a multi-class logistic regression model to fill in the missing values
lg = LogisticRegression(random_state=10, solver='lbfgs', multi_class='multinomial')
lg.fit(combined_x, combined_y)

# Predict the null value
null_x = test.loc[test['BsmtFullBath'].isnull(), ['BsmtFinSF1']]
null_y = lg.predict(np.array(null_x))

test.loc[test['BsmtFullBath'].isnull(), 'BsmtFullBath'] = null_y
sns.countplot(test['BsmtHalfBath'])
test['BsmtHalfBath'].fillna(0, inplace=True)
plot_pct_cat_variables(train, 'Utilities')
plot_pct_cat_variables(test, 'Utilities')
test['Utilities'].fillna('AllPub', inplace=True)
plot_pct_cat_variables(test, 'SaleType')
test['SaleType'].fillna("WD", inplace=True)
test.loc[test['GarageArea'].isnull(), ['GarageArea', 'GarageCars', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']]
mask = ((test['GarageType'] != 'NA') | (test['GarageArea'] > 0) | (test['GarageCars'] > 0)) \
        & ((test['GarageFinish'] == 'NA') | (test['GarageQual'] == 'NA') | (test['GarageCond'] == 'NA'))
test.loc[mask, ['GarageArea', 'GarageCars', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']]
mask = ((train['GarageType'] != 'NA') | (train['GarageArea'] > 0) | (train['GarageCars'] > 0)) \
    & ((train['GarageFinish'] == 'NA') | (train['GarageQual'] == 'NA') | (train['GarageCond'] == 'NA'))
train.loc[mask, ['GarageArea', 'GarageCars', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']]
test.loc[test['GarageArea'].isnull(), 'GarageType'] = 'NA'
test['GarageArea'].fillna(0, inplace=True)
test['GarageCars'].fillna(0, inplace=True)
# Combine both datasets
combined = train.append(test)
combined = combined.reset_index()

mask = (combined['GarageType'] == 'Detchd') & (combined['GarageArea'] == 360) & (combined['GarageCars'] == 1) & (combined['GarageFinish'] == 'NA')
neighborhoods = list(combined.loc[mask, 'Neighborhood'])
print("The neighborhoodwhere the house is in is: {}".format(neighborhoods))
oldtown = combined.loc[combined['Neighborhood'] == 'OldTown', ['GarageArea', 'GarageCars', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']]
plt.figure(figsize=[15,10])
sns.countplot(x='GarageType', hue='GarageFinish', hue_order=['Fin', 'Rfn', 'Unf', 'NA'], data=oldtown)
plt.figure(figsize=[15,10])
sns.countplot(x='GarageType', hue='GarageQual', hue_order=['Ex', 'Gd', 'NA', 'Fa', 'Po', 'TA'], data=oldtown)
plt.figure(figsize=[15,10])
sns.countplot(x='GarageType', hue='GarageQual', hue_order=['Ex', 'Gd', 'NA', 'Fa', 'Po', 'TA'], data=oldtown)
test.loc[666, 'GarageFinish'] = 'Unf'
test.loc[666, 'GarageQual'] = 'TA'
test.loc[666, 'GarageCond'] = 'TA'
test.loc[test['KitchenQual'].isnull(), ['YearRemodAdd', 'KitchenQual', 'OverallQual']]
tmp = test.loc[(test['YearRemodAdd'] == 1950) & (test['OverallQual'] == 5), ['KitchenQual']]
plot_pct_cat_variables(tmp, 'KitchenQual')
test['KitchenQual'].fillna('TA', inplace=True)
test.loc[test['Exterior1st'].isnull(), ['YearBuilt', 'YearRemodAdd', 'Neighborhood', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond']]
edwards = test[test['Neighborhood'] == 'Edwards']
plot_pct_cat_variables(edwards, 'Exterior1st')
plot_pct_cat_variables(edwards, 'Exterior2nd')
test['Exterior1st'].fillna("Wd Sdng", inplace=True)
test['Exterior2nd'].fillna("Wd Sdng", inplace=True)
plt.figure(figsize=[15,9])

sns.boxplot('MSSubClass', 'SalePrice', data=train)

plt.show()
plt.figure(figsize=[15,9])

sns.violinplot('MSSubClass', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'MSSubClass')
plt.figure(figsize=[15,9])

sns.violinplot('MSZoning', 'SalePrice', data=train)

plt.show()
plt.figure(figsize=[15,9])

sns.boxplot('MSZoning', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'MSZoning')
print("The different categories for MSZoning in the test set are: %s" % test['MSZoning'].unique())
print("There are %d houses in the test set that does not have a MSZoning category" % test['MSZoning'].isnull().sum())
sns.jointplot('LotFrontage', 'SalePrice', data=train, size=10)

plt.show()
train.corr()['LotFrontage'].sort_values(ascending=False)
sns.jointplot('LotArea', 'SalePrice', data=train, size=10)

plt.show()
plot_pct_cat_variables(train, 'Street')
print("Number of houses connected to gravel road in the training set {}.".format(train[train['Street'] == 'Grvl'].shape[0]))
print("Number of houses connected to gravel road in the test set {}.".format(test[test['Street'] == 'Grvl'].shape[0]))
plot_pct_cat_variables(train, 'Alley')
plot_pct_cat_variables(train, 'LotShape')
plt.figure(figsize=[15,9])

sns.violinplot('LotShape', 'SalePrice', data=train)

plt.show()
plt.figure(figsize=[15,9])

sns.boxplot('LotShape', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'LandContour')
plt.figure(figsize=[15,9])

sns.violinplot('LandContour', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'Utilities')
plot_pct_cat_variables(test, 'Utilities')
plot_pct_cat_variables(train, 'LotConfig')
plt.figure(figsize=[15,9])

sns.violinplot('LotConfig', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'LandSlope')
plot_pct_cat_variables(train, 'Neighborhood', rotate_xticks=True)
plt.figure(figsize=[15,9])

sns.boxplot('SalePrice', 'Neighborhood', data=train, orient='h')

plt.show()
plot_pct_cat_variables(train, 'Condition1')
plot_pct_cat_variables(train, 'Condition2')
plt.figure(figsize=[15,9])

sns.violinplot('Condition1', 'SalePrice', data=train)

plt.show()
plt.figure(figsize=[15,9])

sns.violinplot('Condition2', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'BldgType')
plt.figure(figsize=[15,9])

sns.violinplot('BldgType', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'HouseStyle')
plt.figure(figsize=[15,9])

sns.violinplot('HouseStyle', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, "OverallQual")
plt.figure(figsize=[15,9])

sns.violinplot('OverallQual', 'SalePrice', data=train)

plt.show()
plt.figure(figsize=[15,9])

sns.boxplot('OverallQual', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'OverallCond')
plt.figure(figsize=[15,9])

sns.violinplot('OverallCond', 'SalePrice', data=train)

plt.show()
sns.jointplot('YearBuilt', 'SalePrice', data=train, stat_func=spearmanr, size=10)

plt.show()
sns.jointplot('YearRemodAdd', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
plot_pct_cat_variables(train, 'RoofStyle')
plt.figure(figsize=[15,9])

sns.violinplot('RoofStyle', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'RoofMatl')
plot_pct_cat_variables(train, 'Exterior1st')
plt.figure(figsize=[15,9])

sns.violinplot('Exterior1st', 'SalePrice', data=train)

plt.show()
plt.figure(figsize=[15,9])

sns.boxplot('Exterior1st', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'MasVnrType')
plt.figure(figsize=[15,9])

sns.violinplot('MasVnrType', 'SalePrice', data=train)

plt.show()
sns.jointplot('MasVnrArea', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
plot_pct_cat_variables(train, 'ExterQual')
plt.figure(figsize=[15,9])

sns.violinplot('ExterQual', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'ExterCond')
plt.figure(figsize=[15,9])

sns.violinplot('ExterCond', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'Foundation')
plt.figure(figsize=[15,9])

sns.violinplot('Foundation', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'BsmtQual')
plt.figure(figsize=[15,9])

sns.violinplot('BsmtQual', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'BsmtCond')
plt.figure(figsize=[15,9])

sns.violinplot('BsmtCond', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'BsmtExposure')
plt.figure(figsize=[15,9])

sns.violinplot('BsmtExposure', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'BsmtFinType1')
plt.figure(figsize=[15,9])

sns.violinplot('BsmtFinType1', 'SalePrice', data=train)

plt.show()
sns.jointplot('BsmtFinSF1', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
plot_pct_cat_variables(train, 'BsmtFinType2')
plt.figure(figsize=[15,9])

sns.violinplot('BsmtFinType2', 'SalePrice', data=train)

plt.show()
sns.jointplot('BsmtFinSF2', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
sns.jointplot('TotalBsmtSF', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
sns.jointplot('BsmtUnfSF', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
plot_pct_cat_variables(train, 'Heating')
plot_pct_cat_variables(train, 'HeatingQC')
plt.figure(figsize=[15,9])

sns.violinplot('HeatingQC', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'CentralAir')
plt.figure(figsize=[15,9])

sns.violinplot('CentralAir', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'Electrical')
plt.figure(figsize=[15,9])

sns.violinplot('Electrical', 'SalePrice', data=train)

plt.show()
sns.jointplot('1stFlrSF', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
sns.jointplot('2ndFlrSF', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
sns.jointplot('LowQualFinSF', 'SalePrice', data=train)

plt.show()
sns.jointplot('GrLivArea', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'BsmtFullBath')
sns.jointplot('BsmtFullBath', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
plot_pct_cat_variables(train, 'BsmtHalfBath')
sns.jointplot('BsmtHalfBath', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
plot_pct_cat_variables(train, 'FullBath')
sns.jointplot('FullBath', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
plot_pct_cat_variables(train, 'HalfBath')
sns.jointplot('HalfBath', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
plot_pct_cat_variables(train, 'BedroomAbvGr')
sns.jointplot('BedroomAbvGr', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
plot_pct_cat_variables(train, 'KitchenAbvGr')
sns.jointplot('KitchenAbvGr', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
plot_pct_cat_variables(train, 'KitchenQual')
plt.figure(figsize=[15,9])

sns.violinplot('KitchenQual', 'SalePrice', data=train)

plt.show()
sns.jointplot('TotRmsAbvGrd', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
plot_pct_cat_variables(train, 'Functional')
plt.figure(figsize=[15,9])

sns.violinplot('Functional', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'Fireplaces')
sns.jointplot('Fireplaces', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
plot_pct_cat_variables(train, 'FireplaceQu')
plt.figure(figsize=[15,9])

sns.violinplot('FireplaceQu', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'GarageType')
plt.figure(figsize=[15,9])

sns.violinplot('GarageType', 'SalePrice', data=train)

plt.show()
sns.jointplot('GarageYrBlt', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
plot_pct_cat_variables(train, 'GarageFinish')
plt.figure(figsize=[15,9])

sns.violinplot('GarageFinish', 'SalePrice', data=train)

plt.show()
sns.jointplot('GarageCars', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'GarageQual')
plt.figure(figsize=[15,9])

sns.violinplot('GarageQual', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'GarageCond')
plt.figure(figsize=[15,9])

sns.violinplot('GarageCond', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'PavedDrive')
sns.jointplot('WoodDeckSF', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
sns.jointplot('OpenPorchSF', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
sns.jointplot('EnclosedPorch', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
sns.jointplot('3SsnPorch', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
sns.jointplot('ScreenPorch', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
sns.jointplot('PoolArea', 'SalePrice', data=train, stat_func=spearmanr)

plt.show()
plot_pct_cat_variables(train, 'PoolQC')
plot_pct_cat_variables(train, 'Fence')
plot_pct_cat_variables(train, 'MiscFeature')
sns.jointplot('MiscVal', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'MoSold', sort=False)
plt.figure(figsize=[15,9])

sns.violinplot('MoSold', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'YrSold', sort=False)
plt.figure(figsize=[15,9])

sns.violinplot('YrSold', 'SalePrice', data=train)

plt.show()
tmp = train['YrSold'].astype(str) + '-' + train['MoSold'].astype(str)
train['YrMoSold'] = tmp.apply(pd.Period)

train.groupby(['YrMoSold'])['SalePrice'].median().plot()
plot_pct_cat_variables(train, 'SaleType')
plt.figure(figsize=[15,9])

sns.violinplot('SaleType', 'SalePrice', data=train)

plt.show()
plot_pct_cat_variables(train, 'SaleCondition')
plt.figure(figsize=[15,9])

sns.violinplot('SaleCondition', 'SalePrice', data=train)

plt.show()