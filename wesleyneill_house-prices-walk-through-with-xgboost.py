# regular expressions
import re 

# math and data utilities
import numpy as np
import pandas as pd
import scipy.stats as ss
import itertools as it

# ML and statistics libraries
import xgboost as xgb
import scipy.stats as stats
import sklearn.preprocessing as pre
from sklearn import model_selection
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC

# visualization libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Set-up default visualization parameters
mpl.rcParams['figure.figsize'] = [10,6]
viz_dict = {
    'axes.titlesize':18,
    'axes.labelsize':16,
}
sns.set_context("notebook", rc=viz_dict)
sns.set_style("darkgrid")
sns.set_palette(sns.color_palette("GnBu_r", 19))
# for kaggle path names
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
train_df.head()
# Roughly same amount of samples in both training and testing data
test_df.shape, train_df.shape
sigma = train_df.SalePrice.std()
mu = train_df.SalePrice.mean()
med = train_df.SalePrice.median()
mode = train_df.SalePrice.mode().to_numpy()

sns.distplot(train_df.SalePrice)
plt.axvline(mode, linestyle='--', color='green', label='mode')
plt.axvline(med, linestyle='--', color='blue', label='median')
plt.axvline(mu, linestyle='--', color='red', label='mean')
plt.legend()
sns.distplot(np.log1p(train_df.SalePrice), kde=True)
fig, axes = plt.subplots(2,2, figsize=(12,12))
sns.scatterplot(x='GrLivArea', y='SalePrice', hue='OverallQual', data=train_df, ax=axes[0,0])
sns.scatterplot(x='GrLivArea', y='SalePrice', hue='OverallCond', data=train_df, ax=axes[0,1])
sns.boxplot(x='OverallQual', y='SalePrice', data=train_df, ax=axes[1,0])
sns.boxplot(x='OverallCond', y='SalePrice', data=train_df, ax=axes[1,1])
plt.tight_layout()
drop_mask = (train_df.GrLivArea > 4000) & (train_df.SalePrice < 200000)
drop_idx = train_df[drop_mask].index
train_df = train_df.drop(drop_idx)
# verify outliers are dropped:
sns.relplot(x='GrLivArea', y='SalePrice', hue='OverallQual',data=train_df)
# A function to vizualize missing data in a DataFrame
def viz_missing(df):

    missing = pd.DataFrame({
        'Missing':df.isnull().sum(),
        '% Missing': df.isnull().sum()/len(df)
    })
    missing = missing[missing['% Missing'] > 0].sort_values(by='Missing', ascending=False)
    sns.barplot(x=missing.index, y='% Missing', data=missing)
    plt.xticks(rotation=45)
    plt.show()

viz_missing(train_df)
nofire_na = train_df.loc[train_df.FireplaceQu.isna() &
                         (train_df.Fireplaces == 0)].shape[0]
train_df.FireplaceQu.isna().sum(), nofire_na
train_df.FireplaceQu.fillna('NA', inplace=True)
pool_na = train_df.loc[train_df.PoolQC.isna() & (train_df.PoolArea == 0)].shape[0]
pool_na, train_df.PoolQC.isna().sum()
train_df.PoolQC.fillna("NA", inplace=True)
viz_missing(train_df)
train_df.MiscFeature.value_counts()
train_df.MiscFeature.fillna("NA", inplace=True)
train_df.MasVnrArea.fillna(train_df.MasVnrArea.median(), inplace=True)
train_df.LotFrontage.fillna(train_df.LotFrontage.median(), inplace=True)
viz_missing(train_df)
for var in ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']:
    print(train_df[var].isna().sum())
for var in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    train_df[var].fillna("NA", inplace=True)
    
viz_missing(train_df)
for var in ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond']:
    print(train_df[var].isnull().sum())
# Are all of the missing values associated with one another? 
train_df.loc[train_df.BsmtExposure.isna(), ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond']]
for var in ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond']:
    train_df[var].fillna('NA', inplace=True)
train_df[['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond']].isna().sum()
train_df.BsmtFinType2.value_counts()
viz_missing(train_df)
train_df.loc[train_df.GarageYrBlt.isna(), 'GarageYrBlt'] = train_df.YearBuilt
train_df.MasVnrType.fillna('None', inplace=True) # fill with mode
train_df.Electrical.fillna('Sbrkr', inplace=True)
train_df = train_df.drop(['Alley', 'Fence'], axis=1)
viz_missing(test_df)
m = test_df.PoolQC.mode()[0]

# Home has a pool, fill in the quality with the mode of existing data
test_df.loc[test_df.PoolQC.isna() & (test_df.PoolArea > 0), 'PoolQC'] = m

#Home has no pool, so instead fill missing value with 'NA'
test_df.loc[test_df.PoolQC.isna() & (test_df.PoolArea == 0), ['PoolQC']] = 'NA'

# And the same for some other variables. 
test_df.MiscFeature.fillna('NA', inplace=True)
test_df.FireplaceQu.fillna('NA', inplace=True)
test_df.MasVnrArea.fillna(test_df.MasVnrArea.median(), inplace=True)
test_df.LotFrontage.fillna(test_df.LotFrontage.median(), inplace=True)

# Drop these two columns, just as we did for the training data.
test_df.drop(['Alley', 'Fence'], axis=1, inplace=True)
viz_missing(test_df)
garage_vals = ['GarageCond', 'GarageQual', 'GarageFinish', 'GarageType']
test_df.loc[test_df.GarageArea.isna(), garage_vals] 
# remove garage type so existing value isn't overwritten
g_type = garage_vals.pop(3)

for g in garage_vals:
    mode = test_df[g].mode()[0]
    test_df.loc[test_df.GarageArea.isna(), g] = mode

# Fill single entry of missing GarageArea with median    
med = test_df.GarageArea.median()
test_df.loc[test_df.GarageArea.isna(), 'GarageArea']  = med

# add garage type back to list    
garage_vals.append(g_type)    

# Check that missing data was filled. Should see empty dataframe    
test_df.loc[test_df.GarageArea.isna(), garage_vals]
for g in garage_vals:
    
    # For any missing value where there is proof a garage exists, fill with mode.
    mode = test_df[g].mode()[0]
    test_df.loc[test_df[g].isna() & (test_df.GarageArea > 0), g] = mode
    
    # For any missing value where no garage exists, fill with 'NA'
    test_df.loc[test_df[g].isna() & (test_df.GarageArea == 0), g] = 'NA'

# If Garage Year Built is missing, fill it with the year the home was built
years = test_df.loc[test_df.GarageYrBlt.isna(), 'YearBuilt']
test_df.loc[test_df.GarageYrBlt.isna(), 'GarageYrBlt'] = years
bsmt_vars = ['BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1']
bsmt_sf = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
test_df.loc[test_df.TotalBsmtSF.isna(), bsmt_vars+bsmt_sf]
test_df.loc[test_df.TotalBsmtSF.isna(), bsmt_sf] = 0
# check to see if all area variables were set to zero
test_df.loc[:, bsmt_sf].isna().sum()
# Fill blanks with same logic as above (garage variables)
for var in bsmt_vars:
    mode = test_df[var].mode()[0]
    test_df.loc[test_df[var].isnull() & (test_df.TotalBsmtSF == 0), var] = 'NA'
    test_df.loc[test_df[var].isnull() & (test_df.TotalBsmtSF > 0), var] = mode

# Assume type 2 finish is the same as type 1 finish if missing data
finish = test_df.loc[test_df.BsmtFinType2.isna(), 'BsmtFinType1']
test_df.loc[test_df.BsmtFinType2.isna(), 'BsmtFinType2'] = finish
viz_missing(test_df)
test_df.loc[test_df.MasVnrType.isna(), ['MasVnrArea']]
# All but one instances don't have any veneer. Fill single value with mode:
mode = test_df.MasVnrType.mode()[0]
test_df.loc[(test_df.MasVnrType.isna() & (test_df.MasVnrArea > 0)), 'MasVnrType'] = mode
test_df.loc[test_df.MasVnrType.isna(), ['MasVnrType']] = 'None'
viz_missing(test_df)
test_df.loc[test_df.MSZoning.isna(), 'Neighborhood']
test_df.loc[test_df.Neighborhood == 'IDOTRR', 'MSZoning'].value_counts()
test_df.loc[test_df.Neighborhood == 'Mitchel', 'MSZoning'].value_counts()
mask1 = (test_df.Neighborhood == 'IDOTRR') & (test_df.MSZoning.isna())
mask2 = (test_df.Neighborhood == 'Mitchel') & (test_df.MSZoning.isna())
test_df.loc[mask1, 'MSZoning'] = 'RM'
test_df.loc[mask2, 'MSZoning'] = 'RL'
viz_missing(test_df)
# most of the remaining variables are categorical
rem_vars = ['Utilities', 'Functional', 'Exterior1st', 
        'Exterior2nd', 'KitchenQual', 'SaleType']

for var in rem_vars:
    mode = test_df[var].mode()[0]
    test_df.loc[test_df[var].isna(), var] = mode
test_df.loc[test_df.GarageCars.isna(), 'GarageArea']
test_df.loc[test_df.GarageCars.isna(), 'GarageCars'] = 2
bsmt_bths = ['BsmtFullBath', 'BsmtHalfBath']
mask = (test_df.BsmtFullBath.isna() | test_df.BsmtHalfBath.isna())
test_df.loc[mask, ['BsmtFinSF1']+bsmt_bths]
test_df.loc[mask, bsmt_bths] = test_df.loc[mask, bsmt_bths].fillna(0)
# Did we fill all missing values? Should see 0
test_df.isna().sum().sum()
# Create variables that signify home has ____ feature.

# Training
train_df['cond*qual'] = (train_df['OverallCond'] * train_df['OverallQual']) / 100.0
train_df['TotIntArea'] = train_df['TotalBsmtSF'] + train_df['GrLivArea']
train_df['AgeSold'] = train_df['YrSold'] - train_df['YearBuilt']
train_df['Total_Bathrooms'] = train_df['FullBath'] \
                               + 0.5 * train_df['HalfBath'] \
                               + train_df['BsmtFullBath'] \
                               + 0.5 * train_df['BsmtHalfBath']
train_df['HasShed'] = train_df.MiscFeature.apply(lambda x: 1 if x == 'Shed' else 0)
train_df['HasTennis'] = train_df.MiscFeature.apply(lambda x: 1 if x == 'TenC' else 0)
train_df['HasGar2'] = train_df.MiscFeature.apply(lambda x: 1 if x == 'Gar2' else 0)
train_df['HasPool'] = train_df.PoolArea.apply(lambda x: 1 if x > 0 else 0)
train_df['HasDeck'] = train_df.WoodDeckSF.apply(lambda x: 1 if x > 0 else 0)
train_df['IsNew'] = train_df.YearBuilt.apply(lambda x: 1 if x > 2000 else 0)
train_df['IsOld'] = train_df.YearBuilt.apply(lambda x: 1 if x < 1946 else 0)


# Test
test_df['cond*qual'] = (test_df['OverallCond'] * test_df['OverallQual']) / 100.0
test_df['TotIntArea'] = test_df['TotalBsmtSF'] + test_df['GrLivArea']
test_df['AgeSold'] = test_df['YrSold'] - test_df['YearBuilt']
test_df['Total_Bathrooms'] = test_df['FullBath'] \
                               + 0.5 * test_df['HalfBath'] \
                               + test_df['BsmtFullBath'] \
                               + 0.5 * test_df['BsmtHalfBath']
test_df['HasShed'] = test_df.MiscFeature.apply(lambda x: 1 if x == 'Shed' else 0)
test_df['HasTennis'] = test_df.MiscFeature.apply(lambda x: 1 if x == 'TenC' else 0)
test_df['HasGar2'] = test_df.MiscFeature.apply(lambda x: 1 if x == 'Gar2' else 0)
test_df['HasPool'] = test_df.PoolArea.apply(lambda x: 1 if x > 0 else 0)
test_df['HasDeck'] = test_df.WoodDeckSF.apply(lambda x: 1 if x > 0 else 0)
test_df['IsNew'] = test_df.YearBuilt.apply(lambda x: 1 if x > 2000 else 0)
test_df['IsOld'] = test_df.YearBuilt.apply(lambda x: 1 if x < 1946 else 0)
porch_area = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
train_df['HasPorch'] = train_df.loc[:, porch_area] \
                            .sum(axis=1) \
                            .apply(lambda x: 1 if x > 0 else 0)

test_df['HasPorch'] = test_df.loc[:, porch_area] \
                            .sum(axis=1) \
                            .apply(lambda x: 1 if x > 0 else 0)
test_df.drop(['MiscFeature'], axis=1, inplace=True)
train_df.drop(['MiscFeature'], axis=1, inplace=True)
train_df[['HasPorch']+porch_area].sort_values(by='HasPorch')
# Show our new dataframe with added features
train_df.head()
# a selection of variables that I expect to add to home value and/or curbside appeal, by variable type
exp_ordinal = ['ExterQual', 'BsmtQual', 'GarageQual', 
               'KitchenQual', 'FireplaceQu', 'PoolQC',  
               'OverallQual', 'ExterCond', 'BsmtCond', 
               'GarageCond', 'OverallCond', 'HeatingQC',
               'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
               'GarageCars', 'GarageFinish', 'BedroomAbvGr', 
               'KitchenAbvGr', 'HouseStyle', 'TotRmsAbvGrd',
               'YearBuilt', 'MoSold', 'YrSold', 'MasVnrArea', 
               'Utilities', 'BsmtFullBath', 'BsmtHalfBath', 
               'FullBath', 'HalfBath', 'CentralAir', 
               'YearRemodAdd','LotShape', 'Functional']

exp_nominal = ['Neighborhood', 'MSZoning', 'Condition1', 
               'Condition2', 'RoofStyle', 
               'RoofMatl', 'Exterior1st', 'Exterior2nd', 
               'MasVnrType', 'Foundation', 'Electrical', 
               'SaleType', 'SaleCondition', 'HasShed',
               'HasTennis', 'HasGar2', 'HasPorch', 
               'HasDeck', 'HasPool', 'GarageType', 
               'LotConfig', 'PavedDrive', 'IsOld', 'IsNew']

exp_contin  = ['LotArea', 'LotFrontage', 'GrLivArea', 
               'GarageArea', 'BsmtFinSF1', 'BsmtFinSF2', 
               'TotalBsmtSF', 'BsmtUnfSF', '1stFlrSF', 
               '2ndFlrSF', 'LowQualFinSF', 'WoodDeckSF', 
               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
               'PoolArea', 'MiscVal', 'TotIntArea', 'SalePrice']
qual_map = {'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
qual_vars = ['ExterQual', 'ExterCond', 'BsmtQual', 
             'BsmtCond', 'HeatingQC', 'KitchenQual', 
             'FireplaceQu', 'GarageQual', 'GarageCond',
             'PoolQC']

for col in qual_vars:
    train_df[col] = train_df[col].map(qual_map)
    test_df[col] = test_df[col].map(qual_map)
    
# Make sure we see all numeric data:    
train_df[qual_vars]
# Maps for Ordinal varibles:

BsmtExposure_map = {
    'Gd':4,
    'Av':3,
    'Mn':2,
    'No':1,
    'NA':0
}

BsmtFinType1_map = {
    'NA': 0,
    'Unf': 1,
    'LwQ': 2,
    'Rec': 3,
    'BLQ': 4,
    'ALQ': 5,
    'GLQ': 6
}

BsmtFinType2_map = {
    'NA': 0,
    'Unf': 1,
    'LwQ': 2,
    'Rec': 3,
    'BLQ': 4,
    'ALQ': 5,
    'GLQ': 6
}

GarageFinish_map = {
    'NA': 0,
    'Unf': 1,
    'RFn': 2,
    'Fin': 3
}

Functional_map = {
    'Sal': 0,
    'Sev': 1,
    'Maj2': 3,
    'Maj1': 4,
    'Mod': 5,
    'Min2': 6,
    'Min1': 7,
    'Typ': 8
}

Utilities_map = {
    'ELO': 0,
    'NoSeWa': 1,
    'NoSewr': 2,
    'AllPub': 3
}

CentralAir_map = {
    'N': 0,
    'Y': 1
}

LotShape_map = {
    'IR3': 0,
    'IR2': 1,
    'IR1': 2,
    'Reg': 3
}

HouseStyle_map = {
    '1Story': 1,
    '1.5Unf': 2,
    '1.5Fin': 3,
    '2Story': 4,
    '2.5Unf': 5,
    '2.5Fin': 6,
    'SFoyer': 7,
    'SLvl': 8
}
train_df.BsmtExposure = train_df.BsmtExposure.map(BsmtExposure_map)
train_df.BsmtFinType1 = train_df.BsmtFinType1.map(BsmtFinType1_map)
train_df.BsmtFinType2 = train_df.BsmtFinType2.map(BsmtFinType2_map)     
train_df.GarageFinish = train_df.GarageFinish.map(GarageFinish_map)
train_df.Functional   = train_df.Functional.map(Functional_map)
train_df.Utilities    = train_df.Utilities.map(Utilities_map)
train_df.CentralAir   = train_df.CentralAir.map(CentralAir_map)
train_df.LotShape     = train_df.LotShape.map(LotShape_map)
train_df.HouseStyle   = train_df.HouseStyle.map(HouseStyle_map)

test_df.BsmtExposure = test_df.BsmtExposure.map(BsmtExposure_map)
test_df.BsmtFinType1 = test_df.BsmtFinType1.map(BsmtFinType1_map)
test_df.BsmtFinType2 = test_df.BsmtFinType2.map(BsmtFinType2_map)     
test_df.GarageFinish = test_df.GarageFinish.map(GarageFinish_map)
test_df.Functional   = test_df.Functional.map(Functional_map)
test_df.Utilities    = test_df.Utilities.map(Utilities_map)
test_df.CentralAir   = test_df.CentralAir.map(CentralAir_map)
test_df.LotShape     = test_df.LotShape.map(LotShape_map)
test_df.HouseStyle   = test_df.HouseStyle.map(HouseStyle_map)
train_df.BsmtExposure
fig, axes = plt.subplots(3, 1, figsize=(10,30))

#Too many variables for a single heatmap of all ordinal data
ord1 = exp_ordinal[0:17] + ['SalePrice']
ord2 = exp_ordinal[17:] + ['SalePrice']

corr1 = train_df[ord1].corr(method='kendall')
corr2 = train_df[ord2].corr(method='kendall')
corr3 = train_df[exp_contin].corr()

sns.heatmap(corr1, annot=True, ax=axes[0], fmt='0.2f').set_title('Ordinal Data 1')
sns.heatmap(corr2, annot=True, ax=axes[1], fmt='0.2f').set_title('Ordinal Data 2')
sns.heatmap(corr3, annot=True, ax=axes[2], fmt='0.2f').set_title('Continuous Data')

plt.tight_layout()
plt.show()
ord1_corr = pd.DataFrame(
    train_df[ord1].corr(method='kendall').loc['SalePrice']
)
ord1_corr.sort_values(by='SalePrice', inplace=True, ascending=False)

ord2_corr = pd.DataFrame(
    train_df[ord2].corr(method='kendall').loc['SalePrice']
)
ord2_corr.sort_values(by='SalePrice', inplace=True, ascending=False)

cont_corr = pd.DataFrame(
    train_df[exp_contin].corr().loc['SalePrice']
)
cont_corr.sort_values(by='SalePrice', inplace=True, ascending=False)
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10, 5))
sns.heatmap(cont_corr, annot=True, ax=ax1).set_title('Continuous')
sns.heatmap(ord1_corr, annot=True, ax=ax2).set_title('Ranked 1')
sns.heatmap(ord2_corr, annot=True, ax=ax3).set_title('Ranked 2')
plt.tight_layout()
plt.show()
def plot_response_corr(df, features, response, corr_type):
    
    cor = pd.DataFrame()
    width = len(features)*0.5
    height = 5
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Measure difference between Spearman's and Pearson's to analyze for non-linearity
    if corr_type == 's-p':
        
        cor['feature'] = features
        cor['spearman'] = [df[f].corr(df[response], 'spearman') for f in features]
        cor['pearson'] = [df[f].corr(df[response], 'pearson') for f in features]
        cor['comparison'] = cor['spearman'] - cor['pearson']
        
        sns.barplot(
            data=cor.sort_values(by='comparison'), 
            x='feature', y='comparison', ax=ax
        ).set_title('S-P Comparison')
           
    else:
        cor['feature'] = features
        cor[corr_type] = [df[f].corr(df[response], corr_type) for f in features]
        cor = cor.sort_values(corr_type)
        plt.figure(figsize=(len(features)*1.5, 5))
        axes = sns.barplot(data=cor, x='feature', y=corr_type, ax=ax) \
            .set_title(f'{corr_type} Association to Response Variable')        
    

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return cor 
df = plot_response_corr(train_df, exp_contin, 'SalePrice', 's-p')
df = plot_response_corr(train_df, exp_ordinal, 'SalePrice', 's-p')
# for ordering the boxplots by median
grouped = train_df.loc[:,['Neighborhood', 'SalePrice']] \
    .groupby(['Neighborhood']) \
    .mean() \
    .sort_values(by='SalePrice')
chart = sns.boxplot(x=train_df.Neighborhood, y=train_df.SalePrice, order=grouped.index)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.show()
grouped = train_df.loc[:,['MSZoning', 'SalePrice']] \
    .groupby(['MSZoning']) \
    .mean() \
    .sort_values(by='SalePrice')

chart = sns.boxplot(x=train_df.MSZoning, y=train_df.SalePrice, order=grouped.index)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.show()
grouped = train_df.loc[:,['Condition1', 'SalePrice']] \
    .groupby(['Condition1']) \
    .mean() \
    .sort_values(by='SalePrice')

chart = sns.boxplot(x=train_df.Condition1, y=train_df.SalePrice, order=grouped.index)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.show()
def anova(df, predictor, target, alpha=0.05):
    
    # get unique, indendant treatments (categories in predictory)
    treatments = df.loc[df[predictor].notnull(), predictor].unique()
    group_values = []
    
    # group target variable by category
    for t in treatments:
        group_values.append(df.loc[df[predictor] == t, target].to_numpy())
    
    # calculate degrees of freedom and confidence level
    k = len(treatments)
    n = df[predictor].notnull().sum()
    conf = 1 - alpha
    df1 = k - 1
    df2 = n - k
    
    # calculate critical value of F-distribution
    crit_val = stats.f.ppf(conf, df1, df2)
    
    # calculate F and p-values
    f, p = stats.f_oneway(*group_values)
    
    return f, p, crit_val
# DataFrame to hold ANOVA results.
f_frame = pd.DataFrame(columns=['variable','p-value','f-value', 'critical', 'passed'])

# Perform ANOVA on each variable. Store data for plotting
for var in exp_nominal+exp_ordinal:
    f, p, c = anova(train_df, var, 'SalePrice')
    f_frame.loc[len(f_frame)] = [var, p, f, c, (f > c)]

# Plot test results
f_frame.sort_values(by='f-value', ascending=False, inplace=True)

# sns.set(rc={'figure.figsize':(15,8)})
sns.barplot(x='variable', y='f-value', data=f_frame)
plt.xticks(rotation=90)
plt.show()
f_frame.loc[f_frame.passed == False]
train_df[exp_nominal].info()
train_idx = train_df.index
test_idx = test_df.index

comb_df = train_df.append(test_df)

to_encode = comb_df.select_dtypes(include='object')
encoder = pre.OneHotEncoder()
encoder.fit(to_encode)
encoded = encoder.transform(to_encode).toarray()
dum_df = pd.DataFrame(encoded, columns=encoder.get_feature_names(to_encode.columns))
comb_df = comb_df.join(dum_df)
comb_df.drop(to_encode.columns, axis=1, inplace=True)

# Split datasets back apart
train_df = comb_df.loc[train_idx].copy()
test_df = comb_df.loc[test_idx].copy()

# delete unused df
del comb_df

# verify correct dimensions (# of columns)
train_df.shape, test_df.shape
# test_df picked up a 'SalePrice' column when we joined DF's. Let's drop it.
test_df.drop(['SalePrice'], axis=1, inplace=True)
train_df.shape, test_df.shape
to_drop = ['BsmtFinSF2', 'BsmtHalfBath', 'MiscVal', 'Utilities', 
           'HasGar2', 'HasTennis', 'MoSold', 'YrSold']

train_df.drop(to_drop, axis=1, inplace=True)
test_df.drop(to_drop, axis=1, inplace=True)
X = train_df.drop(['SalePrice'], axis=1)
Y = train_df.pop('SalePrice')
# If only there was a function called "lock"
# Split Data
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, random_state=1)
# ROUND 1:
param_grid1 = {
    'max_depth':np.arange(1,11,2),
    'min_child_weight':np.arange(1,11,2)
}

# First Search
gsearch1 = model_selection.GridSearchCV(xgb.XGBRFRegressor(),
    param_grid1, 
    scoring='neg_mean_squared_log_error',
    cv=4)

gsearch1 = gsearch1.fit(x_train, y_train)
best_params = gsearch1.best_params_
best_params, -gsearch1.best_score_
# ROUND 2
# use value range one above and one below previous round's results:
a = best_params['max_depth'] - 1
b = best_params['max_depth'] + 2
c = best_params['min_child_weight'] - 1
d = best_params['min_child_weight'] + 2


param_grid2 = {
    'max_depth':range(a, b),
    'min_child_weight':range(c, d)
}

gsearch2 = model_selection.GridSearchCV(
    xgb.XGBRegressor(**best_params), 
    param_grid2, 
    scoring='neg_mean_squared_log_error', 
    cv=4
)

gsearch2 = gsearch2.fit(x_train, y_train)
best_params.update(gsearch2.best_params_)
best_params, -gsearch2.best_score_
# Small drop in performance here. 
param_grid3 = {'gamma':[i/10.0 for i in range(0,6)]}

gsearch3 = model_selection.GridSearchCV(
    xgb.XGBRFRegressor(**best_params),
    param_grid3,
    scoring='neg_mean_squared_log_error', 
    cv=5
)

gsearch3 = gsearch3.fit(x_train, y_train)
best_params.update(gsearch3.best_params_)
best_params, -gsearch3.best_score_
param_grid4 = {
    'colsample_bytree':[i/10.0 for i in range(0,11)],
    'subsample':[i/10.0 for i in range(0,11)]
}

gsearch4 = model_selection.GridSearchCV(
    xgb.XGBRFRegressor(**best_params),
    param_grid4,
    scoring='neg_mean_squared_log_error',
    cv=3   
)

gsearch4 = gsearch4.fit(x_train, y_train)
best_params.update(gsearch4.best_params_)
best_params, -gsearch4.best_score_
param_grid5 = {
    'alpha':[i/10 for i in range(0,11)],
    'lambda':[i/10 for i in range(0,11)]
}

gsearch5 = model_selection.GridSearchCV(
    xgb.XGBRegressor(**best_params),
    param_grid5, 
    scoring='neg_mean_squared_log_error',
    cv=5
)

gsearch5 = gsearch5.fit(x_train, y_train)
best_params.update(gsearch5.best_params_)
best_params, -gsearch5.best_score_
param_grid6 = {
    'n_estimators':np.arange(50, 450, 50),
    'learning_rate':[0.01, 0.05, 0.1, .5, 1]
}

gsearch6 = model_selection.GridSearchCV(
    xgb.XGBRegressor(**best_params),
    param_grid6, 
    scoring='neg_mean_squared_log_error',
    cv=5
)

gsearch6 = gsearch6.fit(x_train, y_train)
best_params.update(gsearch6.best_params_)
best_params, -gsearch6.best_score_
# Compare base model with tuned model: 
base_booster = xgb.XGBRegressor()
tune_booster = xgb.XGBRegressor(**best_params)

# We want to shuffle the data for the cross_val_score 
kf = model_selection.KFold(5, shuffle=True, random_state=1)
score1 = model_selection.cross_val_score(
    base_booster, 
    x_train, 
    y_train, 
    scoring='neg_mean_squared_log_error',
    cv = kf
)

score2 = model_selection.cross_val_score(
    tune_booster, 
    x_train, 
    y_train, 
    scoring='neg_mean_squared_log_error',
    cv = kf
)
# Compare the performance of tuned and untuned models. 
print(f"Base: Root Mean Log Error: {-score1.mean()} (+/-{score1.std()*2})")
print(f"Tuned: Root Mean Log Error: {-score2.mean()} (+/-{score2.std()*2})")   
# Compare CV scores with final score on unseen data:
tune_booster.fit(x_train, y_train)
pred = tune_booster.predict(x_test)
metrics.mean_squared_log_error(y_test, pred)
tune_booster.fit(X, Y)
final_preds = tune_booster.predict(test_df)
submission = pd.DataFrame({'Id':test_df.index, 'SalePrice':final_preds})
submission.to_csv('submission3.csv', index=False)
submission