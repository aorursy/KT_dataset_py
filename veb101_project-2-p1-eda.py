import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew

from sklearn.preprocessing import StandardScaler

import os

import warnings

warnings.filterwarnings('ignore')



#Some styling

sns.set_style("darkgrid")

plt.style.use("fivethirtyeight")

pd.pandas.set_option('display.max_columns', None)



%matplotlib inline
# set dataset path



train_data = os.path.join("/kaggle/input/house-prices-advanced-regression-techniques", "train.csv")

test_data = os.path.join("/kaggle/input/house-prices-advanced-regression-techniques", "test.csv")
# read datasets



train = pd.read_csv(train_data)

test = pd.read_csv(test_data)
# size of dataset

# as 81 features so we can try different feature selection and feature engineering methods



train.shape
# display info about the dataset



train.columns
# sneak peek



train.head()
# let's take look at stats for SalePrice



train["SalePrice"].describe()
_ = sns.distplot(train.SalePrice)
#skewness and kurtosis



print(f"Skewness: {train['SalePrice'].skew()}")

print(f"Kurtosis: {train['SalePrice'].kurt()}")
#scatter plot

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)

plt.scatter(train['GrLivArea'], train['SalePrice'], c='red', alpha=0.25)

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')

plt.ylim(0,800000)



plt.subplot(1, 2, 2)

plt.scatter(train['TotalBsmtSF'], train['SalePrice'], c='k', alpha=0.25)

plt.xlabel('TotalBsmtSF')

plt.ylabel('SalePrice')

plt.ylim(0,800000)

plt.show()
plt.figure(figsize=(8, 6))

var = 'OverallQual'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

_ = sns.boxplot(x=var, y="SalePrice", data=data)

plt.axis(ymin=0, ymax=800000)

plt.show()
plt.figure(figsize=(20, 8))

var = 'YearBuilt'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000)

_ = plt.xticks(rotation=90)
plt.figure(figsize=(20, 6))

var = 'TotRmsAbvGrd'

plt.subplot(1, 2, 1)

data = pd.concat([train['SalePrice'], train[var]], axis=1)

_ = sns.boxplot(x=var, y="SalePrice", data=data)

plt.axis(ymin=0, ymax=800000)

plt.show()
plt.figure(figsize=(18, 10))

corrmat = train.drop('Id', 1).corr()

_ = sns.heatmap(corrmat, vmax=1.0, square=True, fmt='.2f', 

            cmap='coolwarm', annot_kws={'size': 8})

plt.show()
plt.figure(figsize=(6, 6))



k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', cmap='coolwarm',

                 annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
numerical_features = train.select_dtypes('number').columns.to_list()

numerical_features.pop(0)

print(f"Number of numerical features: {len(numerical_features)}")



# although numerical there are features which reprsent years

# there are some dicrete numerical features too.

# we need to remove them from the list.





year_features = [feature for feature in numerical_features 

                 if 'Yr' in feature or 'Year' in feature]

print(f"Number of Temporal features: {len(year_features)}")





discrete_features = [feature for feature in numerical_features 

                     if train[feature].nunique()<= 15 and feature not in year_features]

print(f"Number of discrete numerical features: {len(discrete_features)}")



continuous_num_features = [feature for feature in numerical_features 

                     if feature not in discrete_features + year_features] 

                                               

print(f"Number of continuous numerical features: {len(continuous_num_features)}")
plt.figure(figsize=(15, 10))

corrmat = train[continuous_num_features].corr()

sns.heatmap(corrmat, vmax=1.0, square=True, fmt='.2f', 

            annot=True, cmap='coolwarm', annot_kws={'size': 8});
# SalePrice scatter plot with highly correlated features
plt.figure(figsize=(10, 8))



sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

_ = sns.pairplot(train[cols], size = 2.5, diag_kind='kde')

plt.show()
features_with_na = [features for features in train.columns if train[features].isnull().sum() >= 1]



a = pd.DataFrame({

    'features': features_with_na,

    'Total': [train[i].isnull().sum() for i in features_with_na],

    'Missing_PCT': [np.round(train[i].isnull().sum()/ train.shape[0], 4) for i in features_with_na]

}).sort_values(by='Missing_PCT', ascending=False).reset_index(drop=True)

a.style.background_gradient(cmap='Reds') 

train['FireplaceQu'].value_counts()
train[train['FireplaceQu'].isnull()][['Fireplaces','FireplaceQu']]
#Unique elements

train['MasVnrType'].unique()
train[train['MasVnrType'].isnull()][['MasVnrType','MasVnrArea']]
# Let's look at the repeated value in MasVnrType column



train['MasVnrType'].mode()
train[train['BsmtQual'].isnull()][['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1',

                        'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']].head(15)
train[['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

 'BsmtFinType2']].mode()
train[train['GarageType'].isnull()][['GarageType', 'GarageYrBlt', 'GarageFinish',

       'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']]
train[['GarageType','GarageFinish',

 'GarageQual','GarageCond']].mode()
plt.figure(figsize=(20, 20))

for i, feature in enumerate(features_with_na, 1):

    plt.subplot(5, 5, i)

    data = train.copy()

    

    # let's make a variable that indicates 1 if the observation was missing or zero otherwise

    data[feature] = np.where(data[feature].isnull(), 1, 0)

    

    # calculate the median SalePrice where the information is missing or present

    temp = data.groupby(feature)['SalePrice'].median()

    _ = sns.barplot(x=temp.index, y=temp.values)

    plt.title(feature)

    plt.xlabel("")

    plt.xticks([0, 1], ["Present", "Missing"])

    plt.ylabel("Sales Price", rotation=90)

plt.tight_layout(h_pad=2, w_pad=2)

plt.show()
#standardizing data

saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]



print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
data = train.copy()

# sale_price = np.log(train['SalePrice'])



for i, feature in enumerate(continuous_num_features, 1):

    data = train[feature].copy()

    if 0 in data.unique(): # as log 0 is undefinedz

        continue

    else:

        data = np.log(data)   

        data.name = feature

        _ = plt.figure(figsize=(6, 6))

        _ = sns.boxplot(y=data)

    

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
train.select_dtypes('number').columns
sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend([f'Normal dist. ($\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f} )'],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
# GrLivArea

# histogram and normal probability plot



sns.distplot(train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['GrLivArea'], plot=plt)
# TotalBsmtSF

# histogram and normal probability plot



sns.distplot(train['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['TotalBsmtSF'], plot=plt)
print("\nSkew in numerical features: \n")

pd.DataFrame({

    'Feature': numerical_features,

    'Skewness': skew(train[numerical_features])}).sort_values(by='Skewness',

                                                              ascending=False).set_index('Feature').head(10)
train[year_features].head()
plt.figure(figsize=(15, 9))

for i, feature in enumerate(year_features, 1):

    plt.subplot(2, 2, i)

    temp = train.groupby(feature)['SalePrice'].median().plot()

    plt.xlabel(feature)

    # plt.xlim([2006, 2010])

    # plt.xticks(range(2006, 2011))

    plt.title(f"Median price vs {feature}")

    plt.ylabel("Sale price", rotation=90)

plt.tight_layout(w_pad=1.2, h_pad=1.2)

plt.show()
plt.figure(figsize=(20, 6))



for i, feature in enumerate(year_features, 1):

    

    if feature != 'YrSold':

        data = train.copy()

        

        data[feature] = data['YrSold'] - data[feature]

        plt.subplot(1, 3, i)

#         plt.scatter()

        plt.title(feature)

        

        plt.ylabel('SalePrice')

        sns.regplot(data[feature], data['SalePrice'], 

                   scatter_kws={"color": "black"}, line_kws={"color": "red"})

        plt.xlabel(f"No. of years: {feature}")

plt.tight_layout()

plt.show()        
sns.heatmap(pd.DataFrame({

    'SalePrice': train['SalePrice'],

    'YearBuiltAge': train['YrSold'] - train['YearBuilt'],

    'YearRemodAddAge': train['YrSold'] - train['YearRemodAdd'],

    'GarageYrBltAge': train['YrSold'] - train['GarageYrBlt'],

}).corr(), annot=True, cmap='coolwarm')

plt.title('Features as age vs SalePrice')

plt.show()
discrete_features
train[discrete_features].head()
plt.figure(figsize=(20, 20))

for i, feature in enumerate(discrete_features, 1):

    plt.subplot(6, 3, i)

    data = train.copy()

    

    # let's make a variable that indicates 1 if the observation was missing or zero otherwise

    

    # calculate the median SalePrice where the information is missing or present

    temp = data.groupby(feature)['SalePrice'].median()

    _ = sns.barplot(x=temp.index, y=temp.values)

    plt.title(feature)

    plt.xlabel("")

    plt.ylabel("Sales Price", rotation=90)

    plt.xticks(rotation=45)

plt.tight_layout(h_pad=2, w_pad=2)

plt.show()
print(f"Number of Continuous numerical feature: {len(continuous_num_features)}")
continuous_num_features
train[continuous_num_features].head()
plt.figure(figsize=(20, 20))

for i, feature in enumerate(continuous_num_features, 1):

    plt.subplot(5, 4, i)

#     _ = sns.distplot(train[feature], kde_kws={'bw': 1.05})    

    _ = sns.distplot(train[feature], kde=False, rug=True)

    

plt.tight_layout(h_pad=2, w_pad=2)

plt.show()
data = train.copy()



sale_price = np.log(train['SalePrice'])



for i, feature in enumerate(continuous_num_features[:-1], 1):

    data = train[feature].copy()

    

    if 0 in data.unique(): # as log 0 is undefinedz

        continue

    else:

        data = np.log(data)    

        data.name = feature

        _ = plt.figure(figsize=(15, 8))

        plt.subplot(1, 2, 1)

        sns.regplot(data, sale_price, fit_reg=True,

                   scatter_kws={"color": "black"}, line_kws={"color": "red"}).set_title(f"Correlation: {data.corr(sale_price)}")

        plt.subplot(1, 2, 2)

        sns.distplot(data).set_title(f"Log transformation of: {feature}")

plt.show()
categorical_features = [feature for feature in train.columns if train[feature].dtypes=='O']

print(f"Number of categorical feature: {len(categorical_features)}")
categorical_features
train[categorical_features].head()
pd.DataFrame({

    "features": categorical_features,

    "Nunique": [train[feature].nunique() for feature in categorical_features]             

             })
plt.figure(figsize=(40, 30))



for i, feature in enumerate(categorical_features, 1):

    data = train.copy()

    temp = data.groupby(feature)['SalePrice'].median()

    plt.subplot(9, 5, i)

    sns.barplot(temp.index, temp.values)

    plt.xticks(rotation=45)



plt.tight_layout(h_pad=1.2)

plt.show()