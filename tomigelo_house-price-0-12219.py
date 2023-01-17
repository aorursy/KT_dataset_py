# import libraries for data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# list data files that are connected to the kernel
import os
os.listdir('../input/')
# read the train.csv file into a datframe
df_train = pd.read_csv('../input/train.csv')
print('Shape: ', df_train.shape)
df_train.head()
# read the test.csv file into a datframe
df_test = pd.read_csv('../input/test.csv')
print('Shape: ', df_test.shape)
df_test.head()
df_train.info()
df_test.info()
# number of each type of column
df_train.dtypes.value_counts()
plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])
# remove outliers from train dataset
df_train = df_train[df_train['GrLivArea'] < 4000]

print('Shape: ', df_train.shape)
# create df_full by merging train and test data
df_full = df_train.append(df_test, sort=False)
print('Shape: ', df_full.shape)
# remove target value from the full dataset
df_full = df_full.drop(['SalePrice'], axis=1)

print('Shape: ', df_full.shape)
# create new feature 'RemodAdd' which contains 1 if some remodeling or additions have been done, else 0
df_full['RemodAdd'] = df_full.apply(lambda x: '0' if x['YearRemodAdd'] == x['YearBuilt'] else '1', axis=1)

# create new features which contains 1 if the feature exists, else 0
df_full['hasPool'] = df_full.apply(lambda x: '1' if x['PoolArea'] > 0 else '1', axis=1)
df_full['has2ndFloor'] = df_full.apply(lambda x: '1' if x['2ndFlrSF'] > 0 else '1', axis=1)
df_full['hasGarage'] = df_full.apply(lambda x: '1' if x['GarageArea'] > 0 else '1', axis=1)
df_full['hasBsmt'] = df_full.apply(lambda x: '1' if x['TotalBsmtSF'] > 0 else '1', axis=1)
df_full['hasFireplace'] = df_full.apply(lambda x: '1' if x['Fireplaces'] > 0 else '1', axis=1)

# change the type to numeric
df_full[['RemodAdd', 'hasPool', 'has2ndFloor', 'hasGarage', 'hasBsmt', 'hasFireplace']] = df_full[['RemodAdd', 'hasPool', 'has2ndFloor', 'hasGarage', 'hasBsmt', 'hasFireplace']].apply(pd.to_numeric)
# create a list with all ordinal features having standardized quality descriptions
quality_categories = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence']

# create a mapping list for the standardized values of the quality_category features
quality_mapping = {
    'Ex': '5', 'Gd': '4', 'TA': '3', 'Fa': '2', 'Po': '1', 'NA': '0',
    'Av': '3', 'Mn': '2', 'No': '1', # specific to BsmtExposure feature
    'GdPrv': '4', 'MnPrv': '3', 'GdWo': '2', 'MnWw': '1' # specific to Fence feature
}

# create a for loop to replace the standardized values with digits
# loop over featues from the list quality_categories in the full dataset
for col in df_full[quality_categories]:
    # fill NaN's with string 'NA'
    df_full[col].fillna('NA', inplace=True)
    # replace the values according to quality_mapping
    df_full.replace({col: quality_mapping}, inplace=True)
    # create new column as 'OldName_bin' where the data type is numeric
    df_full[col + '_bin'] = df_full[col].apply(pd.to_numeric)
# create a list with all the other ordinal features
ordinal_categories = ['Utilities', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'GarageFinish']

# import encoder library
from sklearn.preprocessing import LabelEncoder

# create a for loop to encode the categorical features
# loop over featues from the list ordinal_categories in the full dataset
for col in df_full[ordinal_categories]:
    # fill NaN's with string 'NA'
    df_full[col].fillna('NA', inplace=True)
    # define the encoder instance
    enc = LabelEncoder()
    # create new column as 'OldName_bin' with the encoded values
    df_full[col + '_bin'] = enc.fit_transform(df_full[col].astype(str))
# combine previously encoded features into a list
columns_to_drop = quality_categories + ordinal_categories

# drop already encoded features
df_full.drop(columns_to_drop, axis=1, inplace=True)

# check number of each type of column
df_full.dtypes.value_counts()
nulls = np.sum(df_full.isnull())
nullcols = nulls.loc[(nulls != 0)]
dtypes = df_full.dtypes
dtypes2 = dtypes.loc[(nulls != 0)]
info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0, ascending=False)

print(info)
print("There are", len(nullcols), "columns with missing values")
# drop columns with high amount of missing values
df_full.drop(['MiscFeature','Alley'], axis=1, inplace=True)

# create a list of columns for using the most common value method
column_nans = ['GarageType', 'MSZoning']
most_common = ['Electrical', 'Exterior2nd', 'Exterior1st', 'SaleType']

# fill columns with 'NA' or 'None'
df_full.loc[:,column_nans] = df_full.loc[:,column_nans].fillna('NA')
df_full.loc[:, 'MasVnrType'] = df_full.loc[:, 'MasVnrType'].fillna('None')

# create a for loop to fill some columns with the most common value
# loop over featues from the list most_common in the full dataset
for col in df_full[most_common]:
    # fill NaN's with most common value
    df_full[col].fillna(df_full[col].mode()[0], inplace=True)
# create a list of all numeric columns
numeric_columns = df_full._get_numeric_data().columns

# import libraries
from sklearn.impute import SimpleImputer

# define instance
imp = SimpleImputer(strategy='median')

# impute 
df_full[numeric_columns] = imp.fit_transform(df_full[numeric_columns])
# one-hot encoding of categorical features
df_full = pd.get_dummies(df_full)

print('Full data shape: ', df_full.shape)
# create a copy of the dataframe and add back the target
df_corrs = df_full.copy()
df_corrs['SalePrice'] = df_train['SalePrice']

# calculate all correlations in train data
corrs = df_corrs.corr()
corrs = corrs.sort_values('SalePrice', ascending = False)

# 10 most positive correlations
pd.DataFrame(corrs['SalePrice'].head(10))
# 10 most negative correlations
pd.DataFrame(corrs['SalePrice'].dropna().tail(10))
# set the threshold
threshold = 0.8

# empty dictionary to hold correlated variables
above_threshold_vars = {}

# for each column, record the variables that are above the threshold
for col in corrs:
    above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])
# track columns to remove and columns already examined
cols_to_remove = []
cols_seen = []
cols_to_remove_pair = []

# iterate through columns and correlated columns
for key, value in above_threshold_vars.items():
    # keep track of columns already examined
    cols_seen.append(key)
    for x in value:
        if x == key:
            next
        else:
            # only want to remove one in a pair
            if x not in cols_seen:
                cols_to_remove.append(x)
                cols_to_remove_pair.append(key)
            
cols_to_remove = list(set(cols_to_remove))
print('Number of columns to remove: ', len(cols_to_remove))
df_full_corrs_removed = df_full.drop(columns = cols_to_remove)

print('Full data with removed corrs shape: ', df_full_corrs_removed.shape)
# split the data from df_full back into train and test datasets using the length of df_train
train = df_full_corrs_removed.iloc[:len(df_train),:]
test = df_full_corrs_removed.iloc[len(df_train):,:]

# extract the labels
train_labels = df_train['SalePrice']

# log transform the labels:
train_labels_log = np.log1p(train_labels)

# align train and test data, keep only columns present in both dataframes
train, test = train.align(test, join = 'inner', axis = 1)

print('Train shape: ', train.shape)
print('Test shape: ', test.shape)
# import libraries
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

# define the number number of splits, alphas
kfolds = KFold(n_splits=5, random_state=42)

# define a function to calculate the RMSE (root mean squared logarithmic error) for a given model
def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, train, train_labels, scoring="neg_mean_squared_error", cv = kfolds))
    return(rmse)
# import libraries
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeCV

# define a function to select the best alpha for the ridge model
def ridge_selector(k):
    # create a pipeline for the ridge model
    ridge_model = Pipeline([
        ('scl', RobustScaler()),
        ('ridge', RidgeCV(alphas = [k], cv= kfolds))
    ]).fit(train, train_labels)
    # use the rmse_cv function to calculate the RMSE
    ridge_rmse = rmse_cv(ridge_model).mean()
    
    return(ridge_rmse)

# create a list of alpha values to try out
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 20, 30, 40, 60, 80]

# create empty list
ridge_scores = []

for alpha in alphas:
    score = ridge_selector(alpha)
    ridge_scores.append(score)
plt.plot(alphas, ridge_scores, label='Ridge')
plt.legend('center')
plt.xlabel('alpha')
plt.ylabel('score')

ridge_score_table = pd.DataFrame(ridge_scores, alphas, columns=['RMSE'])
ridge_score_table
# import libraries
from sklearn.linear_model import LassoCV

alphas = [5, 1, 0.1, 0.001, 0.0005]

lasso_rmse = rmse_cv(LassoCV(alphas = alphas)).mean()

print('Lowest average RMSE is:', lasso_rmse.min())
from lightgbm import LGBMRegressor

lgbm_model = Pipeline([
    ('scl', RobustScaler()),
    ('lightgbm', LGBMRegressor(objective='regression',
                               n_estimators=1000,
                               learning_rate=0.05,
                               num_leaves=5,
                               max_bin = 55,
                               bagging_fraction = 0.8,
                               bagging_freq = 5,
                               feature_fraction = 0.2319,
                               feature_fraction_seed=9,
                               bagging_seed=9,
                               min_data_in_leaf =6, 
                               min_sum_hessian_in_leaf = 11))
                              
]).fit(train, train_labels)

rmse_cv(lgbm_model).mean()
# predict on the test data
preds = lgbm_model.predict(test)
# make a submission dataframe
submit = df_test.loc[:, ['Id']]
submit.loc[:, 'SalePrice'] = preds

# Save the submission dataframe
submit.to_csv('submission.csv', index = False)