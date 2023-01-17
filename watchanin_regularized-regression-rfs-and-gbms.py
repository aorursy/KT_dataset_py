# General

import numpy as np 

import pandas as pd



# Warnings

import warnings

warnings.filterwarnings('ignore') # Supress any unnecessary warnings for readability



# Graphing

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (13, 8)

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')







# Machine libraries

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import RidgeCV, Ridge, LassoCV, Lasso, ElasticNetCV, ElasticNet

from sklearn.grid_search import GridSearchCV

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from scipy.special import boxcox1p 
# Read the data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# First five rows in the training data

train.head(5)
# List of all the columns in the training data

train.columns
# Histogram w/ distribution of SalePrice

sns.distplot(train['SalePrice'])

plt.title('Distribution of SalePrice');



# Print the skewness and kurtosis of SalePrice

print ("Skewness: %.3f" % train['SalePrice'].skew())

print ("Kurtosis: %.3f" % train['SalePrice'].kurt())
# Plot the SalesPrice data again using a log(1+x) transformation

sns.distplot(np.log1p(train['SalePrice'].astype(float)))

plt.title('Distribution of the log of SalePrice');

print ("Skewness (log): %.3f" % np.log1p(train['SalePrice'].astype(float)).skew())

print ("Kurtosis (log): %.3f" % np.log1p(train['SalePrice'].astype(float)).kurt())
# Which numeric variables are most correlated with SalePrice?

k = 10 # Number of variables for heatmap

corrmat = train.corr()

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1)

hm = sns.heatmap(

                    cm, 

                    cbar=True, 

                    annot=True, 

                    square=True, 

                    fmt='.2f', 

                    annot_kws={'size': 10}, 

                    yticklabels=cols.values, 

                    xticklabels=cols.values)

plt.title('How are correlated are the numeric features with SalePrice?')

plt.show()
# Box plot of OverallQual (the most correlated variable)  vs. SalePrice

data = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)

sns.boxplot(x='OverallQual', y='SalePrice', data=data)

plt.title('OverallQual and SalePrice')

plt.xticks(rotation=90);
# Scatter plot of GrLivArea (2nd most corrleated variable) and SalePrice

data = pd.concat([train['SalePrice'], train['GrLivArea']], axis=1)

data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000))

plt.title('GrLivArea and SalePrice');
# Box plot of Neighborhood and SalePrice.

data = pd.concat([train['SalePrice'], train['Neighborhood']], axis=1)

sns.boxplot(x='Neighborhood', y='SalePrice', data=data)

plt.xticks(rotation=90)

plt.title('Neighborhood and SalePrice');
# Box plot of Neighborhood and LotFrontage.

data = pd.concat([train['LotFrontage'], train['Neighborhood']], axis=1)

sns.boxplot(x='Neighborhood', y='LotFrontage', data=data)

plt.title('LotFrontage and Neighborhood')

plt.xticks(rotation=90);
# Does the MoSold have an effect on SalePrice?

data = pd.concat([train['MoSold'], train['SalePrice']], axis=1)

sns.boxplot(x='MoSold', y='SalePrice', data=data)

plt.title('MoSold and SalePrice')

plt.xticks(rotation=90);
# Does YrSold have an effect on SalePrice?

data = pd.concat([train['YrSold'], train['SalePrice']], axis=1)

sns.boxplot(x='YrSold', y='SalePrice', data=data)

plt.title('YrSold and SalePrice');

plt.xticks(rotation=90);
# Is the Id variable related to SalePrice at all? Or is it random?

data = pd.concat([train['SalePrice'], train['Id']], axis=1)

data.plot.scatter(x='Id', y='SalePrice', ylim=(0,800000))

plt.title('Id and SalePrice');
# Are there any missing data?



# Combine train and test data and remove SalePrice

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)



# How prevalent are null values in these data sets

total = all_data.isnull().sum().sort_values(ascending=False)

percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)

nulls = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

nulls = nulls[nulls['Total'] > 0]

nulls
# Are there any other skewed features besides SalePrice?

numeric_features = all_data.dtypes[all_data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_features = all_data[numeric_features].skew().sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_features})

skewed = skewness.loc[(skewness.Skew > 0.5) | (skewness.Skew < -0.5)]

print ("There are {} features with skewness greater than 0.5.".format(skewed.shape[0]))

skewed
# Store the Id column in separate variables -- use for indexing later

train_ID = train['Id']

test_ID = test['Id']



# Remove the Id variable from both the train and test sets

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
# Removing the two points where GrLivArea > 4000 and SalePrice < 300000

train = train[(train['GrLivArea'] < 4000) | (train['SalePrice'] > 300000)]
# Use np.log1p to apply log(1+x) to all elements in the SalePrice column

train["SalePrice"] = np.log1p(train["SalePrice"])
# Combine training and testing set again. Needs to be updated because two

# outliers were removed from the training data.



# Store values that will be needed later

ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.SalePrice.values



# Combine train and test data and remove SalePrice

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)



##################

##### PoolQC #####

##################

'''If PoolArea == 0, then 'None'; 'Gd' for the only other null.'''

all_data.ix[(all_data.PoolQC.isnull()) & (all_data.PoolArea == 0), 'PoolQC'] = 'None'

all_data.ix[all_data.PoolQC.isnull(), 'PoolQC'] = 'Gd'





#######################

##### MiscFeature #####

#######################

'''If MiscFeature is null, then assume there are no miscellaneous features.'''

all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')





#################

##### Alley #####

#################

'''If Alley is null, then assume there is no alley.'''

all_data.Alley = all_data.Alley.fillna('None')





#################

##### Fence #####

#################

'''If Fence is null, then assume there is no fence.'''

all_data.Fence = all_data.Fence.fillna('None')





#######################

##### FireplaceQu #####

#######################

'''All null observations for FireplaceQu are from houses with no fireplace.'''

all_data.FireplaceQu = all_data.FireplaceQu.fillna('None')





#####################

##### Utilities #####

#####################

'''Majority of the values are the same. Impute the mode.'''

all_data.Utilities = all_data.Utilities.fillna(all_data.Utilities.mode()[0])





######################

##### Functional #####

######################

'''Majority of the values are the same. Impute the mode.'''

all_data.Functional = all_data.Functional.fillna(all_data.Functional.mode()[0])



####################

##### SaleType #####

####################

'''Majority of the values are the same. Impute the mode.'''

all_data.SaleType = all_data.SaleType.fillna(all_data.SaleType.mode()[0])



######################

#### KitchenQual #####

######################

'''Only one missing value. Impute Gd, a common and middle value.'''

all_data.KitchenQual = all_data.KitchenQual.fillna("Gd")





#######################################

##### Exterior1st and Exterior2nd #####

#######################################

'''Impute "Other" for Exterior1st and Exterior2nd'''

all_data.Exterior1st = all_data.Exterior1st.fillna("Other")

all_data.Exterior2nd = all_data.Exterior2nd.fillna("Other")





######################

##### Electrical #####

######################

'''Only one missing value. Impute the mode.'''

all_data.Electrical = all_data.Electrical.fillna(all_data.Electrical.mode()[0])





####################################

##### Garage-related variables #####

####################################



# 1. First step: Impute "NoGarage" for all garage-related categorical variables where GarageArea == 0.'''

all_data.ix[all_data.GarageArea == 0, ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']] = 'None'



# 2. Assume there is no garage for the houses where GarageArea is null.'''

all_data.ix[all_data.GarageArea.isnull(), ['GarageType', 'GarageYrBlt', 'GarageFinish', 

                                       'GarageQual', 'GarageCond']] = 'None'

all_data.ix[all_data.GarageArea.isnull(), ['GarageArea', 'GarageCars']] = 0



# 3. One more observation: Use the YearBuilt for GarageYrBlt and the mode for the rest.'''

all_data.ix[all_data.GarageYrBlt.isnull(), 'GarageYrBlt'] = all_data[all_data.GarageYrBlt.isnull()].YearBuilt

all_data.GarageFinish = all_data.GarageFinish.fillna(all_data.GarageFinish.mode()[0])

all_data.GarageQual = all_data.GarageQual.fillna(all_data.GarageQual.mode()[0])

all_data.GarageCond = all_data.GarageCond.fillna(all_data.GarageCond.mode()[0])





######################################

##### Basement-related variables #####

######################################



# 1. Impute "NoBsmt" for the above variables when "TotalBsmtSF" is 0

bsmt_var_cat = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

all_data.ix[all_data.TotalBsmtSF == 0, bsmt_var_cat] = 'None'

all_data.ix[all_data.TotalBsmtSF == 0, ['BsmtHalfBath', 'BsmtFullBath']] = 0.0



# 2. Assume that there is no basement if TotalBsmtSF is null

all_data.ix[all_data.TotalBsmtSF.isnull(), bsmt_var_cat] = 'None'

all_data.ix[all_data.TotalBsmtSF.isnull(), ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtHalfBath', 'BsmtFullBath']] = 0.0



# 3. Impute the mode for the remaining missing values in the basement-related categorical variables.

all_data.ix[all_data.BsmtExposure.isnull(), 'BsmtExposure'] = all_data.BsmtExposure.mode()[0]

all_data.ix[all_data.BsmtQual.isnull(), 'BsmtQual'] = all_data.BsmtQual.mode()[0]

all_data.ix[all_data.BsmtCond.isnull(), 'BsmtCond'] = all_data.BsmtCond.mode()[0]

all_data.ix[all_data.BsmtFinType2.isnull(), 'BsmtFinType2'] = all_data.BsmtFinType2.mode()[0]





############################

##### MasVnr variables #####

############################



# 1. When the MasVnrType is null and the MasVnr Area is not null, impute the mode.

all_data.ix[(all_data.MasVnrType.isnull()) & (all_data.MasVnrArea.notnull()), 'MasVnrType'] = all_data.MasVnrType.mode()[0]



# 2. When both variables are null, impute "None" for MasVnr and 0.0 for MasVnrArea

all_data.MasVnrType = all_data.MasVnrType.fillna('None')

all_data.MasVnrArea = all_data.MasVnrArea.fillna(0.0)





####################

##### MSZoning #####

####################

'''Impute the mode for MSZoning whenever the data is missing.'''

all_data.MSZoning = all_data.MSZoning.fillna(all_data.MSZoning.mode()[0])





#######################

##### LotFrontage #####

#######################



Neighborhood_g = all_data['LotFrontage'].groupby(train['Neighborhood'])



for key,group in Neighborhood_g:

    # find where we are both simultaneously missing values and where the key exists

    LotFrontageNulls = all_data['LotFrontage'].isnull() & (all_data['Neighborhood'] == key)

    # fill in those blanks with the median of the key's group object

    all_data.loc[LotFrontageNulls,'LotFrontage'] = group.median()
# Create a feature for total square footage

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']



# Create a feature for GrLivArea squared

all_data['GrLivArea2'] = all_data['GrLivArea'] ** 2



# Create a feature for OverallQual sqared

all_data['OverallQual2'] = all_data['OverallQual'] ** 2



# Create a feature for age when sold

all_data['AgeWhenSold'] = all_data['YrSold'].astype(int) - all_data['YearBuilt']



# Binary variable indicating whether or not the house was remodeled

all_data["Remodeled"] = (all_data["YearRemodAdd"] != all_data["YearBuilt"]) * 1



# Drop GarageYrBlt

all_data = all_data.drop(['GarageYrBlt'], axis=1)
# MSSubClass

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)



# OverallCond

all_data['OverallCond'] = all_data['OverallCond'].astype(str)



# YrSold and MoSold

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
# Select columns that will be converted to numeric

cols = ( 

        

    # Basement variables

    'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure',

        

    # Garage variables

    'GarageQual', 'GarageCond', 'GarageFinish',

        

    # Other quality and condition variables

    'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'FireplaceQu', 'OverallCond', 

    

    # Other 

    'Functional', 'Fence',   'LandSlope', 'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass' 

)





# Apply the LabelEncoder to the categorical variables listed above

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values))

    all_data[c] = lbl.transform(list(all_data[c].values))
numeric_features = all_data.dtypes[all_data.dtypes != "object"].index

skewed_features = all_data[numeric_features].skew().sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_features})

skewed = skewness.loc[(skewness.Skew > 0.5) | (skewness.Skew < -0.5)]

skewed_features = skewed.index



lam = 0.15  # Value that worked best during cross validation.

for feat in skewed_features:

    #all_data[feat] = np.log1p(all_data[feat])   # Tried this, but it did not perform as well in cross validation.

    all_data[feat] = boxcox1p(all_data[feat], lam)
all_data = pd.get_dummies(all_data)
print ("Columns in final data set: {}".format(all_data.shape[1]))
# Split the train and test sets again

train = all_data[:ntrain]

test = all_data[ntrain:]



# Training data set with all numeric features

X = train.select_dtypes(include=[np.number]).drop('MSZoning_C (all)', axis=1)



# Remove MSZoning_C (all) from the test data as well

X_test = test.select_dtypes(include=[np.number]).drop('MSZoning_C (all)', axis=1)
# Define scoring function with RMSE for 5-fold cross-validation

def rmsle_CV(model):

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = 5))

    return rmse
# Define an wide array of alphas to test during cross-validation

ALPHAS = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 3, 5, 10, 30, 60, 100]



# Run cross-validation using RidgeCV

ridge = RidgeCV(alphas = ALPHAS, cv = 5)

ridge.fit(X, y_train)

alpha = ridge.alpha_



# What is the best alpha?

print ("Best alpha:", alpha)
# Try get more precision for alpha by re-running CV with alphas centered around the best alpha from above.

print ("Re-run CV with alphas centered around " + str(alpha))

scalars = np.arange(0.6, 1.45, 0.05)

ALPHAS = [alpha * i for i in scalars]

ridge = RidgeCV(alphas = ALPHAS, cv = 5)

ridge.fit(X, y_train)

alpha_final = ridge.alpha_

print ("Alpha selected through cross validation:", alpha_final)
# Create final ridge regression model with selected alpha and RobustScalar

ridge = make_pipeline(RobustScaler(), Ridge(alpha=alpha_final))
# How well does this model perform during cross validation?

score = rmsle_CV(ridge)

print ("The RMSE of the final Ridge regression during cross-validation is {:.4f}".format(score.mean()))
# Define possible alphas for Linear Regression with Lasso regularization

ALPHAS = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1]

lasso = LassoCV(alphas = ALPHAS, max_iter = 50000, cv = 5)

lasso.fit(X, y_train)

alpha = lasso.alpha_

print ("Best alpha:", alpha)
print ("Re-run CV with alphas centered around " + str(alpha))

scalars = np.arange(0.6, 1.45, 0.05)

ALPHAS = [alpha * i for i in scalars]

lasso = LassoCV(alphas = ALPHAS, max_iter = 50000, cv = 5)

lasso.fit(X, y_train)

alpha_final = lasso.alpha_

print ("Alpha selected through cross validation:", alpha_final)
# Fit a lasso model with optimized alpha and RobustScalar()

lasso = make_pipeline(RobustScaler(), Lasso(alpha = alpha_final))

score = rmsle_CV(lasso)

print ("The RMSE of the final Lasso regression during cross-validation is {:.4f}".format(score.mean()))
L1_RATIOS = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1]

ALPHAS = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6]

elasticNet = ElasticNetCV(l1_ratio = L1_RATIOS,

                          alphas = ALPHAS, 

                          max_iter = 50000, cv = 5)

elasticNet.fit(X, y_train)

alpha = elasticNet.alpha_

ratio = elasticNet.l1_ratio_

print ("Best l1_ratio :", ratio)

print ("Best alpha :", alpha)
print ("Re-run the cross-validation with the l1_ratio centered around " + str(ratio))

scalars = np.arange(0.8, 1.25, 0.05)

L1_RATIOS = [ratio * i for i in scalars]

elasticNet = ElasticNetCV(l1_ratio = L1_RATIOS,

                              alphas = ALPHAS,

                              max_iter = 50000, cv = 5)

elasticNet.fit(X, y_train)

   

alpha = elasticNet.alpha_

ratio_final = round(elasticNet.l1_ratio_, 2)

print ("Best l1_ratio :", ratio_final)

print ("Best alpha :", alpha)
print ("Re-run again for more precision on alpha, with l1_ratio fixed at " + str(ratio_final) + " and alpha centered around " + str(alpha))



scalars = np.arange(0.6, 1.45, 0.05)

alphas = [alpha * i for i in scalars]

elasticNet = ElasticNetCV(l1_ratio = ratio_final,

                          alphas = ALPHAS, 

                          max_iter = 50000, cv = 5)

elasticNet.fit(X, y_train)

   

alpha_final = elasticNet.alpha_

ratio_final = elasticNet.l1_ratio_

print ("Best l1_ratio :", ratio_final)

print ("Best alpha :", alpha_final)
# Fit a final elastic net model with the selected paramters and RobustScalar()

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=alpha_final, l1_ratio=ratio_final))

score = rmsle_CV(ENet)

print ("The RMSE of the final Elastic Net model during cross-validation is {:.4f}".format(score.mean()))

rfr = RandomForestRegressor()



param_grid = {

                'n_estimators' : [100, 500, 1000],

                'max_features' : ['auto', 'sqrt', 'log2'],

                'min_samples_leaf' : [50, 100]

              }



CV_rfr = GridSearchCV(estimator=rfr, param_grid=param_grid, cv= 5)

CV_rfr.fit(X, y_train)



print ("Below are the parameters that were selected through cross-validation:")

print ("max_features : {}".format(CV_rfr.best_params_['max_features']))

print ("n_estimators : {}".format(CV_rfr.best_params_['n_estimators']))

print ("min_samples_leaf : {}".format(CV_rfr.best_params_['min_samples_leaf']))
# Fit a random forest model with 

rf = make_pipeline(

                    RobustScaler(), 

                    RandomForestRegressor(

                                            max_features = CV_rfr.best_params_['max_features'],

                                            n_estimators = CV_rfr.best_params_['n_estimators'],

                                            min_samples_leaf = CV_rfr.best_params_['min_samples_leaf']

                                         )

                  )

score = rmsle_CV(rf)

print ("The RMSE of the final random forest model during cross-validation is {:.4f}".format(score.mean()))
# Select a wide array of parameters for max_depth & min_samples_leaf

params = {'max_depth' : [2, 10, 20, 50], 'min_samples_leaf' : [5, 10, 25, 50, 75, 100]}



# Run the GridSearchCV function on these arrays

gs = GridSearchCV(estimator=GradientBoostingRegressor(

                                                        # Place holder values - will be tuned later

                                                        learning_rate = 0.05,

                                                        n_estimators = 3000,

                                                        max_features = 'sqrt',

                                                        subsample = 0.8,

                                                        random_state = 23

                                                    ),

                 param_grid = params)



gs.fit(X,y_train)







# What are the best performing parameters?

print ("Best values after first iteration:")

print ("max_depth:", gs.best_params_['max_depth'])

print ("min_samples_leaf:", gs.best_params_['min_samples_leaf'])
# Update params array to center around those selected in previous step

params = {'max_depth' : np.arange(1, 9), 'min_samples_leaf' : np.arange(1, 9)}



# Re-run the grid search cross-validation with new parameters

gs = GridSearchCV(estimator=GradientBoostingRegressor(

                                                        # Place holder values - will be tuned later

                                                        learning_rate = 0.05,

                                                        n_estimators = 3000,

                                                        max_features = 'sqrt',

                                                        subsample = 0.8,

                                                        random_state = 23

                                                    ),

                 param_grid = params)

gs.fit(X,y_train)



# Store values in variable

max_depth_ = gs.best_params_['max_depth']

min_samples_leaf_ = gs.best_params_['min_samples_leaf']



# What are the final parameters that were selected?

print ("Final selection for max_depth:", max_depth_)

print ("Final selection for min_samples_leaf:", min_samples_leaf_)
# Update params array

params = {'min_samples_split' : [4, 10, 25, 50, 100, 200]}



# Run the grid search cross-validation with new parameters

gs = GridSearchCV(estimator=GradientBoostingRegressor(

                                                        learning_rate = 0.05,

                                                        n_estimators = 3000,

                                                        max_features = 'sqrt',

                                                        subsample = 0.8,

                                                        random_state = 23,

    

                                                        # Selected from CV

                                                        max_depth = max_depth_, 

                                                        min_samples_leaf = min_samples_leaf_ 

                                                    ), 

                 param_grid = params)

gs.fit(X,y_train)







# What are the final parameters that were selected?

print ("Best value for min_samples_split after first iteration:", gs.best_params_['min_samples_split'])
# Update params array

params = {'min_samples_split' : np.arange(2,8)}



# Run the grid search cross-validation with new parameters

gs = GridSearchCV(estimator=GradientBoostingRegressor(

                                                        learning_rate = 0.05,

                                                        n_estimators = 3000,

                                                        max_features = 'sqrt',

                                                        subsample = 0.8,

                                                        random_state = 23,

    

                                                        # Selected from CV

                                                        max_depth = max_depth_,

                                                        min_samples_leaf = min_samples_leaf_

                                                    ),

                 param_grid = params)

gs.fit(X,y_train)



# Store selected value in variable

min_samples_split_ = gs.best_params_['min_samples_split']



# What are the final parameters that were selected?

print ("Final selection for min_samples_split:", min_samples_split_)
params = {'subsample' : np.arange(0.4, 1, .1)}



# Run the grid search cross-validation with new parameters

gs = GridSearchCV(estimator=GradientBoostingRegressor(

                                                        # Place holders

                                                        learning_rate = 0.05,

                                                        n_estimators = 3000,

                                                        max_features = 'sqrt',

                                                        random_state = 23,

                                                        

                                                        # Selected from CV

                                                        max_depth = max_depth_, 

                                                        min_samples_leaf = min_samples_leaf_, 

                                                        min_samples_split = min_samples_split_ 

                                                    ),

                 param_grid = params)

gs.fit(X,y_train)



# Store the best performer in a variable

subsample_ = round(gs.best_params_['subsample'], 1)



# What are the final parameters that were selected?

print ("Final selection for subsample:", subsample_)
params = {'n_estimators' : [100, 500, 1000, 3000, 5000, 8000]}



# Run the grid search cross-validation with new parameters

gs = GridSearchCV(estimator=GradientBoostingRegressor(

                                                        learning_rate = 0.05,

                                                        max_features = 'sqrt',

                                                        random_state = 23,

                                                        

                                                        # Selected from CV

                                                        max_depth = max_depth_, 

                                                        min_samples_leaf = min_samples_leaf_, 

                                                        min_samples_split = min_samples_split_, 

                                                        subsample = subsample_ 

                                                    ),

                 param_grid = params)

gs.fit(X,y_train)



# Store the best performing value in a variables

n_estimators_ = gs.best_params_['n_estimators']





# What are the final parameters that were selected?

print ("Final selection for n_estimators:", n_estimators_)
params = {'learning_rate' : np.arange(0.01, 0.2, 0.01)}



# Run the grid search cross-validation with new parameters

gs = GridSearchCV(estimator=GradientBoostingRegressor(

                                                        max_features = 'sqrt',

                                                        random_state = 23,

                                                        

                                                        # Selected from CV

                                                        max_depth = max_depth_,

                                                        min_samples_leaf = min_samples_leaf_,

                                                        min_samples_split = min_samples_split_,

                                                        subsample = subsample_, 

                                                        n_estimators = n_estimators_

                                                    ),

                 param_grid = params)

gs.fit(X,y_train)



learning_rate_ = round(gs.best_params_['learning_rate'],2)



# What are the final parameters that were selected?

print ("Best learning_rate:", learning_rate_)
gradient_boosting = GradientBoostingRegressor(

                                                n_estimators = n_estimators_, 

                                                learning_rate = learning_rate_,

                                                max_depth = max_depth_, 

                                                max_features = 'sqrt', 

                                                subsample = subsample_,

                                                min_samples_leaf = min_samples_leaf_, 

                                                min_samples_split= min_samples_split_, 

                                                loss='huber', 

                                                random_state=23

                                            )



score = rmsle_CV(gradient_boosting)

print ("\nRMSE: {:.4f}".format(score.mean()))

print ("SD: {:.4f}\n".format(score.std()))
params = {'max_depth' : np.arange(2,12,2), 'min_child_weight' : np.arange(0.5,5,0.5)}



model_xgb = xgb.XGBRegressor(                                

                             # Place holder values - will be tuned

                             learning_rate =0.1, 

                             n_estimators=1000,

                             gamma=0, 

                             subsample=0.5, 

                             colsample_bytree=0.8

                            )



# What are the best values for max_depth & min_child_weight?

gs = GridSearchCV(estimator = model_xgb, param_grid = params)

gs.fit(X, y_train)



# Store the values in variables

max_depth_ = gs.best_params_['max_depth']

min_child_weight_ = gs.best_params_['min_child_weight']



print ("Best values after first iteration:")

print ("max_depth:", max_depth_)

print ("min_child_weight:", min_child_weight_)
params = {'gamma' : np.arange(0,0.5,0.1)}



model_xgb = xgb.XGBRegressor(

                             # Place holder values

                             n_estimators=1000, 

                             learning_rate=0.1,

                             subsample = 0.5,

                             colsample_bytree=0.8,



                             # Selected from CV

                             max_depth=max_depth_,

                             min_child_weight=min_child_weight_

                            )





# Fit the model and run GridSearchCV

gs = GridSearchCV(estimator = model_xgb, param_grid = params)

gs.fit(X, y_train)



# Store value in variable

gamma_ = gs.best_params_['gamma']



# What's the best value?

print ("Best value for gamma:", gamma_)
params = {'subsample' : np.arange(0.6,1.0,0.1), 'colsample_bytree' : np.arange(0.6, 1.0, 0.1)}



model_xgb = xgb.XGBRegressor(

                             # Place holder values

                             n_estimators=1000, 

                             learning_rate=0.1,



                             # Selected using CV

                             max_depth=max_depth_,

                             min_child_weight=min_child_weight_,

                             gamma=gamma_

                            )



# Fit the model and run GridSearchCV

gs = GridSearchCV(estimator = model_xgb, param_grid = params)

gs.fit(X, y_train)



# Store values in variables

subsample_ = gs.best_params_['subsample']

colsample_bytree_ = gs.best_params_['colsample_bytree']



# What are the best values?

print ("Best value for subsample:", subsample_)

print ("Best value for colsample_bytree_:", colsample_bytree_)
params = {'learning_rate' : np.arange(0.01,0.3,0.01)}



model_xgb = xgb.XGBRegressor(

                             # Place holder for values

                             n_estimators=1000, 

 

                             # Selected using CV

                             max_depth=max_depth_,

                             min_child_weight=min_child_weight_,

                             gamma=gamma_,

                             subsample = subsample_,

                             colsample_bytree = colsample_bytree_

                            )



# Fit the model and run GridSearchCV

gs = GridSearchCV(estimator = model_xgb, param_grid = params)

gs.fit(X, y_train)



# What is the best value?

learning_rate_ =  gs.best_params_['learning_rate']

print ("Best value for learning_rate:", learning_rate_)
params = {'n_estimators' : np.arange(500, 4500, 500)}



model_xgb = xgb.XGBRegressor(

                             # Selected using CV

                             max_depth=max_depth_,

                             min_child_weight=min_child_weight_,

                             gamma=gamma_,

                             subsample=subsample_,

                             colsample_bytree=colsample_bytree_,

                             learning_rate=learning_rate_

                            )



# What is the best value for n_estimators after the first iteration

gs = GridSearchCV(estimator = model_xgb, param_grid = params)

gs.fit(X, y_train)

print ("Best value for n_estimators after first iteration:", gs.best_params_['n_estimators'])
params = {'n_estimators' : np.arange(750, 1000, 1350)}



model_xgb = xgb.XGBRegressor( 

                             # Selected using CV

                             max_depth=max_depth_,

                             min_child_weight=min_child_weight_,

                             gamma=gamma_,

                             subsample=subsample_,

                             colsample_bytree=colsample_bytree_,

                             learning_rate=learning_rate_

                            )





# Fit the model and run GridSearchCV

gs = GridSearchCV(estimator = model_xgb, param_grid = params)

gs.fit(X, y_train)



# Store value in variable

n_estimators_ =  gs.best_params_['n_estimators']



# What is the final selection for n_estimators?

print ("Final selection for n_estimators:", n_estimators_)
params = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}

    



model_xgb = xgb.XGBRegressor( 

                             # Selected using CV

                             max_depth=max_depth_,

                             min_child_weight=min_child_weight_,

                             gamma=gamma_,

                             subsample=subsample_,

                             colsample_bytree=colsample_bytree_,

                             learning_rate=learning_rate_,

                             n_estimators=n_estimators_

                            )





# What is the best value for n_estimators after the first iteration

gs = GridSearchCV(estimator = model_xgb, param_grid = params)

gs.fit(X, y_train)

print ("Best value after first iteration:", gs.best_params_['reg_alpha'])
params = {'reg_alpha':[0.001, 0.01, 0.025, 0.05, 0.075]}



model_xgb = xgb.XGBRegressor(  

                             # Selected using CV

                             max_depth=max_depth_,

                             min_child_weight=min_child_weight_,

                             gamma=gamma_,

                             subsample=subsample_,

                             colsample_bytree=colsample_bytree_,

                             learning_rate=learning_rate_,

                             n_estimators=n_estimators_

                            )





# Fit the model and run GridSearchCV

gs = GridSearchCV(estimator = model_xgb, param_grid = params)

gs.fit(X, y_train)



# Store value in variable

reg_alpha_ = gs.best_params_['reg_alpha']



# What is the final selection for reg_alpha?

print ("Final selection for reg_alpha:", reg_alpha_)
model_xgb = xgb.XGBRegressor(

                             # Selected using CV

                             max_depth=max_depth_,

                             min_child_weight=min_child_weight_,

                             gamma=gamma_,

                             subsample=subsample_,

                             colsample_bytree=colsample_bytree_,

                             learning_rate=learning_rate_,

                             n_estimators=n_estimators_,

                             reg_alpha = reg_alpha_

                            )

score = rmsle_CV(model_xgb)

print ("\nRMSE: {:.4f}".format(score.mean()))

print ("SD: {:.4f}\n".format(score.std()))
# Train models on the full training data

ridge_mod = ridge.fit(X, y_train)

lasso_mod = lasso.fit(X, y_train)

enet_mod = ENet.fit(X, y_train)

rf_mod = rf.fit(X, y_train)

gb_mod = gradient_boosting.fit(X, y_train)

xgb_mod = model_xgb.fit(X, y_train)



# Predict on the testing (submission) data

ridge_predict = np.expm1(ridge_mod.predict(X_test))

lasso_predict = np.expm1(lasso_mod.predict(X_test))

enet_predict = np.expm1(enet_mod.predict(X_test))

rf_predict = np.expm1(rf_mod.predict(X_test))

gb_predict = np.expm1(gb_mod.predict(X_test))

xgb_predict = np.expm1(xgb_mod.predict(X_test))
### Ridge - 0.11795

submission = pd.DataFrame()

submission['Id'] = test_ID

submission['SalePrice'] = ridge_predict

submission.to_csv('ridge.csv', index=False)



### Lasso - 0.11834

submission = pd.DataFrame()

submission['Id'] = test_ID

submission['SalePrice'] = lasso_predict

submission.to_csv('lasso.csv', index=False)



### Elastic Net - 0.11832

submission = pd.DataFrame()

submission['Id'] = test_ID

submission['SalePrice'] = enet_predict

submission.to_csv('elastic_net.csv', index=False)



### Random Forest

submission = pd.DataFrame()

submission['Id'] = test_ID

submission['SalePrice'] = rf_predict

submission.to_csv('random_forest.csv', index=False)



### Gradient Boosting - 0.12426

submission = pd.DataFrame()

submission['Id'] = test_ID

submission['SalePrice'] = gb_predict

submission.to_csv('gradient_boosting.csv', index=False)



### XGBoost - 0.12029

submission = pd.DataFrame()

submission['Id'] = test_ID

submission['SalePrice'] = xgb_predict

submission.to_csv('xgb.csv', index=False)

# Compare predictions. How correlated are they?

results = {

                

                'gradient_boosting' : gb_predict,

                'ridge' : ridge_predict,

                'lasso' : lasso_predict,

                'enet' : enet_predict,

                'xgb' : xgb_predict

          }



# Create results data frame

results_df = pd.DataFrame(results)



results_df.corr()
from sklearn.model_selection import KFold



def stackModels(base_models, meta_model):

    

    # Initialize arrays

    base_models_ = [list() for x in base_models]

    out_of_fold_predictions = np.zeros((X.shape[0], len(base_models)))

    kfold = KFold(n_splits=5, shuffle=True, random_state=156)

    

    # Generate predictions on hold out set

    for i, model in enumerate(base_models):

        for train_index, holdout_index in kfold.split(X, y_train):

            model.fit(X.ix[train_index], y_train[train_index])

            y_pred = model.predict(X.ix[holdout_index])

            out_of_fold_predictions[holdout_index, i] = y_pred

    

    # Fit the meta model with the predictions from the base models

    meta_model.fit(out_of_fold_predictions, y_train)

    

    # Create features for meta model

    meta_features = np.column_stack([

        np.column_stack([model.predict(X_test) for model in base_models]).mean(axis=1)

            for base_models_ in base_models_ ])

    

    # Generate predictions

    final_predictions = np.expm1(meta_model.predict(meta_features))

    

    # Return predictions

    return final_predictions
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

final_predictions = stackModels(base_models = (ridge, ENet, lasso), meta_model=lr) 
# Official submisssion file for stacking - 0.11940

submission = pd.DataFrame()

submission['Id'] = test_ID

submission['SalePrice'] = final_predictions

submission.to_csv('stacking_submit.csv', index=False)
#final_predictions = (ridge_predict + lasso_predict + enet_predict)/3 # 0.11803

#final_predictions = (ridge_predict  + xgb_predict)/2 # 0.11549

#final_predictions = .75*ridge_predict  + .25*xgb_predict # 0.11583

#final_predictions = .5*ridge_predict  + .25*xgb_predict + 0.25*enet_predict # 0.11586

final_predictions = .6*ridge_predict  + .4*xgb_predict # .11540

#final_predictions = .65*ridge_predict  + .35*xgb_predict # .11547
# Official submisssion file

submission = pd.DataFrame()

submission['Id'] = test_ID

submission['SalePrice'] = final_predictions

submission.to_csv('ridge_plus_xgb.csv', index=False)