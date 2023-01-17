

# Project packages.

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

import numpy as np



# Visualisations.

import matplotlib.pyplot as plt 

import seaborn as sns



# Statistics.

from scipy import stats

from scipy.stats import norm, skew

from statistics import mode

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



# Machine Learning.

from sklearn.linear_model import Lasso, Ridge

from sklearn import metrics

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression



# Filter out warnings when fitting.

import sys

import warnings



if not sys.warnoptions:

    warnings.simplefilter("ignore")
#Dictionary containing the descriptions of all columns in the data set.

desc = {

 'SalePrice': "The property's sale price in dollars. This is the target variable to be predicted",

 'MSSubClass': "The building class",

 'MSZoning': "The general zoning classification",

 'LotFrontage': "Linear feet of street connected to property",

 'LotArea': "Lot size in square feet",

 'Street': "Type of road access",

 'Alley': "Type of alley access",

 'LotShape': "General shape of property",

 'LandContour': "Flatness of the property",

 'Utilities': "Type of utilities available",

 'LotConfig': "Lot configuration",

 'LandSlope': "Slope of property",

 'Neighborhood': "Physical locations within Ames city limits",

 'Condition1': "Proximity to main road or railroad",

 'Condition2': "Proximity to main road or railroad (if a second is present)",

 'BldgType': "Type of dwelling",

 'HouseStyle': "Style of dwelling",

 'OverallQual': "Overall material and finish quality",

 'OverallCond': "Overall condition rating",

 'YearBuilt': "Original construction date",

 'YearRemodAdd': "Remodel date",

 'RoofStyle': "Type of roof",

 'RoofMatl': "Roof material",

 'Exterior1st': "Exterior covering on house",

 'Exterior2nd': "Exterior covering on house (if more than one material)",

 'MasVnrType': "Masonry veneer type",

 'MasVnrArea': "Masonry veneer area in square feet",

 'ExterQual': "Exterior material quality",

 'ExterCond': "Present condition of the material on the exterior",

 'Foundation': "Type of foundation",

 'BsmtQual': "Height of the basement",

 'BsmtCond': "General condition of the basement",

 'BsmtExposure': "Walkout or garden level basement walls",

 'BsmtFinType1': "Quality of basement finished area",

 'BsmtFinSF1': "Type 1 finished square feet",

 'BsmtFinType2': "Quality of second finished area (if present)",

 'BsmtFinSF2': "Type 2 finished square feet",

 'BsmtUnfSF': "Unfinished square feet of basement area",

 'TotalBsmtSF': "Total square feet of basement area",

 'Heating': "Type of heating",

 'HeatingQC': "Heating quality and condition",

 'CentralAir': "Central air conditioning",

 'Electrical':" Electrical system",

 '1stFlrSF': "First Floor square feet",

 '2ndFlrSF': "Second floor square feet",

 'LowQualFinSF': "Low quality finished square feet (all floors)",

 'GrLivArea': "Above grade (ground) living area square feet",

 'BsmtFullBath': "Basement full bathrooms",

 'BsmtHalfBath': "Basement half bathrooms",

 'FullBath': "Full bathrooms above grade",

 'HalfBath': "Half baths above grade",

 'Bedroom': "Number of bedrooms above basement level",

 'Kitchen': "Number of kitchens",

 'KitchenQual': "Kitchen quality",

 'TotRmsAbvGrd': "Total rooms above grade (does not include bathrooms)",

 'Functional': "Home functionality rating",

 'Fireplaces': "Number of fireplaces",

 'FireplaceQu': "Fireplace quality",

 'GarageType': "Garage location",

 'GarageYrBlt': "Year garage was built",

 'GarageFinish': "Interior finish of the garage",

 'GarageCars': "Size of garage in car capacity",

 'GarageArea': "Size of garage in square feet",

 'GarageQual': "Garage quality",

 'GarageCond': "Garage condition",

 'PavedDrive': "Paved driveway",

 'WoodDeckSF': "Wood deck area in square feet",

 'OpenPorchSF': "Open porch area in square feet",

 'EnclosedPorch': "Enclosed porch area in square feet",

 '3SsnPorch': "Three season porch area in square feet",

 'ScreenPorch': "Screen porch area in square feet",

 'PoolArea': "Pool area in square feet",

 'PoolQC': "Pool quality",

 'Fence': "Fence quality",

 'MiscFeature': "Miscellaneous feature not covered in other categories",

 'MiscVal': "Value of miscellaneous feature",

 'MoSold': "Month Sold",

 'YrSold': "Year Sold",

 'SaleType': "Type of sale",

 'SaleCondition': "Condition of sale"

}
#Reading from and Dumping to outfile for description dictionary

import json

json.dump(desc, open("desc.json",'w'))



with open("desc.json", "r") as read_file:

    d= json.load(read_file)
"""

Custom function that yields a dictionary of missing data in a DataFrame 

"""

def percent_missing(df):    

    cols = list(df.columns)

    outputDict = {}

    for x in range(len(cols)):

        key = cols[x]

        if (df[cols[x]].isnull().sum()) > 0:

            outputDict[key] = round(((df[cols[x]].isnull().sum()) / len(df)*100),2)

    return outputDict



"""

Custom function that yields an overview of missing data in a DataFrame or a list 

of column names with missing data based on data type (Dependant on 'return_' parameter input, 'N'-numerical, 'C'-categorical)

"""

def percent_missing_overview(df,return_ = 'None'):

    

    

    numerical = [x for x in df.columns if df.dtypes[x] != 'object']

    categorical = [x for x in df.columns if df.dtypes[x] == 'object']

    missing_keys = percent_missing(df).keys()

    

    if return_ == 'None':

        #numerical = [x for x in df.columns if df.dtypes[x] != 'object']

        #categorical = [x for x in df.columns if df.dtypes[x] == 'object']

        print("---")

        #missing_keys = list(percent_missing(train).keys())

        print("Missing Data:")

        print("---")

        print("Numerical : ","(",len([x for x in numerical if x in missing_keys]),")")

        for x in numerical:

            if x in missing_keys:

                print(" (",percent_missing(df)[x],"%)",x,":",desc[x])

        print("---")

        print("Categorical : ","(",len([x for x in categorical if x in missing_keys]),")")

        for x in categorical:

            if x in missing_keys:

                print(" (",percent_missing(df)[x],"%)",x,":",desc[x])

        print("---")        

    if return_ == 'N':

        return [x for x in numerical if x in missing_keys]

    if return_ == 'C':

        return [x for x in categorical if x in missing_keys]
#Prints out a bried overview of the shape and column data types of a DataFrame

def data_overview(df):

    print("---")

    print("Data Overview")

    print("---")

    numerical = [x for x in df.columns if df.dtypes[x] != 'object']

    print("There are" , len(numerical) , "numerical features")

    categorical = [x for x in df.columns if df.dtypes[x] == 'object']

    print("There are" , len(categorical) , "categorical features")

    print("---")

    print("Shape : ")

    print(df.shape)
#Reading in Train and Test data from CSV files.

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#The ID column is retained in temporary variables before being dropped.

train_ID = train['Id']

test_ID = test['Id']

train.drop(['Id'], axis=1, inplace=True)

test.drop(['Id'], axis=1, inplace=True)



#Lists of column names of predictor variables whether numerical or categorical

numerical = [x for x in train.columns if train.dtypes[x] != 'object']

#Dropping the target variable as it is not a predictor

numerical.remove('SalePrice')

categorical = [x for x in train.columns if train.dtypes[x] == 'object']



#Retaining the predictor variable before dropping.

y = train.SalePrice.reset_index(drop=True)



#Creating variables to hold our features before EDA.

train_features = train.drop(['SalePrice'], axis=1)

test_features = test



#Combining features from test and train set to perform uniform preprocessing later.

features = pd.concat([train_features, test_features]).reset_index(drop=True)



data_overview(features)
# Checking for outliers in GrLivArea as indicated in dataset documentation

plt.figure(figsize=(10,8))

ax = sns.regplot(x=train['GrLivArea'], y=train['SalePrice'], fit_reg=True)

plt.show()
# Removing two very extreme outliers in the bottom right hand corner

features = features.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

train.reset_index(drop=True, inplace=True)



# Re-check graph

plt.figure(figsize=(10,8))

ax = sns.regplot(x=train['GrLivArea'], y=train['SalePrice'], fit_reg=True)

plt.show()
#Acquiring mu and sigma for normal distribution plot of 'SalePrice'

(mu, sigma) = norm.fit(train['SalePrice'])



#Plotting distribution plot of 'SalePrice'

plt.figure(figsize=(8,8))

ax = sns.distplot(train['SalePrice'] , fit=norm);

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')
#Plotting Q-Q plot

plt.figure(figsize=(8,8))

ax = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
fig, ax = plt.subplots(figsize=(16,16), ncols=2, nrows=2)



sns.distplot(y, kde=False,color = 'green', ax=ax[0][0], fit=stats.norm)

sns.distplot(y, kde=False,color = 'green', ax=ax[0][1], fit=stats.johnsonsu)

sns.distplot(y, kde=False,color = 'green', ax=ax[1][0], fit=stats.lognorm)

sns.distplot(y, kde=False,color = 'green', ax=ax[1][1], fit=stats.johnsonsb)



ax[0][0].set_title("Normal",fontsize=24)

ax[0][1].set_title("Johnson SU",fontsize=24)

ax[1][0].set_title("Log Normal",fontsize=24)

ax[1][1].set_title("Johnson SB",fontsize=24)
# Applying a log(1+x) transformation to SalePrice

train["SalePrice"] = np.log1p(train["SalePrice"])

y = train.SalePrice.reset_index(drop=True)
#Plotting distribution plot of 'SalePrice'

plt.figure(figsize=(8,8))

ax = sns.distplot(train['SalePrice'] , fit=stats.johnsonsu);

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

plt.legend(['Johnson SU'],

            loc='best')
#Plotting Q-Q plot

plt.figure(figsize=(8,8))

ax = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
# Creating a correlation matrix to plot.

corr = train.corr()

# Creating a mask to filter out unnecessary correlations

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



plt.figure(figsize=(16, 10))

plt.title('Correlation Matrix', fontsize=18)

# Plotting the correlation matrix

sns.heatmap(corr, mask=mask, annot=False,cmap='RdYlGn', linewidths=0.4, annot_kws={'size':20})

plt.show()
# show higher correlations between predictor variables and the response variable

corr_dict = train.corr()['SalePrice'][(train.corr()['SalePrice'] > 0.5) & (train.corr()['SalePrice'] < 1.0)].sort_values(ascending=False).to_dict()

print("Higher correlations between predictor variables and the response variable ")

print("-------------------------------------------------------------------------")

for x in corr_dict.keys():

    print("(" , round(corr_dict[x],3), ")",x, ":" , "  -",desc[x])
data_overview(features)
# Visualising missing data.

f, ax = plt.subplots(figsize = (10, 6))

plt.xticks(rotation = '90')

sns.barplot(x = list(percent_missing(features).keys()), y = list(percent_missing(features).values()))

plt.xlabel('Features', fontsize = 15)

plt.ylabel('Percentage of missing values (%)', fontsize = 15)

plt.title('Missing Data', fontsize = 15)



# Generate missing data report using a custom function.

percent_missing_overview(features) 
#Converting categorical predictors that are stored as numbers to strings

features['MSSubClass'] = features['MSSubClass'].apply(str)

features['YrSold'] = features['YrSold'].astype(str)

features['MoSold'] = features['MoSold'].astype(str)
# Imputing above mentioned categorical features to 'None'.

for f in ('MasVnrType','Alley','PoolQC', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',):

    features[f] = features[f].fillna('None')
# Imputing 'MiscFeature' to 'None'.

features['MiscFeature'] = features['MiscFeature'].fillna('None')
# Imputing remaining categorical features to mode.

for f in ('Exterior1st', 'Exterior2nd', 'SaleType','Utilities'):

    features[f] = features[f].fillna(features[f].mode()[0])

features['Functional'] = features['Functional'].fillna('Typ')

features['Electrical'] = features['Electrical'].fillna("SBrkr")

features['KitchenQual'] = features['KitchenQual'].fillna("TA")    

# Grouping property class features and imputing with the most frequent entry.

features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))



# Final imputation of categorical features

objects = []

for i in features.columns:

    if features[i].dtype == object:

        objects.append(i)



features.update(features[objects].fillna('None'))
features[percent_missing_overview(features,'N')].describe().T.round(3)
features[features['GarageYrBlt'] == 2207][['YearBuilt','YearRemodAdd','GarageYrBlt']]
# Changing entry from 2207 to 2007.

features.GarageYrBlt.iloc[2592] = 2007
# Imputing other Garage features to 0.

for f in ['GarageYrBlt', 'GarageArea', 'GarageCars']:

    features[f] = features[f].fillna(0)
# Imputing Basement features to 0.

for f in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:

    features[f] = features[f].fillna(0)
# Imputing 'MasVnrArea' feature to 0.

features['MasVnrArea'] = features['MasVnrArea'].fillna(0)
features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))
# Checking for missing data after imputation.

percent_missing_overview(features) 
# Recreating list of numerical features

numerical = [x for x in features.columns if features.dtypes[x] != 'object']

# Calculating skewness

skewed = features[numerical].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed})

skewness
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics2 = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics2.append(i)

skew_features = features[numerics2].apply(lambda x: stats.skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



for i in skew_index:

    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
# Creating simplified features

features['HasPool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

features['Has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

features['HasGarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

features['HasBsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

features['HasFireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']

features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']



features['Total_SQR_Footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +

                                 features['1stFlrSF'] + features['2ndFlrSF'])



features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +

                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))



features['Total_Porch_SF'] = (features['OpenPorchSF'] + features['3SsnPorch'] +

                              features['EnclosedPorch'] + features['ScreenPorch'] +

                              features['WoodDeckSF'])



features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)
final_features = pd.get_dummies(features).reset_index(drop=True)

print(final_features.shape)
# Splitting up features for training and prediction.

X = final_features.iloc[:len(y), :]

X_pred = final_features.iloc[len(X):, :]



X_train = X

X_test = X_pred

y_train = y



print('X', X.shape, 'y', y.shape, 'X_pred', X_pred.shape)
# RMSE scoring function with cross validation

def rmse_cv(model):

    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 10))

    return(rmse)
# Possible list of alpha values.

alphas = [5,10,15,20,25,30]



# Iterate over alpha's

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]

# Plot findings

cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation")

plt.xlabel("Alpha")

plt.ylabel("Rmse")
# Creating Ridge Regression Model with estimated alpha

model_ridge = Ridge(alpha = 10)
# Setting up list of alpha's

alphas = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]



# Iterate over alpha's

cv_lasso = [rmse_cv(Lasso(alpha = alpha)).mean() for alpha in alphas]



# Plot findings

cv_lasso = pd.Series(cv_lasso, index = alphas)

cv_lasso.plot(title = "Validation")

plt.xlabel("Alpha")

plt.ylabel("Rmse")
# Creating Lasso Regression Model with estimated alpha

model_lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0004))
# Calculating RMSE score with estimated alpha values

cv_ridge = rmse_cv(model_ridge).mean()

cv_lasso = rmse_cv(model_lasso).mean()
# Creating a table of results, ranked highest to lowest

results = pd.DataFrame({

    'Model': ['Ridge','Lasso'],

    'Score': [cv_ridge,cv_lasso]})



# Build dataframe of values

result_df = results.sort_values(by='Score', ascending=True).reset_index(drop=True)

result_df.head()
# Fitting models and calculating predictions.

model_lasso.fit(X_train, y_train)

lasso_pred = np.expm1(model_lasso.predict(X_test))



model_ridge.fit(X_train, y_train)

ridge_pred = np.expm1(model_ridge.predict(X_test))

#Stacking

Stack = (lasso_pred + ridge_pred) / 2
submission = pd.DataFrame({'Id':test_ID, 'SalePrice':lasso_pred})

submission.to_csv('Stack.csv', index=False)