# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import matplotlib.gridspec as gridspec

from scipy import stats

import matplotlib.style as style

import math



%matplotlib inline

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)



from sklearn.linear_model import LinearRegression

from sklearn import ensemble, tree, linear_model

from scipy.stats import skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

import missingno as msno



#Model Train

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold

from lightgbm import LGBMRegressor



# Feature Selection Techniques

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.preprocessing import MinMaxScaler



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.listdir('/kaggle/input/house-prices-advanced-regression-techniques')
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
train.shape
test.shape
def column_unique(col_list):

    for column_name in train.columns:

        if train[column_name].nunique() < 35 and train[column_name].dtypes == 'int64':

            unique_category = len(train[column_name].unique())

            print(f'Feature {column_name} with dtype discrete has {unique_category} unique categories')

        elif train[column_name].dtypes == 'object':

            unique_category = len(train[column_name].unique())

            print(f'Feature {column_name} with dtype object has {unique_category} unique categories')

        else:

            dtype = train[column_name].dtypes

            print(f'Feature {column_name} is of dtype {dtype}')
cat_cols = list(train.select_dtypes('object').columns)

dis_cols = [feature for feature in train.columns if train[feature].nunique() < 25 and train[feature].dtypes == 'int64' ]

num_cols = [feature for feature in train.columns if train[feature].nunique() > 25]
def missing_data(df):

    total = df.isnull().sum()

    percent = round(df.isnull().sum() / df.shape[0]* 100)

    missing_info = pd.concat([total, percent], axis = 1, keys=['Total', 'Percent']).sort_values(by='Percent', ascending=False)

    missing_info = missing_info[missing_info['Total'] > 0]

    return missing_info
train_v1 = train.copy()
# train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace=True)

# train.drop(train[(train['GrLivArea']>4500) & (train['SalePrice']<300000)].index, inplace=True)

# train.reset_index(drop=True, inplace=True)
# #Uncomment when needed

# #Normalize SalePrice

train['SalePrice'] = np.log1p(train['SalePrice'])
y = train['SalePrice'].reset_index(drop=True)

## Remove Id and save target variable as y

train = train.drop(['Id', 'SalePrice'], axis=1)

test = test.drop(['Id'], axis=1)



train.shape
test.shape
train_copy = train.copy()

test_copy = test.copy()
all_data_list  = [train_copy, test_copy]
all_data_list[0].shape
for dataset in all_data_list:

    ## No Data Leakage ################################

    for feature in ['MoSold', 'YrSold', 'MSSubClass']:

        dataset[feature] = dataset[feature].apply(str)

    # Assume typical unless deductions are warranted (from the data description)







    ## Some missing values are intentionally left blank, for example: In the Alley feature 

    ## there are blank values meaning that there are no alley's in that specific house. 

    none_available = [ "Alley", 

                       "PoolQC", 

                       "MiscFeature",

                       "Fence",

                       "FireplaceQu",

                       "GarageType",

                       "GarageFinish",

                       "GarageQual",

                       "GarageCond",

                       'BsmtQual',

                       'BsmtCond',

                       'BsmtExposure',

                       'BsmtFinType1',

                       'BsmtFinType2',

                       'MasVnrType']



    for feature in none_available:

        dataset[feature] = dataset[feature].fillna('None')

    none_available2 =  ['BsmtFinSF1',

                        'BsmtFinSF2',

                        'BsmtUnfSF',

                        'TotalBsmtSF',

                        'BsmtFullBath', 

                        'BsmtHalfBath', 

                        'GarageYrBlt',

                        'GarageArea',

                        'GarageCars',

                        'MasVnrArea']



    for feature in none_available2:

        dataset[feature] = dataset[feature].fillna(0)



    ### Possible Data Leakage ####################################



    dataset['Functional'] = dataset['Functional'].fillna('Typ')



    # Fillna with modes as these columns has very less missing data

    mode_feats = list(missing_data(dataset[cat_cols])[missing_data(dataset[cat_cols])['Total'] <2].index)

    for feature in mode_feats:

        dataset[feature] = dataset[feature].fillna(dataset[feature].mode()[0])

    dataset['MSZoning'] = dataset.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))



    ## Replaced all missing values in LotFrontage by imputing the median value of each neighborhood. 

    dataset['LotFrontage'] = dataset.groupby('Neighborhood')['LotFrontage'].transform( lambda x: x.fillna(x.mean()))

    dataset['Utilities'] = dataset['Utilities'].fillna(dataset['Utilities'].mode()[0])
missing_data(train_copy)
missing_data(test_copy)
train_copy.shape
test_copy.shape
for dataset in all_data_list:

    ## Uncomment when needed

    skew_cols = dataset.dtypes[dataset.dtypes != 'object'].index

    skewness = dataset[skew_cols].apply(lambda x: skew(x)).sort_values(ascending =False)

    skewness = skewness[skewness > 0.5]

    high_skew = pd.DataFrame({'Skew' : skewness })

    high_skew_cols = high_skew.index



    # Normalize skewed features

    for i in high_skew_cols:

        dataset[i] = boxcox1p(dataset[i], boxcox_normmax(dataset[i] + 1))
#Creating More Features

for dataset in all_data_list:

    dataset['BsmtFinType1_Unf'] = 1*(dataset['BsmtFinType1'] == 'Unf')

    dataset['HasWoodDeck'] = (dataset['WoodDeckSF'] == 0) * 1

    dataset['HasOpenPorch'] = (dataset['OpenPorchSF'] == 0) * 1

    dataset['HasEnclosedPorch'] = (dataset['EnclosedPorch'] == 0) * 1

    dataset['Has3SsnPorch'] = (dataset['3SsnPorch'] == 0) * 1

    dataset['HasScreenPorch'] = (dataset['ScreenPorch'] == 0) * 1

    dataset['YearsSinceRemodel'] = dataset['YrSold'].astype(int) - dataset['YearRemodAdd'].astype(int)

    dataset['Total_Home_Quality'] = dataset['OverallQual'] + dataset['OverallCond']

    dataset = dataset.drop(['Utilities', 'Street', 'PoolQC',], axis=1)

    dataset['TotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']

    dataset['YrBltAndRemod'] = dataset['YearBuilt'] + dataset['YearRemodAdd']



    dataset['Total_sqr_footage'] = (dataset['BsmtFinSF1'] + dataset['BsmtFinSF2'] +

                                     dataset['1stFlrSF'] + dataset['2ndFlrSF'])

    dataset['Total_Bathrooms'] = (dataset['FullBath'] + (0.5 * dataset['HalfBath']) +

                                   dataset['BsmtFullBath'] + (0.5 * dataset['BsmtHalfBath']))

    dataset['Total_porch_sf'] = (dataset['OpenPorchSF'] + dataset['3SsnPorch'] +

                                  dataset['EnclosedPorch'] + dataset['ScreenPorch'] +

                                  dataset['WoodDeckSF'])

    dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)

    dataset['2ndFlrSF'] = dataset['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)

    dataset['GarageArea'] = dataset['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)

    dataset['GarageCars'] = dataset['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)

    dataset['LotFrontage'] = dataset['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)

    dataset['MasVnrArea'] = dataset['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)

    dataset['BsmtFinSF1'] = dataset['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)



    dataset['haspool'] = dataset['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

    dataset['has2ndfloor'] = dataset['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

    dataset['hasgarage'] = dataset['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

    dataset['hasbsmt'] = dataset['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

    dataset['hasfireplace'] = dataset['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
train_copy.shape
test_copy.shape
# def logs(res, ls):

#     m = res.shape[1]

#     for l in ls:

#         res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)   

#         res.columns.values[m] = l + '_log'

#         m += 1

#     return res



# log_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

#                  'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',

#                  'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',

#                  'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',

#                  'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd']



# all_data = logs(all_data, log_features)
train_copy = pd.get_dummies(train_copy).reset_index(drop=True)

test_copy = pd.get_dummies(test_copy).reset_index(drop=True)
train_copy.shape
test_copy.shape


# train_clean = all_data.iloc[:len(y), :]

# test_clean = all_data.iloc[len(y):, :]

# train_clean.shape, y.shape, test_clean.shape



## Use train_copy and test_copy instead
num_feats = 280
cor_cols = [x for x in num_cols+dis_cols if x not in ('Id', 'SalePrice')]
num_feats= 300

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

rfe_selector = RFE(estimator=LinearRegression(), n_features_to_select=num_feats, step=10, verbose=5)

rfe_selector.fit(train_copy, y)

rfe_support = rfe_selector.get_support()

rfe_feature = train_copy.loc[:,rfe_support].columns.tolist()

print(str(len(rfe_feature)), 'selected features')
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import Lasso, Ridge



embeded_lr_selector = SelectFromModel(Ridge(), max_features=num_feats)

embeded_lr_selector.fit(train_copy, y)



embeded_lr_support = embeded_lr_selector.get_support()

embeded_lr_feature = train_copy.loc[:,embeded_lr_support].columns.tolist()

print(str(len(embeded_lr_feature)), 'selected features')
embeded_lr_feature
X_train, X_test, y_train, y_test = train_test_split(train_copy[rfe_feature], y, train_size=0.75, shuffle=True, random_state=1)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Fit and Predict on X_test

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print (f' Train Score is {lr.score(X_train, y_train)}')

print (f' Test Score is {lr.score(X_test, y_test)}')

mse = mean_squared_error(y_test, y_pred)

print (f' Mean squared error is {(mse)}')

print (format(mse, '.2f'))
# Train Score is 0.9061921419871923

#  Test Score is 0.8117714739572311

#  Mean squared error is 1263621018.643197
coefficients = pd.concat([pd.DataFrame(X_train.columns),pd.DataFrame(np.transpose(lr.coef_))],keys=['feature', 'importance'],ignore_index=True, axis = 1)
coefficients.columns = ['feature', 'importance']
coefficients.sort_values(by='importance', ascending=False)
from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt

model = DecisionTreeRegressor()

model.fit(X_train,y_train)

# print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

y_pred2 = model.predict(X_test)

print (f' Train Score is {model.score(X_train, y_train)}')

print (f' Test Score is {model.score(X_test, y_test)}')

mse = mean_squared_error(y_test, y_pred2)

print (f' Mean squared error is {mse}')

# feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)

# feat_importances.nlargest(15).plot(kind='barh')

# plt.show()
from sklearn.ensemble import ExtraTreesRegressor

import matplotlib.pyplot as plt

model = ExtraTreesRegressor()

model.fit(X_train,y_train)

# print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)

feat_importances.nlargest(15).plot(kind='barh')

plt.show()
import time

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

import seaborn as sns

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

from sklearn import tree

from sklearn.ensemble import RandomForestRegressor

from sklearn import svm
start = time.process_time()

trainedforest = RandomForestRegressor(n_estimators=700).fit(X_train,y_train)

print(time.process_time() - start)

predictionforest = trainedforest.predict(X_test)

print (f' Train Score is {trainedforest.score(X_train, y_train)}')

print (f' Test Score is {trainedforest.score(X_test, y_test)}')

mse = mean_squared_error(y_test, predictionforest)

print (f' Mean squared error is {mse}')
alpha_ridge = [-3,-2,-1,1e-15, 1e-10, 1e-8,1e-5,1e-4, 1e-3,1e-2,0.5,1,1.5, 2,3,4, 5, 10, 20, 30, 40]
from sklearn.linear_model import Lasso 

temp_rss = {}

temp_mse = {}

for i in alpha_ridge:

    ## Assigin each model. 

    lasso_reg = Lasso(alpha= i, normalize=True)

    ## fit the model. 

    lasso_reg.fit(X_train, y_train)

    ## Predicting the target value based on "Test_x"

    y_pred = lasso_reg.predict(X_test)



    mse = mean_squared_error(y_test, y_pred)

    rss = sum((y_pred-y_test)**2)

    temp_mse[i] = mse

    temp_rss[i] = rss
for key, value in sorted(temp_mse.items(), key=lambda item: item[1]):

    print("%s: %s" % (key, value))
lasso_reg = Lasso(alpha=0.0001 , normalize=True)

## fit the model. 

lasso_reg.fit(X_train, y_train)

## Predicting the target value based on "Test_x"

y_pred = lasso_reg.predict(X_test)
print (f' Train Score is {lasso_reg.score(X_train, y_train)}')

print (f' Test Score is {lasso_reg.score(X_test, y_test)}')

mse = mean_squared_error(y_test, y_pred)

print (f' Mean squared error is {mse}')
from sklearn.linear_model import Ridge 

temp_rss = {}

temp_mse = {}

for i in alpha_ridge:

    ## Assigin each model. 

    ridge_reg = Ridge(alpha= i, normalize=True)

    ## fit the model. 

    ridge_reg.fit(X_train, y_train)

    ## Predicting the target value based on "Test_x"

    y_pred = ridge_reg.predict(X_test)



    mse = mean_squared_error(y_test, y_pred)

    rss = sum((y_pred-y_test)**2)

    temp_mse[i] = mse

    temp_rss[i] = rss
for key, value in sorted(temp_mse.items(), key=lambda item: item[1]):

    print("%s: %s" % (key, value))
ridge_reg = Ridge(alpha=0.5 , normalize=True)

## fit the model. 

ridge_reg.fit(X_train, y_train)

## Predicting the target value based on "Test_x"

y_pred = ridge_reg.predict(X_test)
print (f' Train Score is {ridge_reg.score(X_train, y_train)}')

print (f' Test Score is {ridge_reg.score(X_test, y_test)}')

mse = mean_squared_error(y_test, y_pred)

print (f' Mean squared error is {mse}')
coefficients = pd.concat([pd.DataFrame(X_train.columns),pd.DataFrame(np.transpose(ridge_reg.coef_))],ignore_index=True, axis = 1)
coefficients.columns = ['feature', 'importance']
coefficients.sort_values(by='importance', ascending = False)