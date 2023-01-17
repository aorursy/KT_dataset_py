## Importing required libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Basic Statistics
from scipy import stats
from scipy.stats import norm

# For scaling the data
from sklearn.preprocessing import StandardScaler

# For splitting the dataset into train and test
from sklearn.model_selection import train_test_split

# Light GBM model
import lightgbm as lgb
from sklearn.model_selection import KFold, GridSearchCV
from sklearn import preprocessing

# XGBoost model
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# For calculating accuracy values
from sklearn.metrics import mean_squared_error, r2_score

# For ignoring the warnings
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
# Fetching the data to pandas dataframes
test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
# Concatenating training and test data into a single dataframe
data = pd.concat([train, test], sort=False)
data = data.reset_index(drop=True)
data.head()
print('Train data : ', train.shape)
print('Test data : ' , test.shape)
print('Complete data : ', data.shape)

train['SalePrice'].describe()
## Checking the distribution of target variable
plt.figure(figsize=(12,6))
sns.distplot(train['SalePrice']);

## QQ-plot ( normal probability plot )
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
#applying log transformation
train['SalePrice'] = np.log(train['SalePrice'])
#transformed histogram and normal probability plot
sns.distplot(train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
# Checking the missing values
total_missing = data.isnull().sum().sort_values(ascending=False)
percent_missing = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_missing, percent_missing], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
#correlation matrix
corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corrmat, vmax=.8,cmap="Blues", square=True);
plt.figure(figsize=(7,20))
sns.heatmap(train.corr()[['SalePrice']].sort_values(by=['SalePrice'],ascending=False).head(50), vmin=-1, annot=True);
# Dropping columns
data.drop(["GarageArea"], axis = 1, inplace = True)
# Categorical features
data["GarageType"]   = data["GarageType"].fillna("None")
data["GarageFinish"] = data["GarageFinish"].fillna("None")
data["GarageQual"] = data["GarageQual"].fillna("None")
data["GarageCond"] = data["GarageCond"].fillna("None")

# Numerical features
data["GarageYrBlt"]  = data["GarageYrBlt"].fillna(0)
data["GarageCars"] = data["GarageCars"].fillna(0)
# Categorical features
data["BsmtQual"] = data["BsmtQual"].fillna("None")
data["BsmtCond"] = data["BsmtCond"].fillna("None")
data["BsmtExposure"] = data["BsmtExposure"].fillna("None")
data["BsmtFinType1"] = data["BsmtFinType1"].fillna("None")
data["BsmtFinType2"] = data["BsmtFinType2"].fillna("None")


# Numerical features
data["BsmtFinSF1"]  = data["BsmtFinSF1"].fillna(0)
data["BsmtFinSF2"]  = data["BsmtFinSF2"].fillna(0)
data["BsmtUnfSF"]   = data["BsmtUnfSF"].fillna(0)
data["TotalBsmtSF"] = data["TotalBsmtSF"].fillna(0)
data["BsmtFullBath"] = data["BsmtFullBath"].fillna(0)
data["BsmtHalfBath"] = data["BsmtHalfBath"].fillna(0)
data["MasVnrType"] = data["MasVnrType"].fillna("None")
data["MasVnrArea"] = data["MasVnrArea"].fillna(0)
# Filling None as a value for categroical values
data["Alley"] = data["Alley"].fillna("None")
data["PoolQC"] = data["PoolQC"].fillna("None")
data["Fence"]  = data["Fence"].fillna("None")
data["MiscFeature"] = data["MiscFeature"].fillna("None")
data["FireplaceQu"]  = data["FireplaceQu"].fillna("None")

# After checking the data in the respective columns i felt that filling the missing values with Mode will provide much 
# better insights than filling the values with 0 or median since most of the values are biased towards a single value

data['SaleType']    = data['SaleType'].fillna(data['SaleType'].mode()[0])
data["Electrical"]  = data.groupby("YearBuilt")['Electrical'].transform(lambda x: x.fillna(x.mode()[0]))
data['KitchenQual']  = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])
data['MSZoning']  = data['MSZoning'].fillna(data['MSZoning'].mode()[0])
data['Utilities'] = data['Utilities'].fillna(data['Utilities'].mode()[0])
data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])
data['LotFrontage'].interpolate(method='linear',inplace=True)
data["Functional"]   = data["Functional"].fillna("Typ")

data["SalePrice"] = data["SalePrice"].fillna(0)
# Checking if any null value is remaining in the dataset
data.isnull().sum().sort_values(ascending = False).head(15)
# Finding out the columns with integer datatype
int_type_variables = [column for column in train.columns if train[column].dtype in ['int64']]
int_type_variables
# Creating a function to plot the numerical variables with the target variable
def plotting_numerical(data):
    for col in int_type_variables:
        if col != 'SalePrice':
            print(col)
            print(data[col].dtype)
    
            plt.figure(figsize=(10, 10))
            ax = sns.scatterplot(x=col, y='SalePrice', data=data)
            plt.show();
sns.set_style('whitegrid')
plotting_numerical(train)
# Changing the datatype of integer columns to object 
list_of_columns_for_datatype_change=['MSSubClass','OverallQual','OverallCond','BsmtFullBath','GarageCars','BsmtHalfBath','FullBath',
                                     'HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','MoSold','YrSold']
data[list_of_columns_for_datatype_change]=data[list_of_columns_for_datatype_change].apply(lambda column:column.astype('object'))
# Preprocessing the data by label encoding all the categorical variables in data
# Creating a list of object type variables
obj_type_variables = [column for column in data.columns if data[column].dtype in ['object']]

# Label Encoding all the categorical variables
le = preprocessing.LabelEncoder()
for li in obj_type_variables:
    le.fit(list(set(data[li])))
    data[li] = le.transform(data[li])

# Splitting the test and train data as orignal
train, test = data[:len(train)], data[len(train):]
train.drop(train[(train['BsmtFinSF1']>5000)].index, inplace=True)
train.drop(train[(train['TotalBsmtSF']>6000)].index, inplace=True)
train.drop(train[(train['1stFlrSF']>4000)].index, inplace=True)
train.drop(train[(train['MiscVal']>8000)].index, inplace=True)
train.drop(train[(train['OpenPorchSF']>500) & (train['SalePrice']<250000)].index, inplace=True)
train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace=True)
train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index, inplace=True)
train.reset_index(drop=True, inplace=True)

# Creating the variables for model fitting
X = train.drop(columns=['SalePrice', 'Id']) 
y = train['SalePrice']

# Test variable
test = test.drop(columns=['SalePrice', 'Id'])
train.dtypes.value_counts()
## Training the XGboost regression model
Xgb_model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

## Fitting the model
Xgb_model.fit(X, y)

## Calculating r2 score for the model
r2_score(Xgb_model.predict(X), y)
predictions = Xgb_model.predict(test)
predictions.size
sample_submission=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission=pd.DataFrame({"Id":sample_submission['Id'],
                         "SalePrice":predictions})
submission.to_csv('submission_xgb.csv',index=False)
# For calculating accuracy values
from sklearn.metrics import mean_squared_error, r2_score
kfold = KFold(n_splits=5, random_state = 2020, shuffle = True)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(X, y)
r2_score(model_lgb.predict(X), y)
pred = model_lgb.predict(test)
sample_submission=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission=pd.DataFrame({"Id":sample_submission['Id'],
                         "SalePrice":predictions})
submission.to_csv('submission_lgb.csv',index=False)
# Gradient Boosting Regressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

model_gbr = GradientBoostingRegressor(n_estimators=6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42)  

model_gbr.fit(X, y)
r2_score(model_gbr.predict(X), y)
prediction_gbr = model_gbr.predict(test)
sample_submission=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission=pd.DataFrame({"Id":sample_submission['Id'],
                         "SalePrice":prediction_gbr})
submission.to_csv('submission_gbr.csv',index=False)
# Random Forest Regressor
model_rf = RandomForestRegressor(n_estimators=1200,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=42)

model_rf.fit(X, y)
r2_score(model_rf.predict(X), y)
prediction_rf = model_rf.predict(test)
sample_submission=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission=pd.DataFrame({"Id":sample_submission['Id'],
                         "SalePrice":prediction_rf})
submission.to_csv('submission_rf.csv',index=False)
# Stack up all the models above, optimized using xgboost
stacked_models = StackingCVRegressor(regressors=(Xgb_model, model_lgb, model_gbr, model_rf),
                                meta_regressor=Xgb_model,
                                use_features_in_secondary=True)

stacked_models.fit(X, y)
# Blend models in order to make the final predictions more robust to overfitting
def blended_predictions(X):
    return (((0.1 * model_gbr.predict(X)) + \
            (0.2 * Xgb_model.predict(X)) + \
            (0.25 * model_lgb.predict(X)) + \
            (0.1 * model_rf.predict(X)) + \
            (0.35 * stacked_models.predict(np.array(X)))))


final_predictions  = blended_predictions(test)
sample_submission=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission=pd.DataFrame({"Id":sample_submission['Id'],
                         "SalePrice":final_predictions})
submission.to_csv('submission_stack.csv',index=False)