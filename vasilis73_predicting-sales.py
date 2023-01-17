# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
train.head()
train.dtypes
train.describe()
# count missing values
train.isnull().sum()
# drop variables with a lot of missing values
#vars_to_drop = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']
#train1 = train.drop(vars_to_drop, axis=1)
#train1.head()
# count mmissing values
#train1.isnull().sum()
# define numeric variables
train_numeric = train.select_dtypes(exclude=["bool_","object_"])
train_numeric.columns
#train1_numeric.head()
#train1_numeric.isnull().sum()
#train1_numeric.shape
# impute missing values for numeric variables
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute, MICE
#from sklearn.preprocessing import Imputer
#my_imputer = Imputer()
#numeric_predictors = my_imputer.fit_transform(train1_numeric)
#train2_numeric = pd.DataFrame(numeric_predictors,columns=train1_numeric.columns)
#train2_numeric.head()
#train2_numeric.isnull().sum()
#train1_numeric = MICE(n_imputations=100,impute_type='pmm',init_fill_method='median',n_pmm_neighbors=5).complete(train_numeric) #probablistic moment matching
train1_numeric = MICE(n_imputations=100,impute_type='col',init_fill_method='random',n_pmm_neighbors=5).complete(train_numeric) #fill in with samples from posterior predictive distribution
train2_numeric = pd.DataFrame(train1_numeric,columns=train_numeric.columns)
# drop Id
train3 = train2_numeric.drop(['Id'], axis=1)
train3.head()
# define categorical variables
# define numeric variables
train_categorical = train.select_dtypes(exclude=["number"])
train_categorical.columns
#train1_categorical.head()
#train1_categorical.shape
# create dummy variables for the categorical variables
one_hot_encoded_training_predictors = pd.get_dummies(train_categorical)
one_hot_encoded_training_predictors.isnull().sum()
one_hot_encoded_training_predictors.head()
one_hot_encoded_training_predictors.shape
# merge numeric and categorical predictors
train4= one_hot_encoded_training_predictors.merge(train3,left_index=True,right_index=True)
train4.head()
#train4.isnull().sum()
train4.columns
train4.shape
# Import the Test dataset
test = pd.read_csv("../input/test.csv")
test.head()
# define numeric variables
test_numeric = test.select_dtypes(exclude=["bool_","object_"])
test_numeric.columns
#train1_numeric.head()
#train1_numeric.isnull().sum()
test_numeric.shape
# impute missing values for numeric variables
#from sklearn.preprocessing import Imputer
#my_imputer = Imputer()
#numeric_predictors_test = my_imputer.fit_transform(test1_numeric)
#test2_numeric = pd.DataFrame(numeric_predictors_test,columns=test1_numeric.columns)
#test2_numeric.head()
#test2_numeric.isnull().sum()
#test1_numeric = MICE(n_imputations=100,impute_type='pmm',init_fill_method='median',n_pmm_neighbors=5).complete(test_numeric) #probablistic moment matching
test1_numeric = MICE(n_imputations=100,impute_type='col',init_fill_method='random',n_pmm_neighbors=5).complete(test_numeric) #fill in with samples from posterior predictive distribution
test2_numeric = pd.DataFrame(test1_numeric,columns=test_numeric.columns)
# drop Id
test3 = test2_numeric.drop(['Id'], axis=1)
test3.head()
test3.shape
# define categorical variables
# define numeric variables
test_categorical = test.select_dtypes(exclude=["number"])
test_categorical.columns
#train1_categorical.head()
test_categorical.shape
# create dummy variables for the categorical variables
one_hot_encoded_test_predictors = pd.get_dummies(test_categorical)
one_hot_encoded_test_predictors.isnull().sum()
one_hot_encoded_test_predictors.head()
one_hot_encoded_test_predictors.shape
# merge numeric and categorical predictors
test4= one_hot_encoded_test_predictors.merge(test3,left_index=True,right_index=True)
test4.head()
#train4.isnull().sum()
test4.columns
test4.shape
# outcome
y = train4.SalePrice
y.shape
# predictors
#X = train4.drop(['SalePrice'],axis=1)
X = train4[test4.columns]
X.shape
# split into training and testing datasets
from sklearn.model_selection import train_test_split
train_X,  test_X, train_y, test_y = train_test_split(X,y,random_state=0)
# feature importance
#from sklearn.ensemble import GradientBoostingRegressor
#regressor = GradientBoostingRegressor()
#regressor.fit(train_X, train_y)
#names = train_X
#print("Features sorted by their score:")
#print(sorted(zip(map(lambda x: round(x, 4), regressor.feature_importances_), names), 
#             reverse=True))
# important features
#important_predictors = ['GrLivArea','OverallQual', 'LotArea', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1', 'OverallCond',
#                        'LotFrontage', 'YearBuilt', '1stFlrSF', 'YearRemodAdd', 'GarageArea', 'GarageYrBlt', 
#                        'WoodDeckSF', 'Neighborhood_Crawfor', 'OpenPorchSF', 'SaleCondition_Abnorml', 
#                        'Fireplaces', 'MoSold', 'Functional_Typ', 'Exterior1st_BrkFace', 'BsmtFullBath', 
#                        'Condition1_Norm', 'MasVnrArea', 'MSZoning_FV', 'MSZoning_C (all)']

#important_predictors = ['GrLivArea','OverallQual',  'TotalBsmtSF',  'BsmtFinSF1', 
#                        'LotFrontage', 'YearBuilt', '1stFlrSF', 'YearRemodAdd', 'GarageArea', 'GarageYrBlt', 
#                        'WoodDeckSF', 'Neighborhood_Crawfor', 'OpenPorchSF', 'SaleCondition_Abnorml', 
#                        'Fireplaces',  'MasVnrArea', 'MSZoning_FV', 'MSZoning_C (all)']

#train_X_new = train_X[important_predictors]
#train_X_new.dtypes
#train_X_new.describe()
#train_X_new.shape
#train_X_new.isnull().sum()
# define the same predictors for the test dataset
#test_X_new = test_X[important_predictors]
# fit a model - GBoost
from sklearn.ensemble import GradientBoostingRegressor
regressor_gb = GradientBoostingRegressor()
regressor_gb.fit(train_X,train_y)
# make predictions - GBoost
preds = regressor_gb.predict(test_X)
from sklearn.metrics import mean_absolute_error
print("mean_absolute_error : " + str(mean_absolute_error(preds, test_y)))
# fit a model - XGBoost
from xgboost import XGBRegressor
regressor_xgb = XGBRegressor()
regressor_xgb.fit(train_X,train_y)
# make predictions - XGBoost
preds = regressor_xgb.predict(test_X)
print("mean_absolute_error : " + str(mean_absolute_error(preds, test_y)))
# fit a Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor()
regressor_rf.fit(train_X,train_y)
# make predictions - Random Forets
from sklearn.metrics import mean_absolute_error
preds = regressor_rf.predict(test_X)
print("mean_absolute_error : " + str(mean_absolute_error(preds, test_y)))
# fit a Neural Network 
# # Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#train_X_new = sc.fit_transform(train_X_new)
#test_X_new = sc.transform(test_X_new)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
train_X_new = sc.fit_transform(train_X)
test_X_new = sc.transform(test_X)
# Import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.layers import Input, Dense, Dropout
# fit the model
# Initialising the ANN
regressor = Sequential()

# Adding the input layer and the first hidden layer
regressor.add(Dense(output_dim = 128,  activation = 'relu', input_dim = 270))

# Adding the output layer
regressor.add(Dense(1))
inp = Input(shape=(270,))
hidden_1 = Dense(128, activation='relu')(inp)
dropout_1 = Dropout(0.2)(hidden_1)
hidden_2 = Dense(128, activation='relu')(dropout_1)
dropout_2 = Dropout(0.2)(hidden_2)
hidden_3 = Dense(128, activation='relu')(dropout_2)
dropout_3 = Dropout(0.2)(hidden_3)
hidden_4 = Dense(128, activation='relu')(dropout_3)
dropout_4 = Dropout(0.2)(hidden_4)
hidden_5 = Dense(128, activation='relu')(dropout_4)
dropout_5 = Dropout(0.2)(hidden_5)
out = Dense(1)(dropout_5)

regressor = Model(inputs=inp, outputs=out)
# Compiling the ANN
regressor.compile(optimizer = 'adam', loss = 'mse', metrics=['mae'])
#regressor.summary()
# Fitting the ANN to the Training set
regressor.fit(train_X_new, train_y, epochs = 100, batch_size = 32, verbose=1, validation_split=0.1)
# make predictions - Neural Network
from sklearn.metrics import mean_absolute_error
preds = regressor.predict(test_X_new)
print("mean_absolute_error : " + str(mean_absolute_error(preds, test_y)))
# Import the Test dataset
#test = pd.read_csv("../input/test.csv")
#test.head()
#test.isnull().sum()
# drop variables with a lot of missing values
#vars_to_drop = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']
#test1 = test.drop(vars_to_drop, axis=1)
#test1.head()
# define numeric variables
#test_numeric = test.select_dtypes(exclude=["bool_","object_"])
#test_numeric.columns
#train1_numeric.head()
#train1_numeric.isnull().sum()
#test_numeric.shape
# impute missing values for numeric variables
#from sklearn.preprocessing import Imputer
#my_imputer = Imputer()
#numeric_predictors_test = my_imputer.fit_transform(test1_numeric)
#test2_numeric = pd.DataFrame(numeric_predictors_test,columns=test1_numeric.columns)
#test2_numeric.head()
#test2_numeric.isnull().sum()
#test1_numeric = MICE(n_imputations=100,impute_type='pmm',init_fill_method='median',n_pmm_neighbors=5).complete(test_numeric) #probablistic moment matching
#test1_numeric = MICE(n_imputations=100,impute_type='col',init_fill_method='random',n_pmm_neighbors=5).complete(test_numeric) #fill in with samples from posterior predictive distribution
#test2_numeric = pd.DataFrame(test1_numeric,columns=test_numeric.columns)
# drop Id
#test3 = test2_numeric.drop(['Id'], axis=1)
#test3.head()
#test3.shape
# define categorical variables
# define numeric variables
#test_categorical = test.select_dtypes(exclude=["number"])
#test_categorical.columns
#train1_categorical.head()
#test_categorical.shape
# create dummy variables for the categorical variables
#one_hot_encoded_test_predictors = pd.get_dummies(test_categorical)
#one_hot_encoded_test_predictors.isnull().sum()
#one_hot_encoded_test_predictors.head()
#one_hot_encoded_test_predictors.shape
# merge numeric and categorical predictors
#test4= one_hot_encoded_test_predictors.merge(test3,left_index=True,right_index=True)
#test4.head()
#train4.isnull().sum()
#test4.columns
#test4.shape
# important features
#important_predictors = ['GrLivArea','OverallQual', 'LotArea', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1', 'OverallCond',
#                        'LotFrontage', 'YearBuilt', '1stFlrSF', 'YearRemodAdd', 'GarageArea', 'GarageYrBlt', 
#                        'WoodDeckSF', 'Neighborhood_Crawfor', 'OpenPorchSF', 'SaleCondition_Abnorml', 
##                        'Fireplaces', 'MoSold', 'Functional_Typ', 'Exterior1st_BrkFace', 'BsmtFullBath', 
#                        'Condition1_Norm', 'MasVnrArea', 'MSZoning_FV', 'MSZoning_C (all)']

#important_predictors1 = ['GrLivArea','OverallQual',  'TotalBsmtSF',  'BsmtFinSF1', 
#                        'LotFrontage', 'YearBuilt', '1stFlrSF', 'YearRemodAdd', 'GarageArea', 'GarageYrBlt', 
#                        'WoodDeckSF', 'Neighborhood_Crawfor', 'OpenPorchSF', 'SaleCondition_Abnorml', 
#                        'Fireplaces',  'MasVnrArea', 'MSZoning_FV', 'MSZoning_C (all)']

#test4 = test4[important_predictors]
#test4.dtypes
#test4.shape
#train_X_new.shape
# make predictions - XGBoost
#preds = regressor_xgb.predict(test4)
#preds

# Prepare submission file - XGBoost
#xgb = pd.DataFrame({'Id':test.Id,'SalePrice':preds})
# prepare the csv file
#xgb.to_csv('xgb6.csv',index=False)
# fit a model - GBoost
#from sklearn.ensemble import GradientBoostingRegressor
#regressor = GradientBoostingRegressor()
#regressor.fit(train_X_new,train_y)
#preds = regressor.predict(test4)
#preds
# fit a model - Random Forest
#preds_rf = regressor_rf.predict(test4)
#preds_rf
# standardise the data
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#test4_new = sc.fit_transform(test4)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
test4_new = sc.fit_transform(test4)
# make predictions for DNN
#regressor.fit(train_X_new,train_y)
preds = regressor.predict(test4_new)
#preds.shape
preds1 = pd.DataFrame(preds,columns=['preds'])
#preds1.head()
#preds1.shape

# Prepare submission file - GBoost
#gb3 = pd.DataFrame({'Id':test.Id,'SalePrice':preds})
# prepare the csv file
#gb3.to_csv('gb3.csv',index=False)
# Prepare submission file - RF
#rf = pd.DataFrame({'Id':test.Id,'SalePrice':preds_rf})
# prepare the csv file
#rf.to_csv('rf3.csv',index=False)
# Prepare submission file - DNN
dnn = pd.DataFrame({'Id':test.Id,'SalePrice':preds1.preds})
# prepare the csv file
dnn.to_csv('dnn2.csv',index=False)