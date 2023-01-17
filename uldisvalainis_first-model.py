# Set up code checking
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/house-prices-advanced-regression-techniques/train.csv", "../input/train.csv")  
    os.symlink("../input/house-prices-advanced-regression-techniques/test.csv", "../input/test.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex7 import *
# Random Forest Regression for Kaggle thing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
print(os.listdir("../input"))
# Importing the dataset
dataset = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
testset = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
#CategoricalColumns
categorical_columns = ['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig',
                        'LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
                        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType', 'ExterQual',
                        'Foundation', 'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1', 'BsmtFinType2',
                        'Heating','HeatingQC','CentralAir','Electrical', 'KitchenQual','Functional',
                        'FireplaceQu', 'GarageType','ExterCond','GarageCond', 'GarageFinish','GarageQual','PavedDrive','PoolQC',
                        'Fence', 'MiscFeature','SaleType','SaleCondition']

# Replacing categorical columns with Dummy variables
X = pd.get_dummies(data=dataset, columns=categorical_columns, drop_first=True)
Y = pd.get_dummies(data=testset, columns=categorical_columns, drop_first=True)
#Aligning
X, Y = X.align(Y, join='left', axis=1)

# Taking care of missing data
from sklearn.impute import SimpleImputer as Imputer
imputer = Imputer(missing_values = np.nan, strategy='most_frequent')
imputer = imputer.fit(X)
X = imputer.transform(X)
Y = imputer.transform(Y)

#Converting to DataFrame
data = pd.DataFrame(X)
test = pd.DataFrame(Y)

#features and results for training
X = data.iloc[:,np.r_[1:37,38:247]].values
y = data.iloc[:,37].values

#Features for the test set
X1 = test.iloc[:,np.r_[1:37,38:247]].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

# Feature scaling trian and test for training data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))

#Feature scaling the test set
X1 = sc_X.fit_transform(X1)
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state=0)
regressor.fit(X_train, y_train)


#Second regressor for all the data in Train set
regressor2 = RandomForestRegressor(n_estimators = 100)
regressor2.fit(X,y)
#Predicting and converting to moneys
y_pred= sc_y.inverse_transform(regressor.predict(X_test))

#Mean error for evaluation
print(mean_absolute_error(y_test, y_pred))
#Plotting
#Results
plt.plot(y_pred, color = 'red')
plt.plot(y_test, color = 'blue')
plt.show()
#Importance
important = regressor.feature_importances_
plt.bar(range(len(regressor.feature_importances_)), regressor.feature_importances_)
plt.show()
#Creating predictions for the submission
y_pred1= sc_y.inverse_transform(regressor2.predict(X1))