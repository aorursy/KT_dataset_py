# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import numpy as np





# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)



home_data.head()

home_data.shape
home_data.info()
home_data.columns[home_data.isnull().any()]
# count missing values per column

Isnull = home_data.isnull().sum()/len(home_data)*100

Isnull = Isnull[Isnull>0]

Isnull.sort_values(inplace=True, ascending=False)

Isnull
#Convert into dataframe

Isnull = Isnull.to_frame()

Isnull.columns = ['count']

Isnull.index.names = ['Name']

Isnull['Name'] = Isnull.index

#plot Missing values

plt.figure(figsize=(13, 5))

sns.set(style='whitegrid')

sns.barplot(x='Name', y='count', data=Isnull)

plt.xticks(rotation = 90)

plt.show()
home_data_corr = home_data.select_dtypes(include=[np.number])
del home_data_corr['Id']
#Coralation plot

corr = home_data_corr.corr()

plt.subplots(figsize=(20,9))

sns.heatmap(corr, annot=True)
# PoolQC has missing value ratio is 99%+. So, there is fill by None

home_data['PoolQC'] = home_data['PoolQC'].fillna('None')

#Arround 50% missing values attributes have been fill by None

home_data['MiscFeature'] = home_data['MiscFeature'].fillna('None')

home_data['Alley'] = home_data['Alley'].fillna('None')

home_data['Fence'] = home_data['Fence'].fillna('None')

home_data['FireplaceQu'] = home_data['FireplaceQu'].fillna('None')
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

home_data['LotFrontage'] = home_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

#GarageType, GarageFinish, GarageQual and GarageCond these are replacing with None

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    home_data[col] = home_data[col].fillna('None')



#GarageYrBlt, GarageArea and GarageCars these are replacing with zero

for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:

    home_data[col] = home_data[col].fillna(int(0))

#BsmtFinType2, BsmtExposure, BsmtFinType1, BsmtCond, BsmtQual these are replacing with None

for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'):

    home_data[col] = home_data[col].fillna('None')

#MasVnrArea : replace with zero

home_data['MasVnrArea'] = home_data['MasVnrArea'].fillna(int(0))



#There is put mode value 

home_data['Electrical'] = home_data['Electrical'].fillna(home_data['Electrical']).mode()[0]

#There is no need of Utilities

home_data = home_data.drop(['Utilities'], axis=1)






# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ['MSSubClass', 'LotArea' ,

            'OverallQual', 'OverallCond' , 'YearBuilt' ,

            'YearRemodAdd',  'TotalBsmtSF',

            '1stFlrSF' , '2ndFlrSF',  'LowQualFinSF',

            'GrLivArea', 'BsmtFullBath',  'BsmtHalfBath', 

            'FullBath',  'HalfBath',  'BedroomAbvGr',

            'KitchenAbvGr',	'TotRmsAbvGrd', 'Fireplaces',

            'GarageCars','GarageArea',

            'WoodDeckSF' ,'OpenPorchSF', 'EnclosedPorch',

            '3SsnPorch' , 'ScreenPorch' , 'PoolArea' ,

            'MiscVal', 'MoSold', 'YrSold']



X = home_data[features]

# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model

iowa_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

from sklearn.impute import SimpleImputer

# To improve accuracy, create a new Random Forest model which you will train on all training data

best_size = {}

tree_sizes = [25,50,100,250,500]

for i in tree_sizes:

    best_score = RandomForestRegressor(max_leaf_nodes=i, random_state=1)



    #fit rf_model_on_full_data on all data from the training data

    best_score.fit(X,y)

    val_pred = best_score.predict(val_X)

    best_size[i] = mean_absolute_error(val_pred, val_y)

    



best_tree = min(best_size, key=best_size.get)    

rf_model_on_full_data = RandomForestRegressor(max_leaf_nodes=best_tree, random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X,y)
# path to file you will use for predictions

test_data_path = '../input/test.csv'

# read test data file using pandas

test_data = pd.read_csv(test_data_path)

# PoolQC has missing value ratio is 99%+. So, there is fill by None

test_data['PoolQC'] = test_data['PoolQC'].fillna('None')

#Arround 50% missing values attributes have been fill by None

test_data['MiscFeature'] = test_data['MiscFeature'].fillna('None')

test_data['Alley'] = test_data['Alley'].fillna('None')

test_data['Fence'] = test_data['Fence'].fillna('None')

test_data['FireplaceQu'] = test_data['FireplaceQu'].fillna('None')



#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

test_data['LotFrontage'] = test_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))





#GarageType, GarageFinish, GarageQual and GarageCond these are replacing with None

for col in ['GarageType','MSZoning', 'Exterior1st', 'Exterior2nd', 'MasVnrType' ,'GarageFinish', 'GarageQual', 'GarageCond', 'KitchenQual', 'Functional', 'SaleType']:

		 test_data[col] = test_data[col].fillna('None')



#GarageYrBlt, GarageArea and GarageCars these are replacing with zero

for col in ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:

		  test_data[col] = test_data[col].fillna(int(0))



#BsmtFinType2, BsmtExposure, BsmtFinType1, BsmtCond, BsmtQual these are replacing with None

for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'):

				test_data[col] = test_data[col].fillna('None')



#MasVnrArea : replace with zero

test_data['MasVnrArea'] = test_data['MasVnrArea'].fillna(int(0))



#There is put mode value 

test_data['Electrical'] = test_data['Electrical'].fillna(test_data['Electrical']).mode()[0]



#There is no need of Utilities

test_data = test_data.drop(['Utilities'], axis=1)



print(test_data.columns[test_data.isnull().any()])

test_X = test_data[features]

# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)