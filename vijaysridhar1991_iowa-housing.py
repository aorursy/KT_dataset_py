import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor


def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

def extract(dat):
    dat['bath_per_bedroom'] = (dat['FullBath']/dat['BedroomAbvGr']).replace(np.inf, 0)
    #dat['bath_per_bedroom'].replace(np.inf, 0)
    dat['total_bath_Abvgr'] = dat['FullBath'] + dat['HalfBath']
    dat['total_bath_Base'] = dat['BsmtFullBath'] + dat['BsmtHalfBath']
    dat['total_bath'] = dat['total_bath_Base'] + dat['total_bath_Abvgr']
    dat['bath_per_room'] = (dat['total_bath']/dat['TotRmsAbvGrd']).replace(np.inf, 0)
    
    return 

train_file_path = '../input/train.csv' # this is the path to the Iowa data that you will use
test_file_path = '../input/test.csv' 
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

#print(train_data.head())
#print(data.TotalBsmtSF.head())
train_y = train_data.SalePrice
#predictors = ['LotArea', 'LotFrontage', 'GarageArea', 'YearBuilt' , '1stFlrSF' , '2ndFlrSF' , 'FullBath' , 'BedroomAbvGr', 'TotRmsAbvGrd', 'Street', 'Utilities']
#predictors = ['LotArea', 'YearBuilt' , '1stFlrSF' , '2ndFlrSF' , 'FullBath' , 'BedroomAbvGr', 'TotRmsAbvGrd']
#nums = ['LotArea', 'LotFrontage', 'GarageArea', 'YearBuilt' , '1stFlrSF' , '2ndFlrSF' , 'FullBath' , 'BsmtFullBath','BsmtHalfBath','HalfBath','BedroomAbvGr', 'TotRmsAbvGrd','OverallQual','OverallCond','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF','GrLivArea']
#pd.set_option('display.max_colwidth', -1)  
#print(train_data[nums].head())
#predictors = train_data.columns
#train_X = train_data[predictors]
#val_X = test_data[predictors]
train_X = train_data.copy()
del train_X['SalePrice']
val_X = test_data

extract(train_X)
extract(val_X)

print(train_data.columns)


one_hot_encoded_training_predictors = pd.get_dummies(train_X)
one_hot_encoded_test_predictors = pd.get_dummies(val_X)
train_X, val_X = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='inner', 
                                                                    axis=1)

my_imputer = SimpleImputer()
train_X = my_imputer.fit_transform(train_X)
#val_y = my_imputer.fit_transform(val_y)
#train_X.fillna(0)

val_X = my_imputer.fit_transform(val_X)

#val_y = test_data.SalePrice

#for max_leaf_nodes in [35, 40, 45,50]:
#    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
#    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
    
#model = RandomForestRegressor()

#train_X, test_X, train_y, test_y = train_test_split(train_X, train_y,random_state = 0)

model = XGBRegressor(n_estimators=1700, learning_rate = 0.005)
#model.fit(train_X, train_y)

#print(model.fit(train_X, train_y, early_stopping_rounds=7,eval_set=[(test_X, test_y)], verbose=False))
model.fit(train_X, train_y)
predicted_prices = model.predict(val_X)

my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('mysubmission.csv', index=False)

#mean_absolute_error(test_y, predicted_prices)
# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
#print(info.describe())
