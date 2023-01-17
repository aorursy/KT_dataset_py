# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
pd.options.mode.chained_assignment = None
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
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

#missing data handling


explained_variance = pd.DataFrame(pca.explained_variance_ratio_)
explained_variance.round(4)
new_features=['MSZoning']
one_code_features=pd.get_dummies(home_data[new_features])
X=pd.concat([X,one_code_features], axis=1)
X.head(10)

train_X.head()
sc = StandardScaler()  
train_X_st = pd.DataFrame(sc.fit_transform(train_X))
train_X_st.columns=train_X.columns
train_X_st.head()
pca = PCA(0.95)
train_X_st = pca.fit_transform(train_X)
train_X_st
explained_variance = pd.DataFrame(pca.explained_variance_ratio_)
explained_variance.round(4)
my_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), XGBRegressor(n_estimators=1000, learning_rate=0.05))
#my_imputer = SimpleImputer(strategy='most_frequent')
#new_data = pd.DataFrame(my_imputer.fit_transform(home_data))
new_data =  home_data
#new_data.columns = home_data.columns
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 
            'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd','GrLivArea',
            'OverallQual', 'OverallCond','GarageCars','MSSubClass']
X = new_data[features]


#new_features=['MSZoning', 'LotShape','Utilities', 
              #'Street','Alley', 'LotConfig','Neighborhood','HouseStyle','BldgType'
            # ]

new_features=['MSZoning','LotShape','LandContour','LotConfig',
               'Neighborhood','Condition1','Condition2','BldgType',
               'RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
               'BsmtFinType2','Heating','HouseStyle','Foundation','MasVnrType','BsmtFinType1',
               'Electrical','Functional','GarageType','Alley','Utilities',
               'GarageCond','Fence','MiscFeature','SaleType','SaleCondition','LandSlope','CentralAir',
               'GarageFinish','BsmtExposure','Street','KitchenQual', 'GarageQual','ExterQual']

one_code_features=pd.get_dummies(home_data[new_features])
X=pd.concat([X,one_code_features], axis=1)
X[['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 
   'GrLivArea', 'OverallQual', 'OverallCond', 'GarageCars']]=X[['LotArea', 'YearBuilt', '1stFlrSF', 
                                                                              '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 
                                                                              'GrLivArea', 'OverallQual', 'OverallCond', 'GarageCars']].apply(pd.to_numeric)


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
sc = StandardScaler()
sc.fit(train_X)
train_X = pd.DataFrame(sc.transform(train_X))
#train_X.columns=X.columns
val_X = pd.DataFrame(sc.transform(val_X))
#val_X.columns=X.columns

#PCA applying for train data only
pca = PCA(0.95)
pca.fit(train_X)
train_X=pd.DataFrame(pca.transform(train_X))
val_X=pd.DataFrame(pca.transform(val_X))


#Split into validation and training data


#rf_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
#rf_model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(val_X, val_y)], verbose=False)

#rf_model = RandomForestRegressor(random_state=1,max_leaf_nodes=240)
#rf_model.fit(train_X,train_y)
scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error', cv=5)
print(scores)
#rf_val_predictions = rf_model.predict(val_X)
#rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
#print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
val_X.head()
test_X.columns

pd.options.mode.chained_assignment = None
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
# path to file you will use for predictions
test_data_path = '../input/test.csv'

#copy dataframe from original
new_data = home_data.copy()
#convert to np array
new_data=new_data.values
#fill in missing values # Imputation
my_imputer = SimpleImputer(strategy='most_frequent')
new_data = pd.DataFrame(my_imputer.fit_transform(new_data))
new_data.columns = home_data.columns

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 
            'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd','GrLivArea',
            'OverallQual', 'OverallCond','GarageCars','MSSubClass']

new_features=['MSZoning','LotShape','LandContour','LotConfig',
               'Neighborhood','Condition1','Condition2','BldgType',
               'RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
               'BsmtFinType2','Heating','HouseStyle','Foundation','MasVnrType','BsmtFinType1',
               'Electrical','Functional','GarageType','Alley','Utilities',
               'GarageCond','Fence','MiscFeature','SaleType','SaleCondition','LandSlope','CentralAir',
               'GarageFinish','BsmtExposure','Street','KitchenQual', 'GarageQual','ExterQual']

one_code_features_X=pd.get_dummies(new_data[new_features])
X = new_data[features]
X=pd.concat([X,one_code_features_X], axis=1)
X[['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 
   'GrLivArea', 'OverallQual', 'OverallCond', 'GarageCars', 'MSSubClass']]=X[['LotArea', 'YearBuilt', '1stFlrSF', 
                                                                              '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 
                                                                              'GrLivArea', 'OverallQual', 'OverallCond', 'GarageCars',
                                                                              'MSSubClass']].apply(pd.to_numeric)

# read TEST data file using pandas
test_data = pd.read_csv(test_data_path)
#imputing test data
test_data_2=test_data.values
test_data_2 = pd.DataFrame(my_imputer.fit_transform(test_data))
test_data_2.reset_index(drop=True, inplace=True)
test_data_2.columns = test_data.columns
#get dummies for test_data
one_code_features_test_X=pd.get_dummies(test_data_2[new_features])
#concatinating test data
test_X=test_data_2[features]
test_X=pd.concat([test_X,one_code_features_test_X], axis=1)

test_X[['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 
   'GrLivArea', 'OverallQual', 'OverallCond', 'GarageCars', 'MSSubClass']]=test_X[['LotArea', 'YearBuilt', '1stFlrSF', 
                                                                              '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 
                                                                              'GrLivArea', 'OverallQual', 'OverallCond', 'GarageCars',
                                                                              'MSSubClass']].apply(pd.to_numeric)
#drop columns which exist in train data, but not in test
train_cols = X.columns
test_cols = test_X.columns
train_not_test = train_cols.difference(test_cols).tolist()
X.drop(train_not_test, axis=1,inplace=True)

y=new_data.SalePrice

sc = StandardScaler()
sc.fit(X)
X = pd.DataFrame(sc.transform(X))
#train_X.columns=X.columns
test_X = pd.DataFrame(sc.transform(test_X))
#val_X.columns=X.columns

#PCA applying for train data only
pca = PCA(0.95)
pca.fit(X)
X=pd.DataFrame(pca.transform(X))
test_X=pd.DataFrame(pca.transform(test_X))

#convert fields to numbers

rf_model_on_full_data = XGBRegressor(n_estimators=1000, learning_rate=0.05)
rf_model_on_full_data.fit(X, y, early_stopping_rounds=5, eval_set=[(X, y)], verbose=False)

# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X)

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
pd.options.mode.chained_assignment = None
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
# path to file you will use for predictions
test_data_path = '../input/test.csv'

#copy dataframe from original
new_data = home_data.copy()
#convert to np array
new_data=new_data.values
#fill in missing values # Imputation
my_imputer = SimpleImputer(strategy='most_frequent')
new_data = pd.DataFrame(my_imputer.fit_transform(new_data))
new_data.columns = home_data.columns

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 
            'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd','GrLivArea',
            'OverallQual', 'OverallCond','GarageCars','MSSubClass']

new_features=['MSZoning','LotShape','LandContour','LotConfig',
               'Neighborhood','Condition1','Condition2','BldgType',
               'RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
               'BsmtFinType2','Heating','HouseStyle','Foundation','MasVnrType','BsmtFinType1',
               'Electrical','Functional','GarageType','Alley','Utilities',
               'GarageCond','Fence','MiscFeature','SaleType','SaleCondition','LandSlope','CentralAir',
               'GarageFinish','BsmtExposure','Street','KitchenQual', 'GarageQual','ExterQual']

one_code_features_X=pd.get_dummies(new_data[new_features])
X = new_data[features]
X=pd.concat([X,one_code_features_X], axis=1)
X.drop(['GarageQual_Ex', 'HouseStyle_2.5Fin', 'Utilities_NoSeWa'], axis=1,inplace=True)

# read TEST data file using pandas
test_data = pd.read_csv(test_data_path)
#imputing test data
test_data_2=test_data.values
test_data_2 = pd.DataFrame(my_imputer.fit_transform(test_data))
test_data_2.reset_index(drop=True, inplace=True)
test_data_2.columns = test_data.columns
#get dummies for test_data
one_code_features_test_X=pd.get_dummies(test_data_2[new_features])
#concatinating test data
test_X=test_data_2[features]
test_X=pd.concat([test_X,one_code_features_test_X], axis=1)

#X.drop(['GarageQual_Ex', 'HouseStyle_2.5Fin', 'Utilities_NoSeWa'], axis=1,inplace=True)
pd.options.mode.chained_assignment = None
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
# path to file you will use for predictions
test_data_path = '../input/test.csv'

#copy dataframe from original
new_data = home_data.copy()
#convert to np array
new_data=new_data.values
#fill in missing values # Imputation
my_imputer = SimpleImputer(strategy='most_frequent')
new_data = pd.DataFrame(my_imputer.fit_transform(new_data))
new_data.columns = home_data.columns

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 
            'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd','GrLivArea',
            'OverallQual', 'OverallCond','GarageCars','MSSubClass']

new_features=['MSZoning','LotShape','LandContour','LotConfig',
               'Neighborhood','Condition1','Condition2','BldgType',
               'RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
               'BsmtFinType2','Heating','HouseStyle','Foundation','MasVnrType','BsmtFinType1',
               'Electrical','Functional','GarageType','Alley','Utilities',
               'GarageCond','Fence','MiscFeature','SaleType','SaleCondition','LandSlope','CentralAir',
               'GarageFinish','BsmtExposure','Street','KitchenQual', 'GarageQual','ExterQual']

one_code_features_X=pd.get_dummies(new_data[new_features])

X = new_data[features]
X=pd.concat([X,one_code_features_X], axis=1)
X.drop(['GarageQual_Ex', 'HouseStyle_2.5Fin', 'Utilities_NoSeWa'], axis=1,inplace=True)
X.head()
cols_with_missing = [col for col in test_X.columns 
                                 if test_X[col].isnull().any()]
cols_with_missing
test_data_2 = pd.DataFrame(my_imputer.fit_transform(test_data))
cols_with_missing = [col for col in test_data_2.columns 
                                 if test_data_2[col].isnull().any()]
cols_with_missing