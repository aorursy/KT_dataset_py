import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
#import Data
training_data = pd.read_csv('../input/train.csv')
testing_data = pd.read_csv('../input/test.csv')


# Remove the Header and the ID Column
training_data = training_data.iloc[1:,1:]
testing_id = testing_data.iloc[:,0]
testing_id = pd.DataFrame(testing_id)
testing_data = testing_data.iloc[:,1:]


testing_data = pd.DataFrame(testing_data)
training_data = pd.DataFrame(training_data)


ncols = len(training_data.columns)
nrows = len(training_data.index)
training = training_data.iloc[:, 0:(ncols - 1)].values.reshape(nrows, ncols - 1)
y_train = training_data.iloc[:, (ncols - 1)].values.reshape(nrows, 1)

print("Y_TRAIN: ",y_train.shape)
print("Training_data:",training.shape)
print("Testing_data",testing_data.shape)


training_data = training
testing_data = pd.DataFrame(testing_data, columns = ["MSSubClass","MSZoning","LotFrontage","LotArea","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","OverallQual","OverallCond","YearBuilt","YearRemodAdd","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","MasVnrArea","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinSF1","BsmtFinType2","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","Heating","HeatingQC","CentralAir","Electrical","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","KitchenQual","TotRmsAbvGrd","Functional","Fireplaces","FireplaceQu","GarageType","GarageYrBlt","GarageFinish","GarageCars","GarageArea","GarageQual","GarageCond","PavedDrive","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","PoolQC","Fence","MiscFeature","MiscVal","MoSold","YrSold","SaleType","SaleCondition"])
training_data = pd.DataFrame(training_data)
training = pd.DataFrame(training, columns = ["MSSubClass","MSZoning","LotFrontage","LotArea","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","OverallQual","OverallCond","YearBuilt","YearRemodAdd","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","MasVnrArea","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinSF1","BsmtFinType2","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","Heating","HeatingQC","CentralAir","Electrical","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","KitchenQual","TotRmsAbvGrd","Functional","Fireplaces","FireplaceQu","GarageType","GarageYrBlt","GarageFinish","GarageCars","GarageArea","GarageQual","GarageCond","PavedDrive","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","PoolQC","Fence","MiscFeature","MiscVal","MoSold","YrSold","SaleType","SaleCondition"])


#result = pd.concat([training,testing_data],axis = 0)
result = training.append(testing_data)
print("Result: ",result.shape)
#print(result)

headers_for_category = [1,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,26,27,28,29,30,31,32,34,38,39,40,41,52,54,56,57,59,62,63,64,71,72,73,77,78]
le = {}

for i in headers_for_category:
  #print(i)
  le[i] = preprocessing.LabelEncoder()
  a = pd.DataFrame(result.iloc[:,i])
  a[pd.isnull(a)] = 'NaN'
  
  le[i].fit(a)
  result.iloc[:,i] = le[i].transform(a)

si = Imputer(missing_values=np.nan, strategy='most_frequent')
si.fit(result)
result = si.transform(result)
result = pd.DataFrame(result)
      


ipca = IncrementalPCA(n_components=10, batch_size=20)
ipca.fit(result)

result = ipca.transform(result) 
print(result)
print(result.shape)
print(ipca.explained_variance_ratio_)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(result)
result = scaler.transform(result)


print("\n :::::::::::::::::::::::::::::::: \n")
#print(result)

result = pd.DataFrame(result)
ncols = len(result.columns)
nrows = len(result.index)

final_training_dataset = result.iloc[0:(int(nrows/2)), :].values.reshape((int(nrows/2)), ncols)
final_testing_dataset = result.iloc[(int(nrows/2)):nrows, :].values.reshape((int(nrows/2)), ncols)
final_training_target = y_train

print(final_training_dataset.shape)
print(final_testing_dataset.shape)
print(final_training_target.shape)
from xgboost import XGBRegressor
from statistics import mean
folds = 10


explained_variance_score_array = []
mean_squared_error_array = []
mean_squared_log_error_array = []
r2_score_array = []

final_training_dataset = pd.DataFrame(final_training_dataset)
final_training_target = pd.DataFrame(final_training_target)

final_testing_dataset = pd.DataFrame(final_testing_dataset)


final_training_dataset = final_training_dataset.values
final_training_target = final_training_target.values

final_testing_dataset = final_testing_dataset.values



for i in range(0,folds):
    x_train, x_test, y_train, y_test = train_test_split(final_training_dataset, final_training_target, test_size=.25)
    #reg = RandomForestRegressor(max_depth=50, random_state=1, n_estimators=100, max_features='auto')
    #reg.fit(x_train, y_train)

    reg = XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)
    reg.fit(x_train, y_train)
    
    predicted_value = reg.predict(x_test)
  #print(predicted_value[0])
  #print(predicted_value[0] - y_test[0])

    from sklearn.metrics import explained_variance_score
    explained_variance_score_array.append(explained_variance_score(y_test, predicted_value))

    from sklearn.metrics import mean_squared_error
    mean_squared_error_array.append(mean_squared_error(y_test, predicted_value))

    from sklearn.metrics import mean_squared_log_error
    mean_squared_log_error_array.append(mean_squared_log_error(y_test, predicted_value))

    from sklearn.metrics import r2_score
    r2_score_array.append(r2_score(y_test, predicted_value))

    
    
print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")

#reg = XGBRegressor(n_estimators=50, booster='gbtree', learning_rate=0.1, max_delta_step=5)
reg = XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)
reg.fit(final_training_dataset, final_training_target)

predicted_value = reg.predict(final_testing_dataset)
print(final_training_dataset)
print(final_training_target)
print(final_testing_dataset)
print(predicted_value)
predicted_value = pd.DataFrame(predicted_value,columns = ["SalePrice"])

final_predicted_values = pd.concat([testing_id, predicted_value], axis=1)
final_predicted_values = final_predicted_values.iloc[:,:]
print(final_predicted_values)
final_predicted_values.to_csv("test_results_XGBOOST.csv", encoding='utf-8', index=False)

print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
print("explained_variance_score: ",explained_variance_score_array)
print("mean_squared_error: ",mean_squared_error_array)
print("mean_squared_log_error: ",mean_squared_log_error_array)
print("r2_score: ",r2_score_array)

print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
print("explained_variance_score: ",mean(explained_variance_score_array))
print("mean_squared_error: ",mean(mean_squared_error_array))
print("mean_squared_log_error: ",mean(mean_squared_log_error_array))
print("r2_score: ",mean(r2_score_array))



print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
x_train, x_test, y_train, y_test = train_test_split(final_training_dataset, final_training_target, test_size=.25)

reg = XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)

#scoring_parameter = ['explained_variance','test_score','train_score']
scores = cross_validate(reg, final_training_dataset, final_training_target, cv=10, return_train_score=True)

print("\n\n\n\n\n")
print(scores.keys())
print("Test Score: ",scores['test_score'])
print("Test Score Average: ", scores['test_score'].mean())
print()
print("Train Score: ",scores['train_score'])
print("Train Score Average: ",scores['train_score'].mean())

print("\n\n\n\n\n\n")
scoring_parameter = ['explained_variance']
scores = cross_validate(reg, final_training_dataset, final_training_target, cv=10, return_train_score=True, scoring=scoring_parameter)

print("\n\n\n\n\n")
print(scores.keys())
print("Test Variance: ",scores['test_explained_variance'])
print("Test Variance Average: ",scores['test_explained_variance'].mean())
print()
print("Train Variance: ",scores['train_explained_variance'])
print("Train Variance Average: ",scores['train_explained_variance'].mean())

from sklearn.ensemble import RandomForestRegressor
from statistics import mean
folds = 10


explained_variance_score_array = []
mean_squared_error_array = []
mean_squared_log_error_array = []
r2_score_array = []

final_training_dataset = pd.DataFrame(final_training_dataset)
final_training_target = pd.DataFrame(final_training_target)

final_testing_dataset = pd.DataFrame(final_testing_dataset)


final_training_dataset = final_training_dataset.values
final_training_target = final_training_target.values

final_testing_dataset = final_testing_dataset.values



for i in range(0,folds):
    x_train, x_test, y_train, y_test = train_test_split(final_training_dataset, final_training_target, test_size=.25)
    reg = RandomForestRegressor(max_depth=50, random_state=1, n_estimators=100, max_features='auto')
    reg.fit(x_train, y_train)
    
    predicted_value = reg.predict(x_test)
  #print(predicted_value[0])
  #print(predicted_value[0] - y_test[0])

    from sklearn.metrics import explained_variance_score
    explained_variance_score_array.append(explained_variance_score(y_test, predicted_value))

    from sklearn.metrics import mean_squared_error
    mean_squared_error_array.append(mean_squared_error(y_test, predicted_value))

    from sklearn.metrics import mean_squared_log_error
    mean_squared_log_error_array.append(mean_squared_log_error(y_test, predicted_value))

    from sklearn.metrics import r2_score
    r2_score_array.append(r2_score(y_test, predicted_value))

    
    
print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")

#reg = XGBRegressor(n_estimators=50, booster='gbtree', learning_rate=0.1, max_delta_step=5)
reg = RandomForestRegressor(max_depth=50, random_state=1, n_estimators=100, max_features='auto')
reg.fit(final_training_dataset, final_training_target)

predicted_value = reg.predict(final_testing_dataset)
print(final_training_dataset)
print(final_training_target)
print(final_testing_dataset)
print(predicted_value)
predicted_value = pd.DataFrame(predicted_value,columns = ["SalePrice"])

final_predicted_values = pd.concat([testing_id, predicted_value], axis=1)
final_predicted_values = final_predicted_values.iloc[:,:]
print(final_predicted_values)
final_predicted_values.to_csv("test_results_RANDOMFOREST.csv", encoding='utf-8', index=False)

print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
print("explained_variance_score: ",explained_variance_score_array)
print("mean_squared_error: ",mean_squared_error_array)
print("mean_squared_log_error: ",mean_squared_log_error_array)
print("r2_score: ",r2_score_array)

print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
print("explained_variance_score: ",mean(explained_variance_score_array))
print("mean_squared_error: ",mean(mean_squared_error_array))
print("mean_squared_log_error: ",mean(mean_squared_log_error_array))
print("r2_score: ",mean(r2_score_array))



print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
x_train, x_test, y_train, y_test = train_test_split(final_training_dataset, final_training_target, test_size=.25)

reg = RandomForestRegressor(max_depth=50, random_state=1, n_estimators=100, max_features='auto')

#scoring_parameter = ['explained_variance','test_score','train_score']
scores = cross_validate(reg, final_training_dataset, final_training_target, cv=10, return_train_score=True)

print("\n\n\n\n\n")
print(scores.keys())
print("Test Score: ",scores['test_score'])
print("Test Score Average: ", scores['test_score'].mean())
print()
print("Train Score: ",scores['train_score'])
print("Train Score Average: ",scores['train_score'].mean())

print("\n\n\n\n\n\n")
scoring_parameter = ['explained_variance']
scores = cross_validate(reg, final_training_dataset, final_training_target, cv=10, return_train_score=True, scoring=scoring_parameter)

print("\n\n\n\n\n")
print(scores.keys())
print("Test Variance: ",scores['test_explained_variance'])
print("Test Variance Average: ",scores['test_explained_variance'].mean())
print()
print("Train Variance: ",scores['train_explained_variance'])
print("Train Variance Average: ",scores['train_explained_variance'].mean())

from sklearn.neighbors import KNeighborsRegressor
from statistics import mean
folds = 10


explained_variance_score_array = []
mean_squared_error_array = []
mean_squared_log_error_array = []
r2_score_array = []

final_training_dataset = pd.DataFrame(final_training_dataset)
final_training_target = pd.DataFrame(final_training_target)

final_testing_dataset = pd.DataFrame(final_testing_dataset)


final_training_dataset = final_training_dataset.values
final_training_target = final_training_target.values

final_testing_dataset = final_testing_dataset.values



for i in range(0,folds):
    x_train, x_test, y_train, y_test = train_test_split(final_training_dataset, final_training_target, test_size=.25)

    reg = KNeighborsRegressor(n_neighbors=50, weights="distance", p=2)
    reg.fit(x_train, y_train)
    
    predicted_value = reg.predict(x_test)
  #print(predicted_value[0])
  #print(predicted_value[0] - y_test[0])
    print(i)
    print(predicted_value)
    from sklearn.metrics import explained_variance_score
    explained_variance_score_array.append(explained_variance_score(y_test, predicted_value))

    from sklearn.metrics import mean_squared_error
    mean_squared_error_array.append(mean_squared_error(y_test, predicted_value))

    from sklearn.metrics import mean_squared_log_error
    mean_squared_log_error_array.append(mean_squared_log_error(y_test, predicted_value))

    from sklearn.metrics import r2_score
    r2_score_array.append(r2_score(y_test, predicted_value))

    
    
print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")

reg = KNeighborsRegressor(n_neighbors=50, weights="distance", p=2)
reg.fit(final_training_dataset, final_training_target)

predicted_value = reg.predict(final_testing_dataset)
print(final_training_dataset)
print(final_training_target)
print(final_testing_dataset)
print(predicted_value)
predicted_value = pd.DataFrame(predicted_value,columns = ["SalePrice"])

final_predicted_values = pd.concat([testing_id, predicted_value], axis=1)
final_predicted_values = final_predicted_values.iloc[:,:]
print(final_predicted_values)
final_predicted_values.to_csv("test_results_KNEIGHBORSREGRESSOR.csv", encoding='utf-8', index=False)

print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
print("explained_variance_score: ",explained_variance_score_array)
print("mean_squared_error: ",mean_squared_error_array)
print("mean_squared_log_error: ",mean_squared_log_error_array)
print("r2_score: ",r2_score_array)

print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
print("explained_variance_score: ",mean(explained_variance_score_array))
print("mean_squared_error: ",mean(mean_squared_error_array))
print("mean_squared_log_error: ",mean(mean_squared_log_error_array))
print("r2_score: ",mean(r2_score_array))



print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
x_train, x_test, y_train, y_test = train_test_split(final_training_dataset, final_training_target, test_size=.25)

reg = KNeighborsRegressor(n_neighbors=50, weights="distance", p=2)

#scoring_parameter = ['explained_variance','test_score','train_score']
scores = cross_validate(reg, final_training_dataset, final_training_target, cv=10, return_train_score=True)

print("\n\n\n\n\n")
print(scores.keys())
print("Test Score: ",scores['test_score'])
print("Test Score Average: ", scores['test_score'].mean())
print()
print("Train Score: ",scores['train_score'])
print("Train Score Average: ",scores['train_score'].mean())

print("\n\n\n\n\n\n")
scoring_parameter = ['explained_variance']
scores = cross_validate(reg, final_training_dataset, final_training_target, cv=10, return_train_score=True, scoring=scoring_parameter)

print("\n\n\n\n\n")
print(scores.keys())
print("Test Variance: ",scores['test_explained_variance'])
print("Test Variance Average: ",scores['test_explained_variance'].mean())
print()
print("Train Variance: ",scores['train_explained_variance'])
print("Train Variance Average: ",scores['train_explained_variance'].mean())


from sklearn.ensemble import GradientBoostingRegressor
from statistics import mean
folds = 10


explained_variance_score_array = []
mean_squared_error_array = []
mean_squared_log_error_array = []
r2_score_array = []

final_training_dataset = pd.DataFrame(final_training_dataset)
final_training_target = pd.DataFrame(final_training_target)

final_testing_dataset = pd.DataFrame(final_testing_dataset)


final_training_dataset = final_training_dataset.values
final_training_target = final_training_target.values

final_testing_dataset = final_testing_dataset.values



for i in range(0,folds):
    x_train, x_test, y_train, y_test = train_test_split(final_training_dataset, final_training_target, test_size=.25)

    reg = GradientBoostingRegressor(loss='ls', random_state=11, n_estimators = 100, learning_rate = 0.3, subsample = 0.3)
    reg.fit(x_train, y_train)
    
    predicted_value = reg.predict(x_test)
  #print(predicted_value[0])
  #print(predicted_value[0] - y_test[0])
    print(i)
    print(predicted_value)
    from sklearn.metrics import explained_variance_score
    explained_variance_score_array.append(explained_variance_score(y_test, predicted_value))

    from sklearn.metrics import mean_squared_error
    mean_squared_error_array.append(mean_squared_error(y_test, predicted_value))

    from sklearn.metrics import mean_squared_log_error
    mean_squared_log_error_array.append(mean_squared_log_error(y_test, predicted_value))

    from sklearn.metrics import r2_score
    r2_score_array.append(r2_score(y_test, predicted_value))

    
    
print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")

reg = GradientBoostingRegressor(loss='ls', random_state=11, n_estimators = 100, learning_rate = 0.3, subsample = 0.3)
reg.fit(final_training_dataset, final_training_target)

predicted_value = reg.predict(final_testing_dataset)
print(final_training_dataset)
print(final_training_target)
print(final_testing_dataset)
print(predicted_value)
predicted_value = pd.DataFrame(predicted_value,columns = ["SalePrice"])

final_predicted_values = pd.concat([testing_id, predicted_value], axis=1)
final_predicted_values = final_predicted_values.iloc[:,:]
print(final_predicted_values)
final_predicted_values.to_csv("test_results_GRADIENTBOOSTING.csv", encoding='utf-8', index=False)

print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
print("explained_variance_score: ",explained_variance_score_array)
print("mean_squared_error: ",mean_squared_error_array)
print("mean_squared_log_error: ",mean_squared_log_error_array)
print("r2_score: ",r2_score_array)

print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
print("explained_variance_score: ",mean(explained_variance_score_array))
print("mean_squared_error: ",mean(mean_squared_error_array))
print("mean_squared_log_error: ",mean(mean_squared_log_error_array))
print("r2_score: ",mean(r2_score_array))



print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
x_train, x_test, y_train, y_test = train_test_split(final_training_dataset, final_training_target, test_size=.25)

reg = GradientBoostingRegressor(loss='ls', random_state=11, n_estimators = 100, learning_rate = 0.3, subsample = 0.3)

#scoring_parameter = ['explained_variance','test_score','train_score']
scores = cross_validate(reg, final_training_dataset, final_training_target, cv=10, return_train_score=True)

print("\n\n\n\n\n")
print(scores.keys())
print("Test Score: ",scores['test_score'])
print("Test Score Average: ", scores['test_score'].mean())
print()
print("Train Score: ",scores['train_score'])
print("Train Score Average: ",scores['train_score'].mean())

print("\n\n\n\n\n\n")
scoring_parameter = ['explained_variance']
scores = cross_validate(reg, final_training_dataset, final_training_target, cv=10, return_train_score=True, scoring=scoring_parameter)

print("\n\n\n\n\n")
print(scores.keys())
print("Test Variance: ",scores['test_explained_variance'])
print("Test Variance Average: ",scores['test_explained_variance'].mean())
print()
print("Train Variance: ",scores['train_explained_variance'])
print("Train Variance Average: ",scores['train_explained_variance'].mean())


from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from statistics import mean
folds = 10


explained_variance_score_array = []
mean_squared_error_array = []
mean_squared_log_error_array = []
r2_score_array = []

final_training_dataset = pd.DataFrame(final_training_dataset)
final_training_target = pd.DataFrame(final_training_target)

final_testing_dataset = pd.DataFrame(final_testing_dataset)


final_training_dataset = final_training_dataset.values
final_training_target = final_training_target.values

final_testing_dataset = final_testing_dataset.values



for i in range(0,folds):
    x_train, x_test, y_train, y_test = train_test_split(final_training_dataset, final_training_target, test_size=.25)

    reg = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=8), loss='linear', random_state=11, n_estimators = 100, learning_rate = 0.9)
    reg.fit(x_train, y_train)
    
    predicted_value = reg.predict(x_test)
  #print(predicted_value[0])
  #print(predicted_value[0] - y_test[0])
    print(i)
    print(predicted_value)
    from sklearn.metrics import explained_variance_score
    explained_variance_score_array.append(explained_variance_score(y_test, predicted_value))

    from sklearn.metrics import mean_squared_error
    mean_squared_error_array.append(mean_squared_error(y_test, predicted_value))

    from sklearn.metrics import mean_squared_log_error
    mean_squared_log_error_array.append(mean_squared_log_error(y_test, predicted_value))

    from sklearn.metrics import r2_score
    r2_score_array.append(r2_score(y_test, predicted_value))

    
    
print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")

reg = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=8), loss='linear', random_state=11, n_estimators = 100, learning_rate = 0.9)
reg.fit(final_training_dataset, final_training_target)

predicted_value = reg.predict(final_testing_dataset)
print(final_training_dataset)
print(final_training_target)
print(final_testing_dataset)
print(predicted_value)
predicted_value = pd.DataFrame(predicted_value,columns = ["SalePrice"])

final_predicted_values = pd.concat([testing_id, predicted_value], axis=1)
final_predicted_values = final_predicted_values.iloc[:,:]
print(final_predicted_values)
final_predicted_values.to_csv("test_results_ADABOOSTING.csv", encoding='utf-8', index=False)

print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
print("explained_variance_score: ",explained_variance_score_array)
print("mean_squared_error: ",mean_squared_error_array)
print("mean_squared_log_error: ",mean_squared_log_error_array)
print("r2_score: ",r2_score_array)

print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
print("explained_variance_score: ",mean(explained_variance_score_array))
print("mean_squared_error: ",mean(mean_squared_error_array))
print("mean_squared_log_error: ",mean(mean_squared_log_error_array))
print("r2_score: ",mean(r2_score_array))



print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
x_train, x_test, y_train, y_test = train_test_split(final_training_dataset, final_training_target, test_size=.25)

reg = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=8), loss='linear', random_state=11, n_estimators = 100, learning_rate = 0.9)

#scoring_parameter = ['explained_variance','test_score','train_score']
scores = cross_validate(reg, final_training_dataset, final_training_target, cv=10, return_train_score=True)

print("\n\n\n\n\n")
print(scores.keys())
print("Test Score: ",scores['test_score'])
print("Test Score Average: ", scores['test_score'].mean())
print()
print("Train Score: ",scores['train_score'])
print("Train Score Average: ",scores['train_score'].mean())

print("\n\n\n\n\n\n")
scoring_parameter = ['explained_variance']
scores = cross_validate(reg, final_training_dataset, final_training_target, cv=10, return_train_score=True, scoring=scoring_parameter)

print("\n\n\n\n\n")
print(scores.keys())
print("Test Variance: ",scores['test_explained_variance'])
print("Test Variance Average: ",scores['test_explained_variance'].mean())
print()
print("Train Variance: ",scores['train_explained_variance'])
print("Train Variance Average: ",scores['train_explained_variance'].mean())




from sklearn.tree import DecisionTreeRegressor
from statistics import mean
folds = 10


explained_variance_score_array = []
mean_squared_error_array = []
mean_squared_log_error_array = []
r2_score_array = []

final_training_dataset = pd.DataFrame(final_training_dataset)
final_training_target = pd.DataFrame(final_training_target)

final_testing_dataset = pd.DataFrame(final_testing_dataset)


final_training_dataset = final_training_dataset.values
final_training_target = final_training_target.values

final_testing_dataset = final_testing_dataset.values



for i in range(0,folds):
    x_train, x_test, y_train, y_test = train_test_split(final_training_dataset, final_training_target, test_size=.25)

    reg = DecisionTreeRegressor(criterion='mse', random_state=4, max_depth=10, min_samples_leaf=20, max_features=5)
    reg.fit(x_train, y_train)
    
    predicted_value = reg.predict(x_test)
  #print(predicted_value[0])
  #print(predicted_value[0] - y_test[0])
    print(i)
    print(predicted_value)
    from sklearn.metrics import explained_variance_score
    explained_variance_score_array.append(explained_variance_score(y_test, predicted_value))

    from sklearn.metrics import mean_squared_error
    mean_squared_error_array.append(mean_squared_error(y_test, predicted_value))

    from sklearn.metrics import mean_squared_log_error
    mean_squared_log_error_array.append(mean_squared_log_error(y_test, predicted_value))

    from sklearn.metrics import r2_score
    r2_score_array.append(r2_score(y_test, predicted_value))

    
    
print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")

reg = DecisionTreeRegressor(criterion='mse', random_state=4, max_depth=10, min_samples_leaf=20, max_features=5)
reg.fit(final_training_dataset, final_training_target)

predicted_value = reg.predict(final_testing_dataset)
print(final_training_dataset)
print(final_training_target)
print(final_testing_dataset)
print(predicted_value)
predicted_value = pd.DataFrame(predicted_value,columns = ["SalePrice"])

final_predicted_values = pd.concat([testing_id, predicted_value], axis=1)
final_predicted_values = final_predicted_values.iloc[:,:]
print(final_predicted_values)
final_predicted_values.to_csv("test_results_GRADIENTBOOSTING.csv", encoding='utf-8', index=False)

print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
print("explained_variance_score: ",explained_variance_score_array)
print("mean_squared_error: ",mean_squared_error_array)
print("mean_squared_log_error: ",mean_squared_log_error_array)
print("r2_score: ",r2_score_array)

print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
print("explained_variance_score: ",mean(explained_variance_score_array))
print("mean_squared_error: ",mean(mean_squared_error_array))
print("mean_squared_log_error: ",mean(mean_squared_log_error_array))
print("r2_score: ",mean(r2_score_array))



print("\n :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
x_train, x_test, y_train, y_test = train_test_split(final_training_dataset, final_training_target, test_size=.25)

reg = DecisionTreeRegressor(criterion='mse', random_state=4, max_depth=10, min_samples_leaf=20, max_features=5)

#scoring_parameter = ['explained_variance','test_score','train_score']
scores = cross_validate(reg, final_training_dataset, final_training_target, cv=10, return_train_score=True)

print("\n\n\n\n\n")
print(scores.keys())
print("Test Score: ",scores['test_score'])
print("Test Score Average: ", scores['test_score'].mean())
print()
print("Train Score: ",scores['train_score'])
print("Train Score Average: ",scores['train_score'].mean())

print("\n\n\n\n\n\n")
scoring_parameter = ['explained_variance']
scores = cross_validate(reg, final_training_dataset, final_training_target, cv=10, return_train_score=True, scoring=scoring_parameter)

print("\n\n\n\n\n")
print(scores.keys())
print("Test Variance: ",scores['test_explained_variance'])
print("Test Variance Average: ",scores['test_explained_variance'].mean())
print()
print("Train Variance: ",scores['train_explained_variance'])
print("Train Variance Average: ",scores['train_explained_variance'].mean())



