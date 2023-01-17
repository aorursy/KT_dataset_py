import pandas as pd

import numpy as np

import seaborn as sns

import scipy as sp

import sklearn



from matplotlib import pyplot as plt

from scipy.stats import norm, skew

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn import neighbors

from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import neural_network

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

import datetime

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import confusion_matrix

pd.set_option('display.max_columns', 500)

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
train.shape
train.dtypes
train.isna().sum()
train = train.dropna(thresh=500, axis=1)

train.shape
#train.nunique()

#Ok you may not like it but you might have to retrospect each and every column
train = train.drop(['Id'],axis = 1)
train['SaleCondition'].value_counts()
#Deleting some columns

train = train[['MSSubClass','LotFrontage','LotArea','LotShape','Neighborhood','HouseStyle','OverallCond','YearBuilt','YearRemodAdd','Exterior1st','Exterior2nd','MasVnrType','Foundation','BsmtQual','BsmtFinType1','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','HeatingQC','1stFlrSF','2ndFlrSF','GrLivArea','FullBath','KitchenQual','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageFinish','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','MoSold','YrSold','SalePrice']]
train.head()
train.shape
#Individual column analysis

#sns.distplot(train['MSSubClass'],hist = False)

#train['OpenPorchSF'].value_counts()
def groups(series):

    if series <= 50:

        return "Low"

    elif series > 50 and series <= 100:

        return "Medium"

    else:

        return "High"



train['SubClass'] = train['MSSubClass'].apply(groups)
def groups(series):

    if series not in ['1Story','2Story','1.5Fin','SLvl','SFoyer']:

        return "Others"

    else:

        return series

    

train['HouseStyle'] = train['HouseStyle'].apply(groups)
train['NormLotArea'] = (train['LotArea'] - min(train['LotArea']))/(max(train['LotArea'])-min(train['LotArea']))
def groups(series):

    if series == "IR1" or series == "IR2" or series == "IR3":

        return "Irregular"

    else:

        return "Regular"

    

train['NewLotShape'] = train['LotShape'].apply(groups)
def groups(series):

    if series == "IR1" or series == "IR2" or series == "IR3":

        return "Irregular"

    else:

        return "Regular"

    

train['NewLotShape'] = train['LotShape'].apply(groups)
#Try Year Built also and see results

train['BldgAge'] = train['YrSold']-train['YearRemodAdd']
train.head()
#train['Exterior1st'] .value_counts()

train[['Exterior1st']] = train[['Exterior1st']].replace(['WdShing','Stucco','AsbShng','BrkComm','Stone','CBlock','ImStucc','AsphShn'], 'Others')

train[['Foundation']] = train[['Foundation']].replace(['Slab','Stone','Wood'],'Others')
def groups(series):

    if series > 0:

        return 1

    else:

        return 0



train['WoodDeckSF'] = train['WoodDeckSF'].apply(groups)

train['OpenPorchSF'] = train['OpenPorchSF'].apply(groups)
train = train.drop(['YrSold','MSSubClass','LotFrontage','LotArea','LotShape','YearBuilt','YearRemodAdd','Exterior2nd','MasVnrType','TotalBsmtSF','GrLivArea','GarageYrBlt'],axis = 1)
#transofrming all object dtypes to categorical

def changeDtypes(df,from_dtype,to_dtype):

    #changes inplace, affects the passed dataFrame

#     df[df.select_dtypes(from_dtype).columns] = df.select_dtypes(from_dtype).astype(to_dtype)

    df[df.select_dtypes(from_dtype).columns] = df.select_dtypes(from_dtype).apply(lambda x: x.astype(to_dtype))

    

    

changeDtypes(train,'object','category')

train.dtypes
train.head()
train = train.dropna()

train.isna().sum()
all_dummies = ['Neighborhood','HouseStyle','Exterior1st','Foundation','BsmtQual','BsmtFinType1','HeatingQC','KitchenQual','GarageFinish','MoSold','SubClass','NewLotShape']
def dummyvars(dummy_cols):

    dummy_list = pd.DataFrame()

    for i in dummy_cols:

        dummy = pd.get_dummies(train[i], prefix=i).iloc[:, 1:]

        dummy_list = pd.concat([dummy_list,dummy],axis=1)

    return dummy_list



dummy_list = dummyvars(all_dummies)
train = train.drop(['Neighborhood','HouseStyle','Exterior1st','Foundation','BsmtQual','BsmtFinType1','HeatingQC','KitchenQual','GarageFinish','MoSold','SubClass','NewLotShape'],axis=1)
dummy_list.head()
train = pd.concat([train,dummy_list],axis = 1)
#Now the corrplot

#Using Pearson Correlation

#X-X correlation

train_x = train.drop(['SalePrice'],axis = 1)

train_y = pd.DataFrame(train['SalePrice'])

corr_matrix = train_x.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.6)]

to_drop.append('KitchenQual_Gd')

to_drop.remove('KitchenQual_TA')
#corr_matrix['KitchenQual_TA'].sort_values(ascending = False)
#X-Y correlation

corr_matrix = train.corr().abs()

corr_target = corr_matrix['SalePrice']

relevant_features = corr_target[corr_target > 0.5]

relevant_features
train = train.drop(to_drop,axis = 1)
train_x = train.drop(['SalePrice'],axis = 1)

train_y = pd.DataFrame(train['SalePrice'])

model = LinearRegression()

model.fit(train_x,train_y)

r_square = model.score(train_x,train_y)

1/(1-r_square)
#Grid Search - Linear Regression

#model = LinearRegression()

#param_grid = {}

#grid = GridSearchCV(model, param_grid, cv = 5, scoring = 'r2')

#grid.fit(train_x,train_y)

#max(grid.cv_results_['mean_test_score'])
#Grid Search - L1 Regularization (Lasso)

#model = Lasso()

#param_grid = {'alpha':[0.001,0.01,0.1,1,10,20,50,100]}

#grid = GridSearchCV(model, param_grid, cv = 5, scoring = 'r2')

#grid.fit(train_x,train_y)

#max(grid.cv_results_['mean_test_score'])
#Grid Search - L2 Regularization (Ridge)

#model = Ridge()

#param_grid = {'alpha':[0.001,0.01,0.1,1,10,20,50,100]}

#grid = GridSearchCV(model, param_grid, cv = 5, scoring = 'r2')

#grid.fit(train_x,train_y)

#max(grid.cv_results_['mean_test_score'])
#Grid Search - Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

#model = RandomForestRegressor()

#param_grid = {'max_depth':[5,10],

#              'n_estimators':[50,100,150,200],

#              'random_state':[42]}

#grid = GridSearchCV(model, param_grid, cv = 5, scoring = 'r2')

#grid.fit(train_x,train_y)

#max(grid.cv_results_['mean_test_score'])
#Grid Search - Gradient Boosting Regressor

from sklearn.ensemble import GradientBoostingRegressor

#model = GradientBoostingRegressor()

#param_grid = {'max_depth':[5,10],

#              'n_estimators':[50,100,150,200],

#              'learning_rate':[0.1,0.5,1]}

#grid = GridSearchCV(model, param_grid, cv = 5, scoring = 'r2')

#grid.fit(train_x,train_y)

#max(grid.cv_results_['mean_test_score'])
#grid.best_params_
#Grid Search - Adaptive Boost Regression

#from sklearn.ensemble import AdaBoostRegressor

#model = AdaBoostRegressor()

#param_grid = {'learning_rate':[0.1,0.5,1],

#              'n_estimators':[50,100,150,200],

#              'loss':['linear','square','exponential']}

#grid = GridSearchCV(model, param_grid, cv = 5, scoring = 'r2')

#grid.fit(train_x,train_y)

#max(grid.cv_results_['mean_test_score'])
test = test[['Id','MSSubClass','LotFrontage','LotArea','LotShape','Neighborhood','HouseStyle','OverallCond','YearBuilt','YearRemodAdd','Exterior1st','Exterior2nd','MasVnrType','Foundation','BsmtQual','BsmtFinType1','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','HeatingQC','1stFlrSF','2ndFlrSF','GrLivArea','FullBath','KitchenQual','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageFinish','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','MoSold','YrSold']]
def groups(series):

    if series <= 50:

        return "Low"

    elif series > 50 and series <= 100:

        return "Medium"

    else:

        return "High"



test['SubClass'] = test['MSSubClass'].apply(groups)
temp_train = pd.read_csv('../input/train.csv')

test['NormLotArea'] = (test['LotArea'] - min(temp_train['LotArea']))/(max(temp_train['LotArea'])-min(temp_train['LotArea']))
def groups(series):

    if series == "IR1" or series == "IR2" or series == "IR3":

        return "Irregular"

    else:

        return "Regular"

    

test['NewLotShape'] = test['LotShape'].apply(groups)
def groups(series):

    if series not in ['1Story','2Story','1.5Fin','SLvl','SFoyer']:

        return "Others"

    else:

        return series

    

test['HouseStyle'] = test['HouseStyle'].apply(groups)
test['BldgAge'] = test['YrSold']-test['YearRemodAdd']
test[['Exterior1st']] = test[['Exterior1st']].replace(['WdShing','Stucco','AsbShng','BrkComm','Stone','CBlock','ImStucc','AsphShn'], 'Others')

test[['Foundation']] = test[['Foundation']].replace(['Slab','Stone','Wood'],'Others')
def groups(series):

    if series > 0:

        return 1

    else:

        return 0



test['WoodDeckSF'] = test['WoodDeckSF'].apply(groups)

test['OpenPorchSF'] = test['OpenPorchSF'].apply(groups)
test = test.drop(['YrSold','MSSubClass','LotFrontage','LotArea','LotShape','YearBuilt','YearRemodAdd','Exterior2nd','MasVnrType','TotalBsmtSF','GrLivArea','GarageYrBlt'],axis = 1)
#transofrming all object dtypes to categorical

def changeDtypes(df,from_dtype,to_dtype):

    #changes inplace, affects the passed dataFrame

#     df[df.select_dtypes(from_dtype).columns] = df.select_dtypes(from_dtype).astype(to_dtype)

    df[df.select_dtypes(from_dtype).columns] = df.select_dtypes(from_dtype).apply(lambda x: x.astype(to_dtype))

    

    

changeDtypes(test,'object','category')

test.dtypes
test.isna().sum()
test["Exterior1st"].fillna(temp_train["Exterior1st"].mode().iloc[0], inplace=True)

test["BsmtQual"].fillna(temp_train["BsmtQual"].mode().iloc[0], inplace=True)

test["BsmtFinType1"].fillna(temp_train["BsmtFinType1"].mode().iloc[0], inplace=True)

test["BsmtFinSF1"].fillna(temp_train["BsmtFinSF1"].median(), inplace=True)

test["BsmtUnfSF"].fillna(temp_train["BsmtUnfSF"].median(), inplace=True)

test["KitchenQual"].fillna(temp_train["KitchenQual"].mode().iloc[0], inplace=True)

test["GarageFinish"].fillna(temp_train["GarageFinish"].mode().iloc[0], inplace=True)

test["GarageCars"].fillna(temp_train["GarageCars"].median(), inplace=True)

test["GarageArea"].fillna(temp_train["GarageArea"].median(), inplace=True)
def dummyvars(dummy_cols):

    dummy_list = pd.DataFrame()

    for i in dummy_cols:

        dummy = pd.get_dummies(test[i], prefix=i).iloc[:, 1:]

        dummy_list = pd.concat([dummy_list,dummy],axis=1)

    return dummy_list



dummy_list = dummyvars(all_dummies)
test = pd.concat([test,dummy_list],axis = 1)
test = test.drop(to_drop,axis = 1)
test = test.drop(['Neighborhood','HouseStyle','Exterior1st','Foundation','BsmtQual','BsmtFinType1','HeatingQC','KitchenQual','GarageFinish','MoSold','SubClass','NewLotShape'],axis=1)
test_model = test.drop(['Id'],axis = 1)
boost = GradientBoostingRegressor(learning_rate = 0.1, n_estimators = 150, max_depth = 5)

boost.fit(train_x,train_y)

forest = RandomForestRegressor(max_depth = 10, n_estimators = 200, random_state = 42)

forest.fit(train_x,train_y)

lasso = Lasso(alpha = 50)

lasso.fit(train_x,train_y)
#Stacked Model Sample (Citation: Nithish Kandagadla):

def stackModelsAndPredict(listOfWeights,listOfModels,Test):

    AllModelPreds  = np.zeros(Test.shape[0])

    for weight,model in zip(listOfWeights,listOfModels):

      predictions = model.predict(Test)

      weightedPreds = weight*predictions

      AllModelPreds = np.column_stack((AllModelPreds,weightedPreds))

      

    return np.sum(AllModelPreds,axis = 1)



stackedPreds = stackModelsAndPredict([0.5,0.3,0.2],[boost,forest,lasso],test_model)
test['SalePrice'] = stackedPreds

test = test[['Id','SalePrice']]