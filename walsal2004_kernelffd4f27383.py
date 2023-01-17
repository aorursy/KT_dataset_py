#Imports

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

#data_description = pd.read_csv('dataset/data_description.txt')

submission = pd.read_csv('../input/sample_submission.csv')
train_data.head()
test_data.head()
numeric_features = train_data.select_dtypes(include=['int64','float64'])

categorical_features = train_data.select_dtypes(include=['object'])
print(len(list(numeric_features.columns)))

print(len(list(categorical_features.columns)))
train_data.shape
train_data.isnull().sum()
test_data.shape
test_data.isnull().sum()
#correlation matrix

corrmat = train_data.corr()

f, ax = plt.subplots(figsize=(18, 15))

sns.heatmap(corrmat, vmax=.8, square=True);
plt.figure(figsize=(33,30))

cor = train_data.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
train_data.groupby(by="MSZoning").count()
#missing amount for train set

missing= train_data.isnull().sum().sort_values(ascending=False)

percentage = (train_data.isnull().sum()/ train_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([missing, percentage], axis=1, keys=['Missing Amt', '%'])

missing_data.head(20)
#missing amount for test set

missing= test_data.isnull().sum().sort_values(ascending=False)

percentage = (test_data.isnull().sum()/ test_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([missing, percentage], axis=1, keys=['Missing Amt', '%'])

missing_data.head(20)
train_data['MSZoning'].unique()
test_data['MSZoning'].unique()
train_data.loc[train_data["PoolQC"].isnull(), "PoolQC"] = 'No_Pool'

train_data.loc[train_data["Alley"].isnull(), "Alley"] = 'No_alley'

train_data.loc[train_data["Fence"].isnull(), "Fence"] = 'No_Fence'

train_data.loc[train_data["FireplaceQu"].isnull(), "FireplaceQu"] = 'No_Fireplace'

train_data.loc[train_data["MiscFeature"].isnull(), "MiscFeature"] = 'None_feature '

train_data.loc[train_data["GarageCond"].isnull(), "GarageCond"] = 'No_Garage_Cond'

train_data.loc[train_data["GarageType"].isnull(), "GarageType"] = 'No_Garage_Type'

train_data["GarageYrBlt"] = train_data["GarageYrBlt"].replace(np.NaN, train_data["GarageYrBlt"].median())

train_data.loc[train_data["GarageFinish"].isnull(), "GarageFinish"] = 'No_Garage_Finish'

train_data.loc[train_data["GarageQual"].isnull(), "GarageQual"] = 'No_Garage_Qual'

train_data.loc[train_data["BsmtExposure"].isnull(), "BsmtExposure"] = 'No_Basement_Exposure'

train_data.loc[train_data["BsmtFinType2"].isnull(), "BsmtFinType2"] = 'No_Basement_FinType2'

train_data.loc[train_data["BsmtFinType1"].isnull(), "BsmtFinType1"] = 'No_Basement_FinType1'

train_data.loc[train_data["BsmtCond"].isnull(), "BsmtCond"] = 'No_Basement_BsmtCond'

train_data.loc[train_data["BsmtQual"].isnull(), "BsmtQual"] = 'No_Basement_BsmtQual'

train_data["MasVnrArea"] = train_data["MasVnrArea"].replace(np.NaN, train_data["MasVnrArea"].mean())

train_data["MasVnrType"] = train_data["MasVnrType"].fillna('None')

train_data["Electrical"] = train_data["Electrical"].fillna('SBrkr')
test_data.loc[test_data["PoolQC"].isnull(), "PoolQC"] = 'No_Pool'

test_data.loc[test_data["Alley"].isnull(), "Alley"] = 'No_alley'

test_data.loc[test_data["Fence"].isnull(), "Fence"] = 'No_Fence'

test_data.loc[test_data["FireplaceQu"].isnull(), "FireplaceQu"] = 'No_Fireplace'

test_data.loc[test_data["MiscFeature"].isnull(), "MiscFeature"] = 'None_feature'

test_data.loc[test_data["GarageCond"].isnull(), "GarageCond"] = 'No_Garage_Cond'

test_data.loc[test_data["GarageType"].isnull(), "GarageType"] = 'No_Garage_Type'

test_data["GarageYrBlt"] = test_data["GarageYrBlt"].replace(np.NaN, train_data["GarageYrBlt"].median())

test_data.loc[test_data["GarageFinish"].isnull(), "GarageFinish"] = 'No_Garage_Finish'

test_data.loc[test_data["GarageQual"].isnull(), "GarageQual"] = 'No_Garage_Qual'

test_data.loc[test_data["BsmtExposure"].isnull(), "BsmtExposure"] = 'No_Basement_Exposure'

test_data.loc[test_data["BsmtFinType2"].isnull(), "BsmtFinType2"] = 'No_Basement_FinType2'

test_data.loc[test_data["BsmtFinType1"].isnull(), "BsmtFinType1"] = 'No_Basement_FinType1'

test_data.loc[test_data["BsmtCond"].isnull(), "BsmtCond"] = 'No_Basement_BsmtCond'

test_data.loc[test_data["BsmtQual"].isnull(), "BsmtQual"] = 'No_Basement_BsmtQual'

test_data["MasVnrArea"] = test_data["MasVnrArea"].replace(np.NaN, train_data["MasVnrArea"].mean())

test_data["MasVnrType"] = test_data["MasVnrType"].fillna('None')

test_data["Electrical"] = test_data["Electrical"].fillna('SBrkr')

test_data["MSZoning"] = test_data["MSZoning"].fillna('RL')

test_data["Utilities"] = test_data["Utilities"].fillna('AllPub')

test_data["BsmtFullBath"] = test_data["BsmtFullBath"].fillna(0)

test_data["BsmtHalfBath"] = test_data["BsmtHalfBath"].fillna(0)

test_data["Functional"] = test_data["Functional"].fillna('Typ')

test_data["Exterior1st"] = test_data["Exterior1st"].fillna('VinylSd')

test_data["KitchenQual"] = test_data["KitchenQual"].fillna('TA')

test_data["Exterior2nd"] = test_data["Exterior2nd"].fillna('VinylSd')

test_data["BsmtFinSF1"] = test_data["BsmtFinSF1"].fillna(383.5)

test_data["GarageArea"] = test_data["GarageArea"].fillna(480.0)

test_data["GarageCars"] = test_data["GarageCars"].fillna(2.0)

test_data["SaleType"] = test_data["SaleType"].fillna('WD')

test_data["TotalBsmtSF"] = test_data["TotalBsmtSF"].fillna(991.5)

test_data["BsmtUnfSF"] = test_data["BsmtUnfSF"].fillna(477.5)

test_data["BsmtFinSF2"] = test_data["BsmtFinSF2"].fillna(46.54931506849315)
#from fancyimpute import KNN 

#We use the train dataframe 

#fancy impute removes column names.

#train_cols = list(train_data)

# Use 5 nearest rows which have a feature to fill in each row's

# missing features

#train_data = pd.DataFrame(KNN(k=5).complete(train_data))

#train_data.columns = train_cols
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

train_int = train_data.select_dtypes(include=['int','float'])

#Step-1: Split the dataset that contains the missing values and no missing values are test and train respectively.

x_train = train_int[train_int['LotFrontage'].notnull()].drop(columns='LotFrontage')

y_train = train_int[train_int['LotFrontage'].notnull()]['LotFrontage']

x_test = train_int[train_int['LotFrontage'].isnull()].drop(columns='LotFrontage')

y_test = train_int[train_int['LotFrontage'].isnull()]['LotFrontage']

#Step-2: Train the machine learning algorithm

linreg.fit(x_train, y_train)

#Step-3: Predict the missing values in the attribute of the test data.

predicted = linreg.predict(x_test)

#Step-4: Let’s obtain the complete dataset by combining with the target attribute.

train_data.LotFrontage[train_data.LotFrontage.isnull()] = predicted
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

train_int = test_data.select_dtypes(include=['int','float'])

#Step-1: Split the dataset that contains the missing values and no missing values are test and train respectively.

x_train = train_int[train_int['LotFrontage'].notnull()].drop(columns='LotFrontage')

y_train = train_int[train_int['LotFrontage'].notnull()]['LotFrontage']

x_test = train_int[train_int['LotFrontage'].isnull()].drop(columns='LotFrontage')

y_test = train_int[train_int['LotFrontage'].isnull()]['LotFrontage']

#Step-2: Train the machine learning algorithm

linreg.fit(x_train, y_train)

#Step-3: Predict the missing values in the attribute of the test data.

predicted = linreg.predict(x_test)

#Step-4: Let’s obtain the complete dataset by combining with the target attribute.

test_data.LotFrontage[test_data.LotFrontage.isnull()] = predicted
train_data.shape, test_data.shape
train_data_dummy = pd.get_dummies(train_data)

test_data_dummy = pd.get_dummies(test_data)

test_data_dummy['SalePrice'] = 0.0
train_data_dummy.shape, test_data_dummy.shape
train_data_dummy = train_data_dummy.reindex(columns = test_data_dummy.columns, fill_value=0)
train_data_dummy.shape, test_data_dummy.shape
train_data_dummy.head()
train_data_dummy.info()
test_data_dummy.head()
train_data_dummy.shape, test_data_dummy.shape
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import PolynomialFeatures



ss = StandardScaler()

X = train_data_dummy.drop(['SalePrice'],axis=1)

X = ss.fit_transform(X)

y = train_data_dummy['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, shuffle=True)
test_data_dummy = test_data_dummy.drop(['SalePrice'],axis=1)

test_data_dummy = ss.transform(test_data_dummy)
list_Of_Scores = list()
lr = LinearRegression()

lr.fit(X_train, y_train) # i have betas

print('train score : ',lr.score(X_train, y_train))

print('test score : ',lr.score(X_test, y_test))



temp_results = {'Name':'LinearRegression','Train Score': lr.score(X_train, y_train),'Test Score':lr.score(X_test, y_test)}

list_Of_Scores.append(temp_results)
X.shape ,y.shape
from sklearn.model_selection import cross_val_score



lr2 = LinearRegression()

print('train score: ', np.mean(cross_val_score(lr2, X_train, y_train, cv = 10)))

lr2.fit(X_train, y_train)

print('test score : ',lr2.score(X_test, y_test))
pred = lr.predict(test_data_dummy)
pred
submission.head()
submission = submission.drop(['SalePrice'],axis=1)
submission['SalePrice'] = pred
submission.head()
#submission.to_csv('LR.csv')

from sklearn.linear_model import LassoCV

from sklearn.model_selection import GridSearchCV



reg = LassoCV()

parm_grid = {'eps': [0.001, 0.002,0.003 ],'n_alphas':[30, 40, 50], 

            'max_iter':[700, 800, 900]}

grid = GridSearchCV(reg, parm_grid, cv=5)

grid.fit(X, y)
grid.best_params_
grid.best_score_
gridlass = grid.best_estimator_

#test_data_dummy = test_data_dummy.drop(['SalePrice'],axis=1)

pred = gridlass.predict(test_data_dummy)

submission = submission.drop(['SalePrice'],axis=1)

submission['SalePrice'] = pred
submission.head()
submission.to_csv('submissionLassoCVgrid2.csv')
X_train.shape ,y_train.shape
from sklearn.linear_model import LassoCV



#reg = LassoCV()

#parm_grid = {'eps': [0.01, 0.05,0.001 ],'n_alphas':[50, 100, 200], 'precompute':['array-like', 'auto'],

#            'max_iter':[900, 1000, 1200], 'tol': [0.0001, 0.001, 0.01]}

#grid = GridSearchCV(reg, parm_grid, cv=5)

#grid.fit(X, y)



reg = LassoCV()

reg.fit(X, y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(X_train,y_train))

print("Best score using built-in LassoCV test: %f" %reg.score(X_test,y_test))



temp_results = {'Name':'LassoCV','Train Score': reg.score(X_train,y_train),'Test Score':reg.score(X_test,y_test)}

list_Of_Scores.append(temp_results)

#coef = pd.Series(reg.coef_, index = X.columns)
pred = reg.predict(test_data_dummy)

submission = submission.drop(['SalePrice'],axis=1)

submission['SalePrice'] = pred

submission.head()
submission.to_csv('submissionLassoCV.csv')
from sklearn.linear_model import RidgeCV

reg1 = RidgeCV()

reg1.fit(X_train,y_train)

reg1.score(X_train,y_train), reg1.score(X_test,y_test)

temp_results = {'Name':'RidgeCV','Train Score': reg1.score(X_train,y_train),'Test Score':reg1.score(X_test,y_test)}

list_Of_Scores.append(temp_results)
from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV

parm_grid = {'gamma': [0.01, 0.05, 0.1, 1,0.001 ]}

svm_rbf = SVR()

grid = GridSearchCV(svm_rbf, parm_grid, cv=5)

grid.fit(X, y)



#clf = SVR(kernel='linear',degree=3,gamma='auto_deprecated', C=1.0, epsilon=0.2)

#cross_val_score(clf, X_train, y_train, cv=5).mean()
grid.best_params_
grid.best_score_
temp_results = {'Name':'SVM','Train Score': grid.best_score_ ,'Test Score':'None'}

list_Of_Scores.append(temp_results)
# use RBF kernal 

svm_rbf = svm.SVC(kernel='rbf')

cross_val_score(svm_rbf, X_train, y_train, cv=5).mean()
from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor(n_neighbors=2)

neigh.fit(X_train, y_train)

print('train score : ',neigh.score(X_train, y_train))

print('test score : ',neigh.score(X_test, y_test))

temp_results = {'Name':'KNeighborsRegressor','Train Score': neigh.score(X_train, y_train),'Test Score':neigh.score(X_test, y_test)}

list_Of_Scores.append(temp_results)
neigh2 = KNeighborsRegressor(n_neighbors=2)

print('train score: ', np.mean(cross_val_score(neigh2, X_train, y_train, cv = 10)))

neigh2.fit(X_train, y_train)

print('test score : ',neigh2.score(X_test, y_test))
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV



regressor = DecisionTreeRegressor(random_state=0)

param_grid = {"criterion": ["mse", "mae"],

              "min_samples_split": [10, 20, 40],

              "max_depth": [2, 6, 8],

              "min_samples_leaf": [20, 40, 100],

              "max_leaf_nodes": [5, 20, 100],

              }

grid_cv_dtm = GridSearchCV(regressor, param_grid, cv=5)

grid_cv_dtm.fit(X_train, y_train)

#print('train score: ', np.mean(cross_val_score(regressor, X_train, y_train, cv = 10)))

#regressor.fit(X_train, y_train)

#print('test score : ',regressor.score(X_test, y_test))
temp_results = {'Name':'DecisionTreeRegressor','Train Score': grid_cv_dtm.score(X_train, y_train),'Test Score':grid_cv_dtm.score(X_test, y_test)}

list_Of_Scores.append(temp_results)
print("R-Squared::{}".format(grid_cv_dtm.best_score_))

print("Best Hyperparameters::\n{}".format(grid_cv_dtm.best_params_))
td_grid = grid_cv_dtm.best_estimator_

pred = td_grid.predict(test_data_dummy)

submission = submission.drop(['SalePrice'],axis=1)

submission['SalePrice'] = pred
submission.head()
submission.to_csv('submissiondtgrid.csv')
from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor()

rf.fit(X_train, y_train)

print('train score : ',rf.score(X_train, y_train))

print('test score : ',rf.score(X_test, y_test))



temp_results = {'Name':'RandomForestRegressor','Train Score':rf.score(X_train, y_train),'Test Score':rf.score(X_test, y_test)}

list_Of_Scores.append(temp_results)
from sklearn.model_selection import GridSearchCV



param_grid = [

 {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},

 ]
forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)
grid_search.best_params_
rf = grid_search.best_estimator_
grid_search.best_score_
temp_results = {'Name':'RandomForestRegressor','Train Score':grid_search.best_score_,'Test Score':'None'}

list_Of_Scores.append(temp_results)
scores_df = pd.DataFrame(list_Of_Scores)
scores_df