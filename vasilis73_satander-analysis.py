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
train.drop('ID',axis=1,inplace=True)
# outcome
y = train.target
# predictors
X = train.drop('target',axis=1)
# split to training and testing datasets
from sklearn.model_selection import train_test_split
train_X,  test_X, train_y, test_y = train_test_split(X,y,test_size=0.30,random_state=0)

# # Feature Scaling
# standardize the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(test_X)
train_y_reshape = train_y.values.reshape(-1, 1)
train_y_reshape = sc_y.fit_transform(train_y_reshape)
test_y_reshape = test_y.values.reshape(-1, 1)
test_y_reshape = sc_y.transform(test_y_reshape)
# Normalize the data
from sklearn.preprocessing import Normalizer
#sc = Normalizer()
#train_X = sc.fit_transform(train_X)
#test_X = sc.transform(test_X)
# Applying PCA to identify the right number of features
from sklearn.decomposition import PCA
#pca = PCA()
#train_X = pca.fit_transform(train_X)
#test_X = pca.transform(test_X)
#s = pca.explained_variance_ratio_
#sum=0
#comp=0

#for _ in s:
#    sum += _
#    comp += 1
#    if(sum>=0.95):
#        break
# Right number of features
#comp
# Applying PCA to train and test datasets
from sklearn.decomposition import PCA
pca = PCA(n_components = 1197)
train_X = pca.fit_transform(train_X)
test_X = pca.transform(test_X)
# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#lda = LDA(n_components = 2)
#X_train = lda.fit_transform(X_train, y_train)
#X_test = lda.transform(X_test)
# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
#kpca = KernelPCA(n_components = 519, kernel = 'rbf')
#train_X = kpca.fit_transform(train_X)
#test_X = kpca.transform(test_X)
train_y_reshape.shape
# fit a model - GBoost
from sklearn.ensemble import GradientBoostingRegressor
#regressor_gb = GradientBoostingRegressor()
#train_y_reshape1 =np.ravel(train_y_reshape)
#regressor_gb.fit(train_X,train_y_reshape1)
# model tuning
from sklearn.grid_search import GridSearchCV

#param_grid = {'n_estimators' : np.arange(100,500,100),
#             'learning_rate' : np.arange(0.1,0.5,0.1)}
             #'max_depth': np.arange(3,6,1)}
#grid = GridSearchCV(regressor_gb, param_grid, cv=5)

#grid.fit(train_X, train_y)
#grid.best_params_
# make predictions - GBoost
#preds = regressor_gb.predict(test_X)
#y_pred = np.abs(sc_y.inverse_transform(preds))
#from sklearn.metrics import mean_absolute_error
#print("mean_absolute_error : " + str(mean_absolute_error(y_pred, test_y_reshape)))
# fit a model - XGBoost
from xgboost import XGBRegressor
#regressor_xgb = XGBRegressor()
#train_y_reshape1 =np.ravel(train_y_reshape)
#regressor_xgb.fit(train_X,train_y_reshape1)
# model tuning
from sklearn.grid_search import GridSearchCV

#param_grid = {'n_estimators' : np.arange(100,500,100)}
             #'learning_rate' : np.arange(0.1,0.5,0.1)}
#             'max_depth': np.arange(3,6,1)}
#grid = GridSearchCV(regressor_xgb, param_grid)

#grid.fit(train_X, train_y_reshape1)
#grid.best_params_
# make predictions - XGBoost
from sklearn.metrics import mean_absolute_error
#preds = regressor_xgb.predict(test_X)
#y_pred = np.abs(sc_y.inverse_transform(preds))
#print("mean_absolute_error : " + str(mean_absolute_error(y_pred, test_y_reshape)))
# fit a devision tree
from sklearn.tree import DecisionTreeRegressor
#regressor_dt = DecisionTreeRegressor(random_state = 0)
#regressor_dt.fit(train_X, train_y)
# make predictions - decision tree
from sklearn.metrics import mean_absolute_error
#preds = regressor_dt.predict(test_X)
#print("mean_absolute_error : " + str(mean_absolute_error(preds, test_y)))
# fit a Random Forest
from sklearn.ensemble import RandomForestRegressor
#regressor_rf = RandomForestRegressor(80)
#train_y_reshape1 =np.ravel(train_y_reshape)
#regressor_rf.fit(train_X,train_y_reshape1)
# model tuning
#from dask_searchcv import GridSearchCV, RandomizedSearchCV
from sklearn.grid_search import GridSearchCV

#param_grid = {'n_estimators' : np.arange(10,100,10)}
#grid = GridSearchCV(regressor_rf, param_grid, cv=10)
#grid = GridSearchCV(regressor_rf, param_grid)
#dk_grid_search = GridSearchCV(regressor_rf, param_grid=param_grid, n_jobs=-1)

#grid.fit(train_X, train_y)
#dk_grid_search.fit(train_X, train_y)
#grid.best_params_
#dk_grid_search.best_params_
# make predictions - Random Forets
from sklearn.metrics import mean_absolute_error
#preds = regressor_rf.predict(test_X)
#y_pred = np.abs(sc_y.inverse_transform(preds))
#print("mean_absolute_error : " + str(mean_absolute_error(y_pred, test_y_reshape)))
# Fit an SVR model
from sklearn.svm import SVR
regressor_svr = SVR(kernel = 'rbf')
train_y_reshape1 =np.ravel(train_y_reshape)
regressor_svr.fit(train_X, train_y_reshape1)
# make predictions - SVR
from sklearn.metrics import mean_absolute_error
preds = regressor_svr.predict(test_X)
y_pred = np.abs(sc_y.inverse_transform(preds))

test_y_reshape1 =np.ravel(test_y_reshape)
print("mean_absolute_error : " + str(mean_absolute_error(y_pred, test_y_reshape)))
# import the test dataset
test = pd.read_csv("../input/test.csv")
test1 = test.drop('ID',axis=1)
# # Feature Scaling
# standardize the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
test2 = sc.fit_transform(test1)
#test2 = sc.transform(test1)
# Normalize the data
from sklearn.preprocessing import Normalizer
#sc = Normalizer()
#test2 = sc.fit_transform(test1)
#test_X = sc.transform(test_X)
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 1197)
#train_X = pca.fit_transform(train_X)
test2 = pca.fit_transform(test2)
#explained_variance = pca.explained_variance_ratio_
#from dask.distributed import Client
#client = Client()  # start a local Dask client

#import dask_ml.joblib
#from sklearn.externals.joblib import parallel_backend
#with parallel_backend('dask'):
# Applying Kernel PCA
#    from sklearn.decomposition import KernelPCA
#    kpca = KernelPCA(n_components = 519, kernel = 'rbf')
#train_X = kpca.fit_transform(train_X)
#    test2 = kpca.fit_transform(test1)
# make predictions - GBoost
#preds = regressor_gb.predict(test1)
# Prepare submission file - GBoost
#gb2 = pd.DataFrame({'ID':test.ID,'target':preds})
# prepare the csv file
#gb2.to_csv('gb2.csv',index=False)
# make predictions - XGBoost
#preds = regressor_xgb.predict(test2)
# Prepare submission file - XGBoost
#gb8 = pd.DataFrame({'ID':test.ID,'target':preds})
# prepare the csv file
#gb8.to_csv('gb8.csv',index=False)
# make predictions - Random Forest
#preds = regressor_rf.predict(test2)
#y_pred = np.abs(sc_y.inverse_transform(preds))
# Prepare submission file - Random Forest
#rf6 = pd.DataFrame({'ID':test.ID,'target':y_pred})
# prepare the csv file
#rf6.to_csv('rf6.csv',index=False)
# make predictions - SVR
preds = regressor_svr.predict(test2)
y_pred = np.abs(sc_y.inverse_transform(preds))
# Prepare submission file - SVR
svr4 = pd.DataFrame({'ID':test.ID,'target':y_pred})
# prepare the csv file
svr4.to_csv('svr4.csv',index=False)
