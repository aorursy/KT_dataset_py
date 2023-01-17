import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',index_col = 'Id')

prediction = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv',index_col = 'Id')
df.head()
prediction.head()
df.info()
# plot an histogram to check Sales price distribution

df.SalePrice.hist()

# compute the mean the median

print('mean of Sales price {:.1f}; median of Sales price {:.1f}'.format(np.mean(df.SalePrice),np.median(df.SalePrice)))

# skewness

print("skewness score {:.2f}".format(df.SalePrice.skew()))
missingdata_train = pd.DataFrame(pd.isnull(df).sum().sort_values()[-19:],columns = ['missing count'])

missingdata_test = pd.DataFrame(pd.isnull(prediction).sum().sort_values()[-33:],columns = ['missing count'])

Missing_data = missingdata_train.merge(missingdata_test,how = 'outer',left_index = True,right_index = True)

Missing_data.fillna(0,inplace = True)

Missing_data
has_meaning_none = ['MiscFeature','Fence','PoolQC','GarageCond','GarageQual','GarageFinish','GarageType','FireplaceQu','BsmtFinType2','BsmtFinType1','BsmtExposure',

                   'BsmtCond','BsmtQual','Alley','MasVnrType'] 



for item in has_meaning_none:

    df[item].fillna('None',inplace = True)

    prediction[item].fillna('None',inplace = True)
df.LotFrontage.fillna(df.LotFrontage.mean(),inplace = True)

prediction.LotFrontage.fillna(prediction.LotFrontage.mean(),inplace = True)
no_dependency_categorical = ['Electrical','Exterior1st','Exterior2nd','Functional','KitchenQual','MSZoning','Utilities','SaleType']



for item in no_dependency_categorical:

    df[item].fillna(df[item].mode()[0],inplace = True)

    prediction[item].fillna(prediction[item].mode()[0],inplace = True)
basement_features = ['BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','TotalBsmtSF']



for item in basement_features:

    df[item] = df.apply(lambda row: 0 if row['BsmtCond'] == 'None' else row[item], axis = 1)

    prediction[item] = prediction.apply(lambda row: 0 if row['BsmtCond'] == 'None' else row[item], axis = 1)



garage_features = ['GarageArea','GarageCars','GarageYrBlt']



for item in garage_features:

    df[item] = df.apply(lambda row: 0 if row['GarageType'] == 'None' else row[item], axis = 1)

    prediction[item] = prediction.apply(lambda row: 0 if row['GarageType'] == 'None' else row[item], axis = 1)



item = 'MasVnrArea'

df[item] = df.apply(lambda row: 0 if row['MasVnrType'] == 'None' else row[item], axis = 1)

prediction[item] = prediction.apply(lambda row: 0 if row['MasVnrType'] == 'None' else row[item], axis = 1)
df.isnull().sum().sum(),prediction.isnull().sum().sum()
corr_matrix = df.corr()

corr_matrix.SalePrice.abs().sort_values(ascending = False)[1:6]
plt.plot(df.GrLivArea,df.SalePrice,'o')

plt.xlabel('Ground Living Area, sqft')

plt.ylabel('Sale Price')
sns.boxplot(df.OverallQual,df.SalePrice)
pair_set = corr_matrix.SalePrice.abs().sort_values(ascending = False)[:6].index.tolist()

sns.pairplot(df[pair_set])
df.SalePrice = np.log(df.SalePrice)

df.SalePrice.hist()

# compute the mean the median

print('mean of Sales price {:.1f}; median of Sales price {:.1f}'.format(np.mean(df.SalePrice),np.median(df.SalePrice)))

# skewness

print("skewness score {:.2f}".format(df.SalePrice.skew()))
df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea'] 

prediction['TotalSF'] = prediction['TotalBsmtSF'] + prediction['GrLivArea'] 
list_numeric_categorical = ['MSSubClass','OverallQual','OverallCond','MoSold','YrSold']



for item in list_numeric_categorical:

    df[item] = df[item].astype('category')

    prediction[item] = prediction[item].astype('category')
categorical_col = df.dtypes[df.dtypes == 'object'].index.tolist()



for item in categorical_col:

    new_dict = { _:key for key,_ in enumerate(df[item].unique())}

    df[item].replace(new_dict,inplace = True)

    prediction[item].replace(new_dict,inplace = True)

    df[item] = df[item].astype('category')

    prediction[item] = prediction[item].astype('category')
y = df.SalePrice



df_index = df.index

pred_index = prediction.index

df_drop = df.drop(['SalePrice'],axis = 1)

dummy = pd.concat([df_drop,prediction])

dummy = pd.get_dummies(dummy)

data_train = dummy.loc[df_index]

data_pred = dummy.loc[pred_index]
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score, KFold

from sklearn.model_selection import train_test_split 

def RMSE (pred,act):

    return np.sqrt(mean_squared_error(act, pred))
X_train, X_test, y_train, y_test = train_test_split(data_train, y, test_size=0.2, random_state=0) # split train, validation set
linreg = LinearRegression()

linreg.fit(X_train,y_train)

y_pred = linreg.predict(X_test)



lin_r2 = r2_score(y_test,y_pred)

lin_RMSE = RMSE(y_pred,y_test)

print('The R2 score is {:.3f} and RMSE is {:.3f}'.format(lin_r2,lin_RMSE))
'''

rf = RandomForestRegressor()

paremeters_rf = {"n_estimators" : [10,100,200], "min_samples_split" : [2, 3], 

                 "max_features" : ["auto", "log2",'sqrt'],'max_depth' : [2,3,10],"random_state" : [1]}

grid_rf = GridSearchCV(rf, paremeters_rf, verbose=1, scoring="r2",cv = 5)

grid_rf.fit(X_train, y_train)'''



rf_best = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,

                      max_features='auto', max_leaf_nodes=None,

                      min_impurity_decrease=0.0, min_impurity_split=None,

                      min_samples_leaf=1, min_samples_split=3,

                      min_weight_fraction_leaf=0.0, n_estimators=100,

                      n_jobs=None, oob_score=False, random_state=1, verbose=0,

                      warm_start=False)

rf_best.fit(X_train, y_train)
y_pred = rf_best.predict(X_test)

rf_r2 = r2_score(y_test,y_pred)

rf_RMSE = RMSE(y_pred,y_test)

print('The R2 score is {:.3f} and RMSE is {:.3f}'.format(rf_r2,rf_RMSE))
'''GBR = GradientBoostingRegressor()

paremeters_gbr = {'loss' : ['ls','huber'], "n_estimators" : [50,100,200], "max_leaf_nodes" : [2,3,4,5,None], 

                 'learning_rate':[0.01 , 0.05 ,0.09, 0.1],'subsample':[0.5,1]}

grid_GBR = GridSearchCV(GBR, paremeters_gbr, verbose=1, scoring="r2",cv = 5)

grid_GBR.fit(X_train, y_train)'''



GBR_best = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,

                          learning_rate=0.1, loss='huber', max_depth=3,

                          max_features=None, max_leaf_nodes=4,

                          min_impurity_decrease=0.0, min_impurity_split=None,

                          min_samples_leaf=1, min_samples_split=2,

                          min_weight_fraction_leaf=0.0, n_estimators=200,

                          n_iter_no_change=None, presort='auto',

                          random_state=None, subsample=0.5, tol=0.0001,

                          validation_fraction=0.1, verbose=0, warm_start=False)

GBR_best.fit(X_train, y_train)
y_pred = GBR_best.predict(X_test)

GBR_r2 = r2_score(y_test,y_pred)

GBR_RMSE = RMSE(y_pred,y_test)

print('The R2 score is {:.3f} and RMSE is {:.3f}'.format(GBR_r2,GBR_RMSE))
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



'''xGBR = XGBRegressor(objective='reg:squarederror')

parameters_xgbr = { "n_estimators" : [100,200,300], "max_depth" : [3,4], 

                 'learning_rate':[0.07, 0.1,0.12],'subsample':[0.6,0.7],'colsample_bytree' : [0.5,0.6,0.65]

                  }

grid_xGBR = GridSearchCV(estimator = xGBR, param_grid = parameters_xgbr, verbose=1,scoring='r2',cv = 4)

grid_xGBR.fit(X_train, y_train)

'''

XGB_best = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.6, gamma=0,

             importance_type='gain', learning_rate=0.1, max_delta_step=0,

             max_depth=3, min_child_weight=1, missing=None, n_estimators=300,

             n_jobs=1, nthread=None, objective='reg:squarederror',

             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,

             seed=None, silent=None, subsample=0.7, verbosity=1)



XGB_best.fit(X_train, y_train)
y_pred = XGB_best.predict(X_test)

xGBR_r2 = r2_score(y_test,y_pred)

xGBR_RMSE = RMSE(y_pred,y_test)

print('The R2 score is {:.3f} and RMSE is {:.3f}'.format(xGBR_r2,xGBR_RMSE))
R2s = [lin_r2,rf_r2,GBR_r2,xGBR_r2]

RMSEs = [lin_RMSE,rf_RMSE,GBR_RMSE,xGBR_RMSE]

comparison = pd.DataFrame([R2s,RMSEs], index = ['R square','RMSE' ], columns = ['Linear Regression', ' Random Forest','Gradient Boosting','XGBoost'] )
comparison