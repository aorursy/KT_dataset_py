import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.regression import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.ensemble import RandomForestRegressor
import matplotlib as plt
from sklearn.linear_model import SGDRegressor
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from catboost import Pool, cv
data = pd.read_csv('train.csv').set_index('Id')
test = pd.read_csv('test.csv').set_index('Id')
y = np.log(data['SalePrice'])
data = data.drop('SalePrice', axis=1)
frames = [data, test]
all_data = pd.concat(frames)
all_data.info()
all_data = all_data.drop(['Alley', 'FireplaceQu', 'PoolQC' , 'Fence', 'MiscFeature'], axis=1)
all_data.columns[all_data.isna().sum() > 0]
all_data['LotFrontage'] = all_data['LotFrontage'].fillna(all_data['LotFrontage'].mean())
all_data['Utilities'] = all_data['Utilities'].fillna('AllPub')
all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
all_data['BsmtQual'] = all_data['BsmtQual'].fillna('NA')
all_data['BsmtCond'] = all_data['BsmtCond'].fillna('NA')
all_data['BsmtExposure'] = all_data['BsmtExposure'].fillna('NA')
all_data['BsmtFinType1'] = all_data['BsmtFinType1'].fillna('NA')
all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].fillna(0)
all_data['BsmtFinType2'] = all_data['BsmtFinType2'].fillna('NA')
all_data['BsmtFinSF2'] = all_data['BsmtFinSF2'].fillna(0)
all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].fillna(0)
all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(0)
all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna(0)
all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].fillna(0)
all_data['Exterior1st'] = all_data['Exterior1st'].fillna('VinylSd')
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna('AsbShng')
all_data['KitchenQual'] = all_data['KitchenQual'].fillna('TA')
all_data['Functional'] = all_data['Functional'].fillna('Typ')
all_data['GarageType'] = all_data['GarageType'].fillna('NA')
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(all_data['GarageYrBlt'].mean()).astype('int')
all_data['GarageFinish'] = all_data['GarageFinish'].fillna('NA')
all_data['GarageCars'] = all_data['GarageCars'].fillna(0)
all_data['GarageArea'] = all_data['GarageArea'].fillna(0)
all_data['GarageQual'] = all_data['GarageQual'].fillna('NA')
all_data['GarageCond'] = all_data['GarageCond'].fillna('NA')
all_data['SaleType'] = all_data['SaleType'].fillna('Oth')
all_data['MSSubClass'] = all_data['MSSubClass'].astype('str')
all_data['MSZoning'] = all_data['MSZoning'].fillna('RL')
sns.scatterplot(x=all_data['GrLivArea'], y=np.exp(y));
sns.scatterplot(x=all_data['OverallQual'], y=np.exp(y));
cat_data = all_data
all_data = pd.get_dummies(all_data)
all_data.shape
#all_data['Age_House']= (all_data['YrSold']-all_data['YearBuilt'])
scaler = StandardScaler()
cols = list(all_data.columns[all_data.dtypes != object])
all_data[cols] = scaler.fit_transform(all_data[cols])
split = data.shape[0]
X = all_data[:split]
X_test = all_data[split:]
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3)
X_train.shape, X_holdout.shape, X_test.shape
alphas = np.logspace(-4, 4, 200)
lasso_cv = LassoCV(cv=10, alphas=alphas)
lasso_cv.fit(X_train, y_train)
lasso_cv.alpha_
lasso = Lasso(alpha=lasso_cv.alpha_)
lasso.fit(X_train, y_train)
lasso_coef = pd.DataFrame(data={'coef' : lasso.coef_, 'abs_coef' : np.abs(lasso.coef_)}, 
                          index=X_train.columns).sort_values(by='abs_coef', ascending=False)
lasso_coef.head(10)
print("Mean squared error (train): %.3f" % mean_squared_error(y_train, lasso.predict(X_train)))
print("Mean squared error (holdout): %.5f" % mean_squared_error(y_holdout, lasso.predict(X_holdout)))
pd.DataFrame(np.exp(lasso.predict(X_test)), index=X_test.index,columns=['SalePrice']).to_csv('lasso_submission.csv')
alphas = np.logspace(-4, 4, 10)
ridge_cv = RidgeCV(cv=10, alphas=alphas)
ridge_cv.fit(X_train, y_train)
ridge_cv.alpha_
ridge = Ridge(alpha=ridge_cv.alpha_)
ridge.fit(X_train, y_train)
ridge_coef = pd.DataFrame(data={'coef' : ridge.coef_, 'abs_coef' : np.abs(ridge.coef_)}, 
                          index=X_train.columns).sort_values(by='abs_coef', ascending=False)
ridge_coef.head(20)
print("Mean squared error (train): %.3f" % mean_squared_error(y_train, ridge.predict(X_train)))
print("Mean squared error (holdout): %.3f" % mean_squared_error(y_holdout, ridge.predict(X_holdout)))
pd.DataFrame(np.exp(ridge.predict(X_test)), index=X_test.index,columns=['SalePrice']).to_csv('submission.csv')
%%time
forest = RandomForestRegressor()
forest_params = {'n_estimators' : range(100, 300, 20), 'max_depth': range(10, 30, 5), 
                 'max_features': range(30, 70, 5)}

locally_best_forest = GridSearchCV(forest, param_grid=forest_params, cv=10, n_jobs=-1) 
locally_best_forest.fit(X_train, y_train)
locally_best_forest.best_params_, locally_best_forest.best_score_
locally_best_forest = RandomForestRegressor(n_estimators = 500, max_depth=20, max_features=50)
locally_best_forest.fit(X_train, y_train)
print("Mean squared error (train): %.3f" % mean_squared_error(y_train, locally_best_forest.predict(X_train))) 
print("Mean squared error (holdout): %.3f" % mean_squared_error(y_holdout, locally_best_forest.predict(X_holdout)))
forest_feat= pd.DataFrame(data={'feature' : np.log(locally_best_forest.feature_importances_)}, 
                          index=X_train.columns).sort_values(by='feature', ascending=False)
pd.DataFrame(np.exp(locally_best_forest.predict(X_test)), index=X_test.index,columns=['SalePrice']).to_csv('forest_submission.csv')
forest_feat.head(20)
sgd = SGDRegressor()
np.logspace(-4, 4, 10)
sgd_grid = {
    'alpha': np.logspace(-4, 0, 10),
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'learning_rate': ['constant', 'optimal', 'invscaling']}
%%time
locally_best_sgd = GridSearchCV(sgd, param_grid=sgd_grid, cv=10, n_jobs=-1)
locally_best_sgd.fit(X_train, y_train)
locally_best_sgd.best_params_
sgd_best = locally_best_sgd.best_estimator_
print("Mean squared error (train): %.3f" % mean_squared_error(y_train, sgd_best.predict(X_train))) 
print("Mean squared error (holdout): %.3f" % mean_squared_error(y_holdout, sgd_best.predict(X_holdout)))
pd.DataFrame(np.exp(sgd_best.predict(X_test)), index=X_test.index,columns=['SalePrice']).to_csv('submission.csv')
xgb1 = XGBRegressor(objective='reg:squarederror', nthread = 2)
xgb_params1 = {'max_depth': [3, 5, 6, 7],
              'min_child_weight': [1, 5],
              'subsample': [0.3, 0.7, 1],
              'colsample_bytree': [0.3, 0.7, 1]}
xgb_grid = RandomizedSearchCV(xgb1,
                        param_distributions = xgb_params1,
                        cv = 10,
                        scoring='neg_root_mean_squared_error')
%%time
xgb_grid.fit(X_train, y_train)
xgb_grid.best_params_
xgb2 = XGBRegressor(objective='reg:squarederror', nthread = 2, n_estimators=500,
                   colsample_bytree=xgb_grid.best_params_['colsample_bytree'],
                   max_depth=xgb_grid.best_params_['max_depth'],
                   min_child_weight=xgb_grid.best_params_['min_child_weight'],
                   subsample=xgb_grid.best_params_['subsample'])
xgb_params2 = {'learning_rate' : np.arange(0.1, 1, 0.2)}               
xgb2_grid = GridSearchCV(xgb2,
                        xgb_params2,
                        cv = 10,
                        scoring='neg_root_mean_squared_error')
%%time
xgb2_grid.fit(X_train, y_train)
xgb2_grid.best_params_
best_xgb = xgb2_grid.best_estimator_
print("Mean squared error (train): %.3f" % mean_squared_error(y_train, best_xgb.predict(X_train))) 
print("Mean squared error (holdout): %.5f" % mean_squared_error(y_holdout, best_xgb.predict(X_holdout)))
pd.DataFrame(np.exp(best_xgb.predict(X_test)), index=X_test.index,columns=['SalePrice']).to_csv('Xgbost_submission.csv')
w = np.arange(0.1, 1, 0.1)
d = {}
for i in w:
    d[i] = mean_squared_error(y_holdout, i*lasso.predict(X_holdout) + (1 - i)*best_xgb.predict(X_holdout))
w1 = max(d, key=d.get)
print(w1)
print(d[w1])
pd.DataFrame(np.exp(w1*lasso.predict(X_test) + (1 - w1)*best_xgb.predict(X_test)), 
             index=X_test.index,columns=['SalePrice']).to_csv('lin+xgb_submission.csv')
col = cat_data.columns[cat_data.dtypes == 'object']
cat_data[col] = cat_data[col].astype(str)
X_test_cat = cat_data[split:]
X_train_cat, X_holdout_cat, y_train_cat, y_holdout_cat = train_test_split(cat_data[:split], y, test_size=0.3)
cat = CatBoostRegressor(cat_features=np.where(X_train_cat.dtypes == 'object')[0].tolist())
cat.fit(X_train_cat, y_train_cat)
print("Mean squared error (train): %.3f" % mean_squared_error(y_train_cat, cat.predict(X_train_cat))) 
print("Mean squared error (holdout): %.5f" % mean_squared_error(y_holdout_cat, cat.predict(X_holdout_cat)))
