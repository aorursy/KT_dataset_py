import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import  GradientBoostingRegressor

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV



import seaborn as sns

import xgboost as xgb







from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)





import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr



from sklearn import metrics

from sklearn.metrics import mean_squared_error

import numpy as np
pd.set_option('display.max_columns', 500)
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

sample = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.describe()
mask = np.triu(np.ones_like(train.corr(), dtype=bool))

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(train.corr(), mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))
train["SalePrice"] = np.log1p(train["SalePrice"])

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) 

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)

all_data = all_data.fillna(all_data.mean())
# создан матриц sklearn

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
model_ridge = Ridge()

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in [0.05, 0.1, 0.3, 1, 3, 5,  30, 50, 75]]

cv_ridge = pd.Series(cv_ridge, index = [0.05, 0.1, 0.3, 1, 3, 5, 30, 50, 75])

print('ridge test  score ',cv_ridge.min())

print("alpha: ",cv_ridge.sort_values().index[0])
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)

print("Lasso test score " ,rmse_cv(model_lasso).mean())

coef = pd.Series(model_lasso.coef_, index = X_train.columns)

imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])

matplotlib.rcParams['figure.figsize'] = (6.0, 3.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso")
dtrain = xgb.DMatrix(X_train, label = y)

dtest = xgb.DMatrix(X_test)

params = {"max_depth":2, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model['test-rmse-mean'].mean()
model_ridge = RandomForestRegressor()

cv_rf = [rmse_cv(RandomForestRegressor(n_estimators = estimators)).mean() 

            for estimators in [1,10,20,40,100]]

cf_rf = pd.Series(cv_ridge, index = [1,10,20,40,100])

print('random forest   score ',cf_rf.min())

print("number of estimators: ",cf_rf.sort_values().index[0])
rf = RandomForestRegressor(n_estimators=100)

rf.fit(X_train,y)
param_grid = {

    'max_depth': [80, 90,],

    'max_features': [2, 3],

    'min_samples_leaf': [3, 4, 5],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [100, 200]

}

grid_search = GridSearchCV(estimator = GradientBoostingRegressor(), scoring="neg_mean_squared_error", param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train, y)
pd.DataFrame(grid_search.cv_results_)['mean_test_score'].mean()*(-1)
gradient_boost_preds = np.expm1(grid_search.best_estimator_.predict(X_test))

rf_preds = np.expm1(rf.predict(X_test))

lasso_preds = np.expm1(model_lasso.predict(X_test))
preds = 0.45*gradient_boost_preds +0.1*rf_preds + 0.45*lasso_preds

kaggle_solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

kaggle_solution.to_csv("test_pred.csv", index = False)