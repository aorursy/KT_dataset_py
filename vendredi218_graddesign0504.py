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
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split

import xgboost as xgb
import lightgbm as lgb
all_vio3 = pd.read_csv('../input/all-vio4-56/all_vio3_56.csv')
all_vio3.head(5)
all_vio3 = pd.get_dummies(all_vio3,columns=['weekday'])

def rush(a):
    if a in (7,8,17,18):
        return 1
    else:
        return 0

all_vio3['if_rush'] = all_vio3.hourperiod.apply(lambda x:rush(x))
all_vio3 = pd.get_dummies(all_vio3,columns=['hourperiod'])
all_vio3.head()
all_vio3['tot_volume'].fillna(all_vio3.groupby('MidTZoneId')['tot_volume'].transform('mean'),inplace=True)
all_vio3['speed'].fillna(all_vio3.groupby('MidTZoneId')['tot_volume'].transform('mean'),inplace=True)

all_vio3['tot_volume'].fillna(all_vio3['tot_volume'].mean(),inplace=True)
all_vio3['speed'].fillna(all_vio3['speed'].mean(),inplace=True)
#Correlation map to see how features are correlated with viocnt
corrmat = all_vio3.drop(['MidTZoneId','date'],axis=1).corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"viocnt":all_vio3['3vio_cnt'], "log(viocnt + 1)":np.log1p(all_vio3['3vio_cnt'])})
prices.hist()
#log transform the target:
all_vio3['3vio_cnt'] = np.log1p(all_vio3['3vio_cnt'])

#log transform skewed numeric features:
numeric_feats = all_vio3.dtypes[all_vio3.dtypes != "object"].index

skewed_feats = all_vio3[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_vio3[skewed_feats] = np.log1p(all_vio3[skewed_feats])
X_train = all_vio3[:23878].drop('3vio_cnt',axis=1)
X_train = X_train.drop(['MidTZoneId','date'],axis=1)
X_test = all_vio3[23878:].drop('3vio_cnt',axis=1)
X_test = X_test.drop(['MidTZoneId','date'],axis=1)
y = all_vio3[:23878]['3vio_cnt']
y_test = all_vio3[23878:]['3vio_cnt']
all_vio3
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

alphas = [0.001, 0.0005,0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    mae = -cross_val_score(model, X_train, y, scoring="neg_mean_absolute_error", cv = 5)
#     msle = -cross_val_score(model, X_train, y, scoring="neg_mean_squared_log_error", cv = 5)
    r2 = cross_val_score(model, X_train, y, scoring="r2", cv = 5)
    print('rmse:',rmse.mean())
    print('mae:',mae.mean())
    print('r2:',r2.mean())
    return(rmse.mean(),mae.mean(),r2.mean())

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def metrics(y, y_pred):
    mape = np.mean(np.abs((y-y_pred)/y))*100
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y,y_pred)
    r2 = r2_score(y,y_pred)
    return(rmse,mae,r2,mape)
from sklearn.linear_model import Ridge, RidgeCV

model_ridge = RidgeCV(alphas = alphas).fit(X_train,y)

ridge_rmse, ridge_mae, ridge_r2 = rmse_cv(model_ridge)
print('alpha:', model_ridge.alpha_)
model_ridge.fit(X_train,y)
ridge_pred = model_ridge.predict(X_test)
print('rmse','mae','r2')
print(metrics(y_test,ridge_pred))
model_lasso = LassoCV(alphas = alphas).fit(X_train, y)

lasso_rmse, lasso_mae, lasso_r2 = rmse_cv(model_lasso)
print('alpha:', model_lasso.alpha_)
from sklearn.linear_model import ElasticNetCV
model_enet = ElasticNetCV().fit(X_train, y)

lasso_rmse, lasso_mae, lasso_r2 = rmse_cv(model_lasso)
print('alpha:', model_enet.alpha_)

from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

model_gb = GradientBoostingRegressor(random_state=10)
model_gb.fit(X_train,y)

# _,_,_=rmse_cv(model_gb)

def metrics(y, y_pred):
    mape = np.mean(np.abs((y-y_pred)/y))*100
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y,y_pred)
    r2 = r2_score(y,y_pred)
    return(rmse,mae,r2,mape)

gb_pred = model_gb.predict(X_test)
print('rmse','mae','r2')
print(metrics(y_test,gb_pred))
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
def rmse(y_true,y_pred):
    rmse= np.sqrt(mean_squared_error(y_true,y_pred))
    print("rmse:",rmse)
    return rmse
my_rmse = make_scorer(rmse,greater_is_better=False)
# param_test1 = {'n_estimators':np.arange(70,200,15)}

# model_gb = GradientBoostingRegressor(learning_rate=0.1,max_depth=8)
# gb_cv = GridSearchCV(estimator=model_gb,param_grid=param_test1,cv=5,scoring=my_rmse)
# gb_cv.fit(X_train,y)

# print(gb_cv.best_params_)
# print(gb_cv.best_estimator_)
# print(gb_cv.best_score_)

# GBoost = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
#              learning_rate=0.1, loss='ls', max_depth=8, max_features=None,
#              max_leaf_nodes=None, min_impurity_decrease=0.0,
#              min_impurity_split=None, min_samples_leaf=1,
#              min_samples_split=2, min_weight_fraction_leaf=0.0,
#              n_estimators=85, n_iter_no_change=None, presort='auto',
#              random_state=None, subsample=1.0, tol=0.0001,
#              validation_fraction=0.1, verbose=0, warm_start=False)

# gboost_rmse, gboost_mae, gboost_r2 = rmse_cv(GBoost)

# grid_learn = [ .01,0.05, .05, .1]
# param_test1 = {'n_estimators':np.arange(80,1000,50), 'learning_rate':grid_learn,'max_depth':[ 4, 6, 8, 10]}

# model_xgb = xgb.XGBRegressor()
# xgb_cv = GridSearchCV(estimator=model_xgb,param_grid=param_test1,cv=5,scoring=my_rmse)
# xgb_cv.fit(X_train,y)

# print(gb_cv.best_params_)
# print(gb_cv.best_estimator_)
# print(gb_cv.best_score_)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.01, max_depth=8, 
                             min_child_weight=2, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

# xgboost_rmse, xgboost_mae, xgboost_r2 = rmse_cv(XGBoost)
model_xgb.fit(X_train,y)
xgb_pred = model_xgb.predict(X_test)
print('rmse','mae','r2')
print(metrics(y_test,xgb_pred))
pd.set_option('display.max_rows', 100)
ft_weights = pd.DataFrame(model_xgb.feature_importances_, columns=['weights'], index=X_train.columns.values).sort_values(by='weights',
                                                                                                                       ascending=False)
ft_weights
xgb.plot_importance(model_xgb)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

# lgbm_rmse, lgbm_mae, lgbm_r2 = rmse_cv(model_lgb)

model_lgb.fit(X_train,y)
lgb_pred = model_lgb.predict(X_test)
print('rmse','mae','r2')
print(metrics(y_test,lgb_pred))
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   
averaged_models = AveragingModels(models = (model_lasso,model_ridge,model_enet,model_gb))

avg_rmse, avg_mae, avg_r2 = rmse_cv(averaged_models)
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
# stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, ridge),
#                                                  meta_model = model_lasso)

# stk_rmse, stk_mae, stk_r2 = rmse_cv(stacked_averaged_models)
# print('rmse:',stk_rmse)
# print('mae:',stk_mae)
# # print('msle:',ridge_msle)
# print('r2:',stk_r2)
averaged_models.fit(X_train,y)
avg_pred = averaged_models.predict(X_test)
print('rmse','mae','r2','mape')
print(metrics(y_test,avg_pred))
model_lgb.fit(X_train,y)
lgb_pred = model_lgb.predict(X_test)
print('rmse','mae','r2')
print(metrics(y_test,lgb_pred))
output = avg_pred*0.1+xgb_pred*0.5+lgb_pred*0.4

print('rmse','mae','r2')
print(metrics(y_test,output))
origin = pd.read_csv('../input/all-vio4-56/all_vio3_56.csv')[23878:][['MidTZoneId','date','weekday','hourperiod','3vio_cnt']]
origin['pred'] = np.expm1(output)
# pd.DataFrame({'diff':(np.expm1(output)-np.expm1(y_test))}).to_csv("diff.csv")
origin.to_csv('pred.csv')
# pd.DataFrame(np.expm1(output),columns=["pred"])
# np.expm1(y_test)
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)

rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " 
      +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(6),coef.sort_values().tail(6)])

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y)
xgb_preds = np.expm1(model_xgb.predict(X_test))
lasso_preds = np.expm1(model_lasso.predict(X_test))

predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})
predictions.plot(x = "xgb", y = "lasso", kind = "scatter")
preds = 0.7*lasso_preds + 0.3*xgb_preds
# 'MidTZoneId','date','weekday'
solution = pd.DataFrame({"id":test_16252_6.MidTZoneId,'date':test_16252_6.date,
                         'weekday':test_16252_6.weekday, "viocnt":preds})
# solution.to_csv("ridge_sol.csv", index = False)
solution