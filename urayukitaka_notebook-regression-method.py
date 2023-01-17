# Basic Libraries

import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Visualization

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

import seaborn as sns



# Statistics

from scipy import stats



# Data preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



# StratifiedKFold

from sklearn.model_selection import StratifiedKFold



# Grid search

from sklearn.model_selection import GridSearchCV



# Validataion

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
def plot_point(y_train, y_test, y_train_pred, y_test_pred):

    fig, ax = plt.subplots(1,2,figsize=(20,10))

    # Scatter plot

    ax[0].scatter(y_train, y_train_pred, color="blue", alpha=0.3, s=5,

                  label="Train data R2 score:{}".format(r2_score(y_true=y_train, y_pred=y_train_pred).round(3)))

    ax[0].scatter(y_test, y_test_pred, color="red", alpha=0.3, s=5,

                  label="Test data R2 score:{}".format(r2_score(y_true=y_test, y_pred=y_test_pred).round(3)))

    ax[0].set_xlabel("true value")

    ax[0].set_ylabel("predicted value")

    ax[0].set_title("Scatter plot")

    ax[0].legend()

    # Residual plot

    ax[1].scatter(y_train_pred, y_train_pred - y_train, color="blue", alpha=0.3, s=5,

                  label="Train data MSE:{}".format(mean_squared_error(y_true=y_train, y_pred=y_train_pred).round(3)))

    ax[1].scatter(y_test_pred, y_test_pred - y_test, color="red", alpha=0.3, s=5,

                  label="Test data MSE:{}".format(mean_squared_error(y_true=y_test, y_pred=y_test_pred).round(3)))

    ax[1].set_xlabel("prediction")

    ax[1].set_ylabel("residual")

    ax[1].set_title("Residual plot")

    ax[1].legend()
## Data loading

df = pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv", header=0)
# data frame

df.head()
# Null values

df.isnull().sum().sum()
# Data info

df.info()
fig, ax = plt.subplots(1,2,figsize=(20,6))



sns.distplot(df["price"], ax=ax[0])

ax[0].set_xlabel("price")

ax[0].set_ylabel("frequency")



sns.distplot(np.log10(df["price"]), ax=ax[1])

ax[1].set_xlabel("price_log")

ax[1].set_ylabel("frequency")
# create log price columns

df["price_log"] = np.log(df["price"])
# target value

y = df["price_log"]
ex_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',

              'condition', 'grade','sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']
# yr_built

latest_year = df["yr_built"].max()

df["yr_built"] = latest_year - df["yr_built"]
# define function

def renov(x):

    if x["yr_renovated"] == 0:

        res = x["yr_built"]

    else:

        res = latest_year - x["yr_renovated"]

    return res



# apply function

df["yr_renovated"] = df.apply(renov, axis=1)
X = df[ex_columns]
# Sample 200

sns.pairplot(X.sample(200))
## Correlation

matrix = X.corr()

plt.figure(figsize=(10,10))

sns.heatmap(matrix, vmax=1, vmin=-1, cmap="bwr", square=True)
# data split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
# Scaling

sc = StandardScaler()

sc.fit(X_train)



X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
# Library

from sklearn.linear_model import LinearRegression

import statsmodels.formula.api as smf

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Instance

ln = LinearRegression()
# Fitting

ln.fit(X_train_std, y_train)
X_train_std
# Prediction

y_train_pred = ln.predict(X_train_std)

y_test_pred = ln.predict(X_test_std)
# Validation score

print("MSE train %.3f" % mean_squared_error(y_true=y_train, y_pred=y_train_pred))

print("MSE test %.3f" % mean_squared_error(y_true=y_test, y_pred=y_test_pred))



print("R2 score train %.3f" % r2_score(y_true=y_train, y_pred=y_train_pred))

print("R2 score test %.3f" % r2_score(y_true=y_test, y_pred=y_test_pred))
# visualization, scatter and residual

plot_point(y_train, y_test, y_train_pred, y_test_pred)
# coefficients and intercept

coef_ln_df = pd.DataFrame({"params":X.columns, "Coefficient":ln.coef_}).sort_values(by="Coefficient")

intercept = pd.DataFrame([["intercept", ln.intercept_]], columns=coef_ln_df.columns)



coef_ln_df = coef_ln_df.append(intercept)



coef_ln_df
# Create dataframe

data = pd.DataFrame(X_train_std, columns=X_train.columns)

# Note ! stats model need to intercept columns for training

data = sm.add_constant(data)

data["price"] = y_train.values
# create model

lm = smf.ols(formula="price ~ bedrooms+bathrooms+sqft_living+sqft_lot+floors+waterfront+view+condition+grade+sqft_above+sqft_basement+yr_built+yr_renovated+sqft_living15+sqft_lot15", data=data)

model = lm.fit()



model.summary()
# Distribution

resid = model.resid



fig, ax = plt.subplots(1,2,figsize=(20,6))

sns.distplot(resid, ax=ax[0])

ax[0].set_xlabel("resid")

ax[0].set_ylabel("frequency")

ax[0].set_title("Prediction resid")



# Probability  plot

stats.probplot(resid, plot=ax[1])
n = lm.exog.shape[1]

vifs = [variance_inflation_factor(lm.exog, i) for i in range(0, n)]

pd.DataFrame(vifs, index=lm.exog_names, columns=["VIF"])
# Library

from sklearn.linear_model import Ridge



# Instance

ri = Ridge()
# Training and score

params = {'alpha': [1000, 100, 10, 1, 0.1, 0.01, 0.001]}



# Fitting

cv_r = GridSearchCV(ri, params, cv=10, n_jobs =1)

cv_r.fit(X_train_std, y_train)



print("Best params:{}".format(cv_r.best_params_))
# best params

best_r = cv_r.best_estimator_



# prediction

y_train_pred_r = best_r.predict(X_train_std)

y_test_pred_r = best_r.predict(X_test_std)



print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_r)))

print("MSE val;{}".format(mean_squared_error(y_test, y_test_pred_r)))



print("R2 score train:{}".format(r2_score(y_train, y_train_pred_r)))

print("R2 score val:{}".format(r2_score(y_test, y_test_pred_r)))
# visualization, scatter and residual

plot_point(y_train, y_test, y_train_pred_r, y_test_pred_r)
# coefficients and intercept

coef_ri_df = pd.DataFrame({"params":X.columns, "Coefficient":best_r.coef_}).sort_values(by="Coefficient")

intercept = pd.DataFrame([["intercept", best_r.intercept_]], columns=coef_ri_df.columns)



coef_ri_df = coef_ri_df.append(intercept)



coef_ri_df
# Resid

resid = y_train_pred_r - y_train



# Distribution

fig, ax = plt.subplots(1,2,figsize=(20,6))

sns.distplot(resid, ax=ax[0])

ax[0].set_xlabel("resid")

ax[0].set_ylabel("frequency")

ax[0].set_title("Prediction resid")



# Probability  plot

stats.probplot(resid, plot=ax[1])
# Library

import statsmodels.formula.api as smf

import statsmodels.api as sm



# create model

model = smf.ols(formula="price ~ bedrooms+bathrooms+sqft_living+sqft_lot+floors+waterfront+view+condition+grade+sqft_above+sqft_basement+yr_built+yr_renovated+sqft_living15+sqft_lot15", data=data)

ridge = model.fit_regularized(L1_wt=0)

# L1_wt = 0:Ridge, 1:Lasso, 0.5:Elastic Net
# params, cannot sumary table

ridge.params
# Library

from sklearn.linear_model import Lasso



# Instance

la = Lasso()
# Training and score

params = {'alpha': [10, 1, 0.1, 0.01, 0.001, 0.0001]}



# Fitting

cv_l = GridSearchCV(la, params, cv=10, n_jobs =1)

cv_l.fit(X_train_std, y_train)



print("Best params:{}".format(cv_l.best_params_))
# best params

best_l = cv_l.best_estimator_



# prediction

y_train_pred_l = best_l.predict(X_train_std)

y_test_pred_l = best_l.predict(X_test_std)



print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_l)))

print("MSE val;{}".format(mean_squared_error(y_test, y_test_pred_l)))



print("R2 score train:{}".format(r2_score(y_train, y_train_pred_l)))

print("R2 score val:{}".format(r2_score(y_test, y_test_pred_l)))
# visualization, scatter and residual

plot_point(y_train, y_test, y_train_pred_l, y_test_pred_l)
# coefficients and intercept

coef_la_df = pd.DataFrame({"params":X.columns, "Coefficient":best_l.coef_}).sort_values(by="Coefficient")

intercept = pd.DataFrame([["intercept", best_l.intercept_]], columns=coef_la_df.columns)



coef_la_df = coef_la_df.append(intercept)



coef_la_df
# Resid

resid = y_train_pred_l - y_train



# Distribution

fig, ax = plt.subplots(1,2,figsize=(20,6))

sns.distplot(resid, ax=ax[0])

ax[0].set_xlabel("resid")

ax[0].set_ylabel("frequency")

ax[0].set_title("Prediction resid")



# Probability  plot

stats.probplot(resid, plot=ax[1])
# Library

import statsmodels.formula.api as smf

import statsmodels.api as sm



# create model

model = smf.ols(formula="price ~ bedrooms+bathrooms+sqft_living+sqft_lot+floors+waterfront+view+condition+grade+sqft_above+sqft_basement+yr_built+yr_renovated+sqft_living15+sqft_lot15", data=data)

lasso = model.fit_regularized(L1_wt=1)

# L1_wt = 0:Ridge, 1:Lasso, 0.5:Elastic Net
# params, cannot sumary table

lasso.params
# Library

from sklearn.linear_model import ElasticNet



# Instance

el = ElasticNet()
# Training and score

params = {'alpha': [10, 1, 0.1, 0.01, 0.001, 0.0001]}



# Fitting

cv_e = GridSearchCV(el, params, cv=10, n_jobs =1)

cv_e.fit(X_train_std, y_train)



print("Best params:{}".format(cv_l.best_params_))
# best params

best_e = cv_e.best_estimator_



# prediction

y_train_pred_e = best_e.predict(X_train_std)

y_test_pred_e = best_e.predict(X_test_std)



print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_e)))

print("MSE val;{}".format(mean_squared_error(y_test, y_test_pred_e)))



print("R2 score train:{}".format(r2_score(y_train, y_train_pred_e)))

print("R2 score val:{}".format(r2_score(y_test, y_test_pred_e)))
# visualization, scatter and residual

plot_point(y_train, y_test, y_train_pred_e, y_test_pred_e)
# coefficients and intercept

coef_el_df = pd.DataFrame({"params":X.columns, "Coefficient":best_e.coef_}).sort_values(by="Coefficient")

intercept = pd.DataFrame([["intercept", best_e.intercept_]], columns=coef_el_df.columns)



coef_el_df = coef_el_df.append(intercept)



coef_el_df
# Resid

resid = y_train_pred_e - y_train



# Distribution

fig, ax = plt.subplots(1,2,figsize=(20,6))

sns.distplot(resid, ax=ax[0])

ax[0].set_xlabel("resid")

ax[0].set_ylabel("frequency")

ax[0].set_title("Prediction resid")



# Probability  plot

stats.probplot(resid, plot=ax[1])
# Library

import statsmodels.formula.api as smf

import statsmodels.api as sm



# create model

model = smf.ols(formula="price ~ bedrooms+bathrooms+sqft_living+sqft_lot+floors+waterfront+view+condition+grade+sqft_above+sqft_basement+yr_built+yr_renovated+sqft_living15+sqft_lot15", data=data)

elast = model.fit_regularized(L1_wt=0.5)

# L1_wt = 0:Ridge, 1:Lasso, 0.5:Elastic Net
# params, cannot sumary table

elast.params
# Library

from sklearn.tree import DecisionTreeRegressor



# Instance

tree = DecisionTreeRegressor(random_state=10)
# Training and score

param_range = [5, 10,15]

leaf = [50,55,60,65,70]

param_grid = {"max_depth":param_range, "max_leaf_nodes":leaf}



# Fitting

cv_tree = GridSearchCV(tree, param_grid, cv=10, n_jobs =1)

cv_tree.fit(X_train, y_train)



print("Best params:{}".format(cv_tree.best_params_))
# best params

best_tr = cv_tree.best_estimator_



# prediction

y_train_pred_tr = best_tr.predict(X_train)

y_test_pred_tr = best_tr.predict(X_test)



print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_tr)))

print("MSE val;{}".format(mean_squared_error(y_test, y_test_pred_tr)))



print("R2 score train:{}".format(r2_score(y_train, y_train_pred_tr)))

print("R2 score val:{}".format(r2_score(y_test, y_test_pred_tr)))
# visualization, scatter and residual

plot_point(y_train, y_test, y_train_pred_tr, y_test_pred_tr)
# Resid

resid = y_train_pred_tr - y_train



# Distribution

fig, ax = plt.subplots(1,2,figsize=(20,6))

sns.distplot(resid, ax=ax[0])

ax[0].set_xlabel("resid")

ax[0].set_ylabel("frequency")

ax[0].set_title("Prediction resid")



# Probability  plot

stats.probplot(resid, plot=ax[1])
#! pip install dtreeviz

#!brew install poppler

#!brew install pdf2svg

#!brew install graphviz --with-librsvg --with-app --with-pango
# Library

from sklearn import tree

#from dtreeviz.trees import *

#import graphviz
# Fitting

tree_c = tree.DecisionTreeRegressor(max_depth=10, max_leaf_nodes=60)

tree_c.fit(X_train, y_train)
# Visualization

# Omitted because image is heavy

# viz = dtreeviz(tree_c, X_train, y_train, target_name="price_flg", feature_names=list(X_train.columns), class_names=list(y_train))

# viz
# Library

from sklearn.ensemble import RandomForestRegressor



# Instance

forest = RandomForestRegressor(random_state=10)
# Training and score

param_range = [10,15,20,25]

leaf = [55, 60,65,70]

param_grid = {"n_estimators":param_range, "max_depth":param_range, "max_leaf_nodes":leaf}



# Fitting

cv_forest = GridSearchCV(forest, param_grid, cv=10, n_jobs =1)

cv_forest.fit(X_train, y_train)



print("Best params:{}".format(cv_forest.best_params_))
# best params

best_fo = cv_forest.best_estimator_



# prediction

y_train_pred_fo = best_fo.predict(X_train)

y_test_pred_fo = best_fo.predict(X_test)



print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_fo)))

print("MSE val;{}".format(mean_squared_error(y_test, y_test_pred_fo)))



print("R2 score train:{}".format(r2_score(y_train, y_train_pred_fo)))

print("R2 score val:{}".format(r2_score(y_test, y_test_pred_fo)))
# visualization, scatter and residual

plot_point(y_train, y_test, y_train_pred_fo, y_test_pred_fo)
# Resid

resid = y_train_pred_fo - y_train



# Distribution

fig, ax = plt.subplots(1,2,figsize=(20,6))

sns.distplot(resid, ax=ax[0])

ax[0].set_xlabel("resid")

ax[0].set_ylabel("frequency")

ax[0].set_title("Prediction resid")



# Probability  plot

stats.probplot(resid, plot=ax[1])
# Library

import xgboost as xgb



# Instance

xgbr = xgb.XGBRegressor(random_state=10)
# Training and score

learning_rate = [0.05, 0.1, 0.15]

max_depth = [3, 5, 7]

subsample = [0.85, 0.9, 0.95, 1]

colsample_bytree = [0.3, 0.5, 0.8]



param_grid = {'learning_rate': learning_rate, 'max_depth': max_depth, 

          'subsample': subsample, 'colsample_bytree': colsample_bytree}



# Fitting

cv_xgb = GridSearchCV(xgbr, param_grid, cv=10, n_jobs =1)

cv_xgb.fit(X_train, y_train)



print("Best params:{}".format(cv_xgb.best_params_))
# best params

best_xg = cv_xgb.best_estimator_



# prediction

y_train_pred_xg = best_xg.predict(X_train)

y_test_pred_xg = best_xg.predict(X_test)



print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_xg)))

print("MSE val;{}".format(mean_squared_error(y_test, y_test_pred_xg)))



print("R2 score train:{}".format(r2_score(y_train, y_train_pred_xg)))

print("R2 score val:{}".format(r2_score(y_test, y_test_pred_xg)))
# visualization, scatter and residual

plot_point(y_train, y_test, y_train_pred_xg, y_test_pred_xg)
# Resid

resid = y_train_pred_xg - y_train



# Distribution

fig, ax = plt.subplots(1,2,figsize=(20,6))

sns.distplot(resid, ax=ax[0])

ax[0].set_xlabel("resid")

ax[0].set_ylabel("frequency")

ax[0].set_title("Prediction resid")



# Probability  plot

stats.probplot(resid, plot=ax[1])
# Library

import lightgbm as lgb



# Instance

lgbm = lgb.LGBMRegressor()
# Training and score

learning_rate = [0.05, 0.1, 0.15]

max_depth = [5,10,15]



param_grid = {'learning_rate': learning_rate, 'max_depth': max_depth}



# Fitting

cv_lgbm = GridSearchCV(lgbm, param_grid, cv=10, n_jobs =1)

cv_lgbm.fit(X_train, y_train)



print("Best params:{}".format(cv_lgbm.best_params_))
# best params

best_lg = cv_lgbm.best_estimator_



# prediction

y_train_pred_lg = best_lg.predict(X_train)

y_test_pred_lg = best_lg.predict(X_test)



print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_lg)))

print("MSE val;{}".format(mean_squared_error(y_test, y_test_pred_lg)))



print("R2 score train:{}".format(r2_score(y_train, y_train_pred_lg)))

print("R2 score val:{}".format(r2_score(y_test, y_test_pred_lg)))
# visualization, scatter and residual

plot_point(y_train, y_test, y_train_pred_lg, y_test_pred_lg)
# Resid

resid = y_train_pred_lg - y_train



# Distribution

fig, ax = plt.subplots(1,2,figsize=(20,6))

sns.distplot(resid, ax=ax[0])

ax[0].set_xlabel("resid")

ax[0].set_ylabel("frequency")

ax[0].set_title("Prediction resid")



# Probability  plot

stats.probplot(resid, plot=ax[1])
# Library

from sklearn.ensemble import GradientBoostingRegressor



# Instance

gbr = GradientBoostingRegressor(random_state=10)
# Training and score

learning_rate = [0.05, 0.1, 0.15]

max_depth = [3, 5, 7]



param_grid = {'learning_rate': learning_rate, 'max_depth': max_depth}



# Fitting

cv_gbr = GridSearchCV(gbr, param_grid, cv=10, n_jobs =1)

cv_gbr.fit(X_train, y_train)



print("Best params:{}".format(cv_gbr.best_params_))
# best params

best_gbr = cv_gbr.best_estimator_



# prediction

y_train_pred_gbr = best_gbr.predict(X_train)

y_test_pred_gbr = best_gbr.predict(X_test)



print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_gbr)))

print("MSE val;{}".format(mean_squared_error(y_test, y_test_pred_gbr)))



print("R2 score train:{}".format(r2_score(y_train, y_train_pred_gbr)))

print("R2 score val:{}".format(r2_score(y_test, y_test_pred_gbr)))
# visualization, scatter and residual

plot_point(y_train, y_test, y_train_pred_gbr, y_test_pred_gbr)
# Resid

resid = y_train_pred_gbr - y_train



# Distribution

fig, ax = plt.subplots(1,2,figsize=(20,6))

sns.distplot(resid, ax=ax[0])

ax[0].set_xlabel("resid")

ax[0].set_ylabel("frequency")

ax[0].set_title("Prediction resid")



# Probability  plot

stats.probplot(resid, plot=ax[1])
name = ["Linear", "Ridge", "Lasso", "E-Net", "D-tree", "R-forest", "XGB", "LGBM", "GBR"]



mse = mean_squared_error(y_train, y_train_pred)

mse_r = mean_squared_error(y_train, y_train_pred_r)

mse_l = mean_squared_error(y_train, y_train_pred_l)

mse_e = mean_squared_error(y_train, y_train_pred_e)

mse_tr = mean_squared_error(y_train, y_train_pred_tr)

mse_fo = mean_squared_error(y_train, y_train_pred_fo)

mse_xg = mean_squared_error(y_train, y_train_pred_xg)

mse_lg = mean_squared_error(y_train, y_train_pred_lg)

mse_gbr = mean_squared_error(y_train, y_train_pred_gbr)



r2 = r2_score(y_train, y_train_pred)

r2_r = r2_score(y_train, y_train_pred_r)

r2_l = r2_score(y_train, y_train_pred_l)

r2_e = r2_score(y_train, y_train_pred_e)

r2_tr = r2_score(y_train, y_train_pred_tr)

r2_fo = r2_score(y_train, y_train_pred_fo)

r2_xg = r2_score(y_train, y_train_pred_xg)

r2_lg = r2_score(y_train, y_train_pred_lg)

r2_gbr = r2_score(y_train, y_train_pred_gbr)



mse = [mse, mse_r, mse_l, mse_e, mse_tr, mse_fo, mse_xg, mse_lg, mse_gbr]

r2 = [r2, r2_r, r2_l, r2_e, r2_tr, r2_fo, r2_xg, r2_lg, r2_gbr]



# bar plot

fig, ax = plt.subplots(1,2, figsize=(20,6))



ax[0].bar(name, mse, color="blue")

ax[0].set_ylabel("mese")

ax[0].set_xlabel("method")

ax[0].set_title("MSE")



ax[1].bar(name, r2, color="red")

ax[1].set_ylabel("R2 score")

ax[1].set_xlabel("method")

ax[1].set_title("R2 score")