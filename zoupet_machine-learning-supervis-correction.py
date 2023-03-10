# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings



warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pylab

import calendar



import numpy as np

import pandas as pd

import seaborn as sn

from scipy import stats



from datetime import datetime

import matplotlib.pyplot as plt

import warnings



pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline
df = pd.read_csv("/kaggle/input/ml-training-vlib/vlib.csv")
df.head(10)
df.shape
df.info()
df.describe()
df.tail()
df.shape
df.columns
df.duplicated().sum()
for i in df.columns:

    print(i," : ",df[i].isnull().sum())
list_variables = ["datetime","season","weather","holiday","workingday"]
"2011-01-01 00:00:00".split()
df["date"] = df.datetime.apply(lambda x : x.split()[0])

df["date"]
df["hour"] = df.datetime.apply(lambda x : x.split()[1])

df["hour"]
df["hour"] = df.datetime.apply(lambda x : x.split()[1].split(":")[0])

df["hour"] 
datetime.strptime("2011-01-01","%Y-%m-%d").weekday()
calendar.day_name[datetime.strptime("2011-01-01","%Y-%m-%d").weekday()]
df["weekday"] = df.date.apply(lambda x : 

                                            calendar.day_name[datetime.strptime(x,"%Y-%m-%d").weekday()])

df["month"] = df.date.apply(lambda x : 

                                          calendar.month_name[datetime.strptime(x,"%Y-%m-%d").month])
df["weekday"]
df["month"]
dictionnaire_saisons = {2: "Spring", 3 : "Summer", 4 : "Fall", 1 :"Winter" }

df["season"] = df.season.map(dictionnaire_saisons)



df["weather"] = df.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",\

                                        2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ", \

                                        3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", \

                                        4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })
df.head()
categoryVariableList = ["hour","weekday","month","season","weather","holiday","workingday"]



for var in categoryVariableList:

    df[var] = df[var].astype("category")
df  = df.drop(["datetime"],axis=1)
df.head()
df.info()
fig, axes = plt.subplots(nrows=1,ncols=1)

fig.set_size_inches(8, 8)

sn.boxplot(data=df,y="count",orient="v",ax=axes)



axes.set(ylabel='Count',title="Box Plot On Count")
fig, axes = plt.subplots(nrows=1,ncols=1)

fig.set_size_inches(8, 8)

sn.boxplot(data=df,y="count",x="season",orient="v",ax=axes)

axes.set(xlabel='Season', ylabel='Count',title="Box Plot On Count Across Season")
fig, axes = plt.subplots(nrows=1,ncols=1)

fig.set_size_inches(8, 8)

sn.boxplot(data=df,y="count",x="hour",orient="v",ax=axes)

axes.set(xlabel='Hour Of The Day', ylabel='Count',title="Box Plot On Count Across Hour Of The Day")
fig, axes = plt.subplots(nrows=1,ncols=1)

fig.set_size_inches(8, 8)

sn.boxplot(data=df,y="count",x="workingday",orient="v",ax=axes)

axes.set(xlabel='Working Day', ylabel='Count',title="Box Plot On Count Across Working Day")
df.shape[0] - df[np.abs(df["count"] - df["count"].mean()) <= (3 * df["count"].std())].shape[0]
df_without_outliers = df[

    np.abs(df["count"] - df["count"].mean()) <= (3 * df["count"].std())

] 
print ("Shape Of The Before Ouliers: ",df.shape)

print ("Shape Of The After Ouliers: ",df_without_outliers.shape)
corrMatt = df.corr()

mask = np.array(corrMatt)

mask[np.tril_indices_from(mask)] = False

fig,ax= plt.subplots()

fig.set_size_inches(20,10)

sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
fig,ax = plt.subplots()

fig.set_size_inches(12, 5)

sn.regplot(x="registered", y="count", data=df,ax=ax)
sortOrder = ["January","February","March","April","May","June","July","August","September","October","November","December"]

hueOrder = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]



monthAggregated = pd.DataFrame(df.groupby("month")["count"].mean()).reset_index()
fig,ax= plt.subplots()

fig.set_size_inches(12,5)



sn.barplot(data=monthAggregated,x="month",y="count",ax=ax,order=sortOrder)

ax.set(xlabel='Month', ylabel='Avearage Count',title="Average Count By Month")
hourAggregated = pd.DataFrame(df.groupby(["hour","season"],sort=True)["count"].mean()).reset_index()
fig,ax= plt.subplots()

fig.set_size_inches(12,5)

sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"],hue=hourAggregated["season"], data=hourAggregated, join=True,ax=ax)

ax.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across Season",label='big')
hourAggregated = pd.DataFrame(df.groupby(["hour","weekday"],sort=True)["count"].mean()).reset_index()
fig,ax= plt.subplots()

fig.set_size_inches(12,5)



sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"],hue=hourAggregated["weekday"],hue_order=hueOrder, data=hourAggregated, join=True,ax=ax)

ax.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across Weekdays",label='big')
hourTransformed = pd.melt(df[["hour","casual","registered"]], id_vars=['hour'], value_vars=['casual', 'registered'])

hourAggregated = pd.DataFrame(hourTransformed.groupby(["hour","variable"],sort=True)["value"].mean()).reset_index()

hourAggregated.head()
fig,ax= plt.subplots()

fig.set_size_inches(12,5)



sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["value"],hue=hourAggregated["variable"],hue_order=["casual","registered"], data=hourAggregated, join=True,ax=ax)

ax.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across User Type",label='big')
dropFeatures = ["casual","date","registered"]

df = df.drop(dropFeatures, axis=1)
df = pd.get_dummies(df,columns=["holiday"])

df = pd.get_dummies(df,columns=["weekday"])

df = pd.get_dummies(df,columns=["month"])

df = pd.get_dummies(df,columns=["season"])

df = pd.get_dummies(df,columns=["weather"])

df.columns
df.head()
import sklearn

from sklearn.model_selection import train_test_split



X = df.drop('count', axis=1)

Y = df['count']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X_train.shape, X_test.shape)

print(Y_train.shape, Y_test.shape)
from sklearn import metrics



def rmse(y, y_):

    return np.sqrt(np.mean((y-y_)**2))



def mae(y, y_):

    return np.mean(np.abs(y-y_))
from sklearn.dummy import DummyRegressor



dummy_regr = DummyRegressor(strategy = "mean")



dummy_regr.fit(X_train, Y_train)

y_pred = dummy_regr.predict(X_test)



print("R2: ", metrics.r2_score(Y_test, y_pred))

print("MAE: ", mae(Y_test, y_pred))
from sklearn.linear_model import LinearRegression



lModel = LinearRegression()

lModel.fit(X=X_train, y=Y_train)



y_pred = lModel.predict(X= X_test)



print("R2: ", metrics.r2_score(Y_test, y_pred))

print("MAE: ", mae(Y_test, y_pred))
plt.scatter(Y_test, y_pred, color='b')
from sklearn.linear_model import Lasso



lModel = Lasso()

lModel.fit(X=X_train, y=Y_train)



y_pred = lModel.predict(X= X_test)



print("R2: ", metrics.r2_score(Y_test, y_pred))

print("MAE: ", mae(Y_test, y_pred))
from sklearn.linear_model import Ridge



lModel = Ridge()

lModel.fit(X=X_train, y=Y_train)



y_pred = lModel.predict(X= X_test)



print("R2: ", metrics.r2_score(Y_test, y_pred))

print("MAE: ", mae(Y_test, y_pred))
from sklearn import tree



lModel = tree.DecisionTreeRegressor()

lModel.fit(X=X_train, y=Y_train)



y_pred = lModel.predict(X= X_test)



print("R2: ", metrics.r2_score(Y_test, y_pred))

print("MAE: ", mae(Y_test, y_pred))
#tree.plot_tree(lModel) 
from sklearn.ensemble import RandomForestRegressor



lModel = RandomForestRegressor()

lModel.fit(X=X_train, y=Y_train)



y_pred = lModel.predict(X= X_test)



print("R2: ", metrics.r2_score(Y_test, y_pred))

print("MAE: ", mae(Y_test, y_pred))
plt.scatter(Y_test, y_pred, color='b')
from sklearn.ensemble import ExtraTreesRegressor



lModel = ExtraTreesRegressor()

lModel.fit(X=X_train, y=Y_train)



y_pred = lModel.predict(X= X_test)



print("R2: ", metrics.r2_score(Y_test, y_pred))

print("MAE: ", mae(Y_test, y_pred))
from sklearn.ensemble import GradientBoostingRegressor



lModel = GradientBoostingRegressor(n_estimators=500)

lModel.fit(X = X_train, y = Y_train)



y_pred = lModel.predict(X= X_test)



print("R2: ", metrics.r2_score(Y_test, y_pred))

print("MAE: ", mae(Y_test, y_pred))
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



lm = LinearRegression()

mae_scorer = metrics.make_scorer(mae, greater_is_better=False)



# Train the model with 10-folds

# Warning : without `scoring` parameter it returns a R2 score ! 

scores = cross_val_score(lm, scaler.fit_transform(X_train), Y_train, cv = 10)#, scoring=mae_scorer) 

print("Cross-validated scores:", scores.mean())
scores
lm = Ridge()

mae_scorer = metrics.make_scorer(mae, greater_is_better=False)



# Train the model with 10-folds

# Warning : without `scoring` parameter it returns a R2 score ! 

scores = cross_val_score(lm, scaler.fit_transform(X_train), Y_train, cv=10)#, scoring=mae_scorer) 

print("Cross-validated scores:", scores.mean())
scores
lm = Lasso()

mae_scorer = metrics.make_scorer(mae, greater_is_better=False)



# Train the model with 10-folds

# Warning : without `scoring` parameter it returns a R2 score ! 

scores = cross_val_score(lm, scaler.fit_transform(X_train), Y_train, cv=10)#, scoring=mae_scorer) 

print("Cross-validated scores:", scores.mean())
scores
from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV



ridge_m_ = Ridge()

ridge_params_ = {'alpha':[25, 30, 35, 40, 50, 60, 70, 100, 1000]}

#rmsle_scorer = metrics.make_scorer(rmse, greater_is_better=False)



grid_ridge_m = GridSearchCV(ridge_m_, 

                            ridge_params_, 

                            #scoring = rmsle_scorer,

                            cv = 5)



grid_ridge_m.fit(X_train, Y_train)

preds = grid_ridge_m.predict(X=X_test)



print ("Best parameters:", grid_ridge_m.best_params_)

print ("R2 Value For Ridge Regression: ", metrics.r2_score(Y_test, preds))

#print ("RMSE Value For Ridge Regression: ", rmse(Y_test, preds))

print ("MAE Value For Ridge Regression: ", mae(Y_test, preds))
pd.DataFrame(grid_ridge_m.cv_results_)
# Plot scores

fig,ax= plt.subplots()

fig.set_size_inches(7,3)

df = pd.DataFrame(grid_ridge_m.cv_results_)[['params','mean_test_score']]

df["alpha"] = df["params"].apply(lambda x:x["alpha"])

df["r2"] = df["mean_test_score"].apply(lambda x:x)

sn.pointplot(data=df,x="alpha",y="r2",ax=ax)
from sklearn.linear_model import Lasso



lasso_m_ = Lasso()



alpha  = 1/np.array([1, 2, 3, 4, 10, 30,100,200,300,400,800,900,1000])

lasso_params_ = { 'alpha':alpha}

grid_lasso_m = GridSearchCV(lasso_m_,

                            lasso_params_,

                            #scoring = rmsle_scorer,

                            cv=5)



grid_lasso_m.fit(X_train, Y_train)

preds = grid_lasso_m.predict(X=X_test)



print ("Best parameters:", grid_lasso_m.best_params_)

print ("R2 Value For Lasso Regression: ", metrics.r2_score(Y_test, preds))

print ("RMSE Value For Lasso Regression: ", rmse(Y_test, preds))

print ("MAE Value For Lasso Regression: ", mae(Y_test, preds))
# Plot scores

fig,ax= plt.subplots()

fig.set_size_inches(7,3)

df = pd.DataFrame(grid_lasso_m.cv_results_)[['params','mean_test_score']]

df["alpha"] = df["params"].apply(lambda x:x["alpha"])

df["r2"] = df["mean_test_score"].apply(lambda x:x)

sn.pointplot(data=df,x="alpha",y="r2",ax=ax)
from sklearn.ensemble import RandomForestRegressor



rf_Model = RandomForestRegressor(n_jobs = -1)



rf_params = { 'n_estimators':[150],'max_depth':[17, 20, 22, 24, 25, 30]}

grid_rf_m = GridSearchCV(rf_Model, 

                         rf_params, 

                         #scoring=rmsle_scorer, 

                         cv = 5,

                         n_jobs = -1)



grid_rf_m.fit(X_train, Y_train)

preds = grid_rf_m.predict(X=X_test)



print ("Best parameters:", grid_rf_m.best_params_)

print ("R2 Value For Random Forest: ", metrics.r2_score(Y_test, preds))

print ("RMSE Value For Random Forest: ", rmse(Y_test, preds))

print ("MAE Value For Random Forest: ", mae(Y_test, preds))
# Plot scores

fig,ax= plt.subplots()

fig.set_size_inches(7,3)

df = pd.DataFrame(grid_rf_m.cv_results_)[['params','mean_test_score']]

df["max_depth"] = df["params"].apply(lambda x:x["max_depth"])

df["r2"] = df["mean_test_score"].apply(lambda x:x)

sn.pointplot(data=df,x="max_depth",y="r2",ax=ax)
from sklearn.ensemble import GradientBoostingRegressor

gbm = GradientBoostingRegressor()



gbm_params = {'n_estimators':[500], 'learning_rate':[0.001 ,0.01, 0.1]}

grid_gbm = GridSearchCV(gbm, 

                        gbm_params, 

                        #scoring=rmsle_scorer, 

                        cv=5)



grid_gbm.fit(X_train, Y_train)

preds = grid_gbm.predict(X=X_test)



print ("Best parameters:", grid_gbm.best_params_)

print ("R2 Value For Gradient Boosting: ", metrics.r2_score(Y_test, preds))

print ("RMSE Value For Gradient Boosting: ", rmse(Y_test, preds))

print ("MAE Value For Gradient Boosting: ", mae(Y_test, preds))
# Plot scores

fig,ax= plt.subplots()

fig.set_size_inches(7,3)

df = pd.DataFrame(grid_gbm.cv_results_)[['params','mean_test_score']]

df["learning_rate"] = df["params"].apply(lambda x:x["learning_rate"])

df["r2"] = df["mean_test_score"].apply(lambda x:x)

sn.pointplot(data=df,x="learning_rate",y="r2",ax=ax)
lm = RandomForestRegressor(max_depth= 22, n_estimators= 150)



# Warning : without `scoring` parameter it returns a R2 score ! 

scores = cross_val_score(lm, X_train, Y_train, cv=5)

print("Cross-validated scores:", scores.mean())
scores
lm = RandomForestRegressor(max_depth= 22, n_estimators= 150)

lm.fit(X_train, Y_train)

y_pred = lm.predict(X_test)



print("R2: ", metrics.r2_score(Y_test, y_pred))

print("MAE: ", mae(Y_test, y_pred))
from catboost import CatBoostRegressor



cb = CatBoostRegressor(verbose = 0, cat_features = ['workingday','hour'])



cb.fit(X_train, Y_train)

y_pred = cb.predict(X_test)



# Warning : without `scoring` parameter it returns a R2 score ! 

scores = cross_val_score(cb, X_train, Y_train, cv=5)



print("Cross-validated scores:", scores.mean())



print("R2: ", metrics.r2_score(Y_test, y_pred))

print("MAE: ", mae(Y_test, y_pred))
import lightgbm as lgb



lgb_train = lgb.Dataset(X_train, Y_train)

lgb_test = lgb.Dataset(X_test, Y_test, reference=lgb_train)



params = {

    'num_leaves': 5,

    'metric': ['l1', 'l2'],

    'verbose': -1,

    'n_estimators': 750

}



evals_result = {}  # to record eval results for plotting

gbm = lgb.train(params,

                lgb_train,

                num_boost_round=100,

                valid_sets=[lgb_train, lgb_test],

                feature_name=['f' + str(i + 1) for i in range(X_train.shape[-1])],

                categorical_feature=[21],

                evals_result=evals_result,

                verbose_eval=0)



y_pred = gbm.predict(X_test)



print("R2: ", metrics.r2_score(Y_test, y_pred))

print("MAE: ", mae(Y_test, y_pred))
import xgboost as xgb



X_train['hour'] = X_train['hour'].astype('int8')

X_train['workingday'] = X_train['workingday'].astype('int8')



X_test['hour'] = X_test['hour'].astype('int8')

X_test['workingday'] = X_test['workingday'].astype('int8')



#gbmdata_dmatrix = xgb.DMatrix(data=X_train, label=Y_train)



xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.9, learning_rate = 0.1, n_estimators = 500)



# Apprentissage sur X_train

xg_reg.fit(X_train,Y_train)



y_pred = xg_reg.predict(X_test)



print("R2: ", metrics.r2_score(Y_test, y_pred))

print("MAE: ", mae(Y_test, y_pred))