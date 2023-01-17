import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# for the Q-Q plots
import scipy.stats as stats

# to split the datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
# from feature-engine
from feature_engine import missing_data_imputers as mdi

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler,RobustScaler

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/hitters/Hitters.csv')
data
data.info()
data.Salary.unique()
fig = data.Salary.hist(bins=50)

fig.set_title("Salaries")
fig.set_xlabel("Salary mount")
fig.set_ylabel("Number of Salaries")
data.groupby("Division")["Salary"].mean()
data.groupby("League")["Salary"].mean()
data.groupby("League")["Salary"].median()
# we call the imputer from feature-engine
#we specify the imputation strategy,median in this case

imputer = mdi.MeanMedianImputer(imputation_method='mean',
                                       variables=['Salary'])

imputer.fit(data)

imputer.variables

tmp = imputer.transform(data)
tmp.head()
data.describe().T
tmp.describe().T

from sklearn.neighbors import LocalOutlierFactor
y=tmp.Salary
x=tmp.drop(['Salary', 'League','NewLeague','Division'], axis=1, inplace=False)
columns=x.columns.tolist()
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred=clf.fit_predict(x)

y_pred
y_inlier= np.count_nonzero(y_pred == -1)
y_inlier  #outlier olmayanlar inlier


X_score=clf.negative_outlier_factor_
outlier_score=pd.DataFrame()
outlier_score['score']=X_score
with pd.option_context("display.max_rows", 1000):
    display(outlier_score.sort_values(by='score', ascending=True))

threshold= -2.129953
filtre=outlier_score['score']<threshold
outlier_index=outlier_score[filtre].index.tolist()
outlier_index_x=x.drop(outlier_index)

outlier_index_y=y.drop(outlier_index)
tmp=outlier_index_x
tmp['Salary'] = outlier_index_y
tmp["League"] = data["League"]
tmp["Division"] = data["Division"]
tmp["NewLeague"] = data["NewLeague"]
tmp
# function to create histogram, Q-Q plot and
# boxplot. We learned this in section 3 of the course


def diagnostic_plots(df, variable):
    # function takes a dataframe (df) and
    # the variable of interest as arguments

    # define figure size
    plt.figure(figsize=(16, 4))

    # histogram
    plt.subplot(1, 3, 1)
    sns.distplot(df[variable], bins=30)
    plt.title('Histogram')

    # Q-Q plot
    plt.subplot(1, 3, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.ylabel('Variable quantiles')

    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')

    plt.show()

def find_skewed_boundaries(df, variable):
    # Let's calculate the boundaries outside which sit the outliers
    # for skewed distributions

    # distance passed as an argument, gives us the option to
    # estimate 1.5 times or 3 times the IQR to calculate
    # the boundaries.

    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)

    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)


    return upper_boundary, lower_boundary
def find_boundaries_with_quantiles(df, variable):
   
    #Better Result
    #the boundaries are the quantiles
    
    lower_boundary = df[variable].quantile(0.05)
    upper_boundary = df[variable].quantile(0.95)
        

    return upper_boundary, lower_boundary
#Find limits for Salary
RM_upper_limit, RM_lower_limit = find_boundaries_with_quantiles(tmp, 'Salary')
RM_upper_limit, RM_lower_limit

# let's trimm the dataset
tmp[tmp["Salary"] > RM_upper_limit] = RM_upper_limit
tmp[tmp["Salary"] < RM_lower_limit] = RM_lower_limit
RM_upper_limit, RM_lower_limit = find_boundaries_with_quantiles(tmp, 'Years')
RM_upper_limit, RM_lower_limit

tmp[tmp["Years"] > RM_upper_limit] = RM_upper_limit
tmp[tmp["Years"] < RM_lower_limit] = RM_lower_limit
RM_upper_limit, RM_lower_limit = find_boundaries_with_quantiles(tmp, 'Errors')
RM_upper_limit, RM_lower_limit

# let's trimm the dataset
tmp[tmp["Errors"] > RM_upper_limit] = RM_upper_limit
tmp[tmp["Errors"] < RM_lower_limit] = RM_lower_limit

RM_upper_limit, RM_lower_limit = find_boundaries_with_quantiles(tmp, 'PutOuts')
RM_upper_limit, RM_lower_limit

# let's trimm the dataset
tmp[tmp["PutOuts"] > RM_upper_limit] = RM_upper_limit
tmp[tmp["PutOuts"] < RM_lower_limit] = RM_lower_limit

CRM_upper_limit, RM_lower_limit = find_boundaries_with_quantiles(tmp, 'CHmRun')
RM_upper_limit, RM_lower_limit


tmp[tmp["CHmRun"] > RM_upper_limit] = RM_upper_limit
tmp[tmp["CHmRun"] < RM_lower_limit] = RM_lower_limit
RM_upper_limit, RM_lower_limit = find_boundaries_with_quantiles(tmp, 'HmRun')
RM_upper_limit, RM_lower_limit

# let's trimm the dataset

tmp[tmp["HmRun"] > RM_upper_limit] = RM_upper_limit
tmp[tmp["HmRun"] < RM_lower_limit] = RM_lower_limit
RM_upper_limit, RM_lower_limit = find_boundaries_with_quantiles(tmp, 'Assists')
RM_upper_limit, RM_lower_limit

# let's trimm the dataset
tmp[tmp["Assists"] > RM_upper_limit] = RM_upper_limit
tmp[tmp["Assists"] < RM_lower_limit] = RM_lower_limit

RM_upper_limit, RM_lower_limit = find_boundaries_with_quantiles(tmp, 'Runs')
RM_upper_limit, RM_lower_limit

# let's trimm the dataset
tmp[tmp["Runs"] > RM_upper_limit] = RM_upper_limit
tmp[tmp["Runs"] < RM_lower_limit] = RM_lower_limit
RM_upper_limit, RM_lower_limit = find_boundaries_with_quantiles(tmp, 'Errors')
RM_upper_limit, RM_lower_limit

# let's trimm the dataset
tmp[tmp["Errors"] > RM_upper_limit] = RM_upper_limit
tmp[tmp["Errors"] < RM_lower_limit] = RM_lower_limit
RM_upper_limit, RM_lower_limit = find_boundaries_with_quantiles(tmp, 'CRBI')
RM_upper_limit, RM_lower_limit

# let's trimm the dataset
tmp[tmp["CRBI"] > RM_upper_limit] = RM_upper_limit
tmp[tmp["CRBI"] < RM_lower_limit] = RM_lower_limit
RM_upper_limit, RM_lower_limit = find_boundaries_with_quantiles(tmp, 'Hits')
RM_upper_limit, RM_lower_limit

# let's trimm the dataset
tmp[tmp["Hits"] > RM_upper_limit] = RM_upper_limit
tmp[tmp["Hits"] < RM_lower_limit] = RM_lower_limit
tmp = pd.get_dummies(tmp, columns = ['League', 'Division', 'NewLeague'], drop_first = True)
tmp.head()
# Separate into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
            tmp.drop(['Salary'], axis=1),
            tmp['Salary'], test_size=0.2, random_state=46)

X_train.shape,X_test.shape,
train_t= X_train
test_t= X_test

train_t.shape,test_t.shape
from sklearn.preprocessing import StandardScaler,RobustScaler

# let's scale the features
scaler = RobustScaler()
scaler.fit(train_t)
# for linear regression

# model build using the natural distributions

# call the model
linreg = LinearRegression()

# fit the model
linreg.fit(scaler.transform(train_t), y_train)

# make predictions and calculate the mean squared
# error over the train set
print('Train set')
pred = linreg.predict(scaler.transform(train_t))
print('Linear Regression mse: {}'.format(mean_squared_error(y_train, pred)))

# make predictions and calculate the mean squared
# error over the test set
print('Test set')
pred = linreg.predict(scaler.transform(test_t))
print('Linear Regression mse: {}'.format(mean_squared_error(y_test, pred)))
print(np.sqrt(mean_squared_error(y_test, pred)))
from sklearn.neighbors import KNeighborsRegressor

knn_model=KNeighborsRegressor().fit(train_t,y_train)
y_pred=knn_model.predict(test_t)
np.sqrt(mean_squared_error(y_test,y_pred))
from sklearn.svm import SVR

svr_model=SVR().fit(train_t,y_train)
y_pred=svr_model.predict(test_t)
np.sqrt(mean_squared_error(y_test,y_pred))
from sklearn.neural_network import MLPRegressor

mlp_model = MLPRegressor().fit(train_t, y_train)
y_pred = mlp_model.predict(test_t)
np.sqrt(mean_squared_error(y_test, y_pred))
from sklearn.tree import DecisionTreeRegressor

cart_model=DecisionTreeRegressor().fit(train_t,y_train)
y_pred = mlp_model.predict(test_t)
np.sqrt(mean_squared_error(y_test, y_pred))
#Random Forest
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor().fit(train_t, y_train)
y_pred = mlp_model.predict(test_t)
np.sqrt(mean_squared_error(y_test, y_pred))
#GBM

from sklearn.ensemble import GradientBoostingRegressor

gbm_model=GradientBoostingRegressor().fit(train_t,y_train)
y_pred=gbm_model.predict(test_t)
np.sqrt(mean_squared_error(y_test, y_pred))
# XGB

import xgboost
from xgboost import XGBRegressor

xgb = XGBRegressor().fit(train_t, y_train)
y_pred = xgb.predict(test_t)
np.sqrt(mean_squared_error(y_test, y_pred))
#Light GBM
#!pip install lightgbm

from lightgbm import LGBMRegressor
lgb_model = LGBMRegressor().fit(train_t, y_train)
y_pred = lgb_model.predict(test_t)
np.sqrt(mean_squared_error(y_test, y_pred))
# Cat Boost

from catboost import CatBoostRegressor
catb_model = CatBoostRegressor(verbose = False).fit(train_t, y_train)
y_pred = catb_model.predict(test_t)
np.sqrt(mean_squared_error(y_test, y_pred))
import lightgbm as lgb
from lightgbm import LGBMRegressor
models = []

models.append(('Regression', LinearRegression()))
models.append(('Ridge', Ridge()))
models.append(('Lasso', Lasso()))
models.append(('ElasticNet', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('RF', RandomForestRegressor()))
models.append(('SVR', SVR()))
models.append(('GBM', GradientBoostingRegressor()))
models.append(("XGBoost", XGBRegressor()))
models.append(("LightGBM", LGBMRegressor()))
models.append(("CatBoost", CatBoostRegressor(verbose = False)))

for name, model in models:
    model.fit(train_t,y_train)
    y_pred=model.predict(test_t)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(name,rmse)
lgbm_params = {"learning_rate": [0.01,0.001, 0.1, 0.5, 1],
              "n_estimators": [200,500,1000,5000],
              "max_depth": [2,4,6,7,10],
              "colsample_bytree": [1,0.8,0.5,0.4]}

lgb_model = LGBMRegressor()

clf=GridSearchCV(lgb_model,lgbm_params,cv=n_folds,scoring="neg_mean_squared_error",refit=True)
clf.fit(train_t,y_train)
scores=clf.cv_results_["mean_test_score"]
scores_std= clf.cv_results_["std_test_score"]

y_predicted_dummy=clf.predict(test_t)
mse=mean_squared_error(y_test, y_predicted_dummy)
np.sqrt(mse)
# Ridge Regression Tuning

from sklearn.linear_model import RidgeCV

alpha = [0.01,0.002,0.5,0.7,1]


ridge=Lasso(random_state=46,max_iter=1000)
alphas=np.linspace(0,1,1000)
tuned_parameters=[ {"alpha":alpha} ]
n_folds=10

clf=GridSearchCV(ridge,tuned_parameters,cv=n_folds,scoring="neg_mean_squared_error",refit=True)
clf.fit(train_t,y_train)
scores=clf.cv_results_["mean_test_score"]
scores_std= clf.cv_results_["std_test_score"]
clf.best_estimator_.coef_

ridge= clf.best_estimator_
y_predicted_dummy=clf.predict(test_t)
mse=mean_squared_error(y_test, y_predicted_dummy)
print(np.sqrt(mse))


ridge=Lasso(random_state=46,max_iter=1000)
alphas = [0.01,0.002,0.5,0.7,1]
tuned_parameters=[ {"alpha":alphas} ]
n_folds=10

clf=GridSearchCV(ridge,tuned_parameters,cv=n_folds,scoring="neg_mean_squared_error",refit=True)
clf.fit(train_t,y_train)
scores=clf.cv_results_["mean_test_score"]
scores_std= clf.cv_results_["std_test_score"]
clf.best_estimator_.coef_

ridge= clf.best_estimator_
y_predicted_dummy=clf.predict(test_t)
mse=mean_squared_error(y_test, y_predicted_dummy)
np.sqrt(mse)
ridge=ElasticNet(random_state=46,max_iter=1000)
enet_params = {"l1_ratio": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
              "alpha":[0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]}
tuned_parameters=[ {"alpha":enet_params} ]
n_folds=10

clf=GridSearchCV(ridge,enet_params,cv=n_folds,scoring="neg_mean_squared_error",refit=True)
clf.fit(train_t,y_train)
scores=clf.cv_results_["mean_test_score"]
scores_std= clf.cv_results_["std_test_score"]
clf.best_estimator_.coef_

ridge= clf.best_estimator_
y_predicted_dummy=clf.predict(test_t)
mse=mean_squared_error(y_test, y_predicted_dummy)
np.sqrt(mse)
# KNN Tuning


knn_params = {"n_neighbors": np.arange(2,30,1)}


knn_model = KNeighborsRegressor()
clf=GridSearchCV(knn_model,knn_params,cv=n_folds,scoring="neg_mean_squared_error",refit=True)
clf.fit(train_t,y_train)
scores=clf.cv_results_["mean_test_score"]
scores_std= clf.cv_results_["std_test_score"]


ridge= clf.best_estimator_
y_predicted_dummy=clf.predict(test_t)
mse=mean_squared_error(y_test, y_predicted_dummy)
np.sqrt(mse)
# SVR
svr_params={"C":[1,2,3,5,10]}


svr = SVR()
clf=GridSearchCV(svr,svr_params,cv=n_folds,scoring="neg_mean_squared_error",refit=True)
clf.fit(train_t,y_train)
scores=clf.cv_results_["mean_test_score"]
scores_std= clf.cv_results_["std_test_score"]

y_predicted_dummy=clf.predict(test_t)
mse=mean_squared_error(y_test, y_predicted_dummy)
np.sqrt(mse)
#ANN

mlp_params = {"alpha": [0.1, 0.01, 0.02, 0.001, 0.0001], 
             "hidden_layer_sizes": [(10,20), (5,5), (100,100), (1000,100,10)]}

mlp_model = MLPRegressor()
mlp_cv = GridSearchCV(mlp_model, mlp_params, cv = 10, verbose = 2, n_jobs = -1)
clf.fit(train_t,y_train)
scores=clf.cv_results_["mean_test_score"]
scores_std= clf.cv_results_["std_test_score"]

y_predicted_dummy=clf.predict(test_t)
mse=mean_squared_error(y_test, y_predicted_dummy)
np.sqrt(mse)
#CART

cart_params= {"max_depth": [5,6,10,15,20,50,100,1000],
             "min_samples_split":[20,30,40,50]}

cart_model=DecisionTreeRegressor()
clf=GridSearchCV(cart_model,cart_params,cv=n_folds,scoring="neg_mean_squared_error",refit=True)
clf.fit(train_t,y_train)
scores=clf.cv_results_["mean_test_score"]
scores_std= clf.cv_results_["std_test_score"]

y_predicted_dummy=clf.predict(test_t)
mse=mean_squared_error(y_test, y_predicted_dummy)
np.sqrt(mse)
#RF

rf_model=RandomForestRegressor()
rf_params = {"max_depth": [5,8,10,3],
            "max_features": [2,5,10,15,17],
            "n_estimators": [100,200, 500, 1000],
            "min_samples_split": [10,20,30,40,50]}


clf=GridSearchCV(rf_model,rf_params,cv=n_folds,scoring="neg_mean_squared_error",refit=True)
clf.fit(train_t,y_train)
scores=clf.cv_results_["mean_test_score"]
scores_std= clf.cv_results_["std_test_score"]

y_predicted_dummy=clf.predict(test_t)
mse=mean_squared_error(y_test, y_predicted_dummy)
np.sqrt(mse)
# GBM

gbm_params = {"learning_rate": [0.001,0.1,0.01, 0.05],
             "max_depth": [3,5,8,9,10],
             "n_estimators": [200,500,1000,1500],
             "subsample": [1,0.4,0.5,0.7],
             "loss": ["ls","lad","quantile"]}


gbm_model = GradientBoostingRegressor()


clf=GridSearchCV(gbm_model,gbm_params,cv=n_folds,scoring="neg_mean_squared_error",refit=True)
clf.fit(train_t,y_train)
scores=clf.cv_results_["mean_test_score"]
scores_std= clf.cv_results_["std_test_score"]

y_predicted_dummy=clf.predict(test_t)
mse=mean_squared_error(y_test, y_predicted_dummy)
np.sqrt(mse)

#XGBoost

xgb_params = {"learning_rate": [0.1,0.01,0.5],
             "max_depth": [2,3,4,5,8],
             "n_estimators": [100,200,500,1000],
             "colsample_bytree": [0.4,0.7,1]}

xgb = XGBRegressor()


clf=GridSearchCV(xgb,xgb_params,cv=n_folds,scoring="neg_mean_squared_error",refit=True)
clf.fit(train_t,y_train)
scores=clf.cv_results_["mean_test_score"]
scores_std= clf.cv_results_["std_test_score"]

y_predicted_dummy=clf.predict(test_t)
mse=mean_squared_error(y_test, y_predicted_dummy)
np.sqrt(mse)



#CatBoost

catb_params = {"iterations": [200,500,100],
              "learning_rate": [0.01,0.1],
              "depth": [3,6,8]}

catb_model = CatBoostRegressor(verbose = False)

clf=GridSearchCV(catb_model,catb_params,cv=n_folds,scoring="neg_mean_squared_error",refit=True)
clf.fit(train_t,y_train)
scores=clf.cv_results_["mean_test_score"]
scores_std= clf.cv_results_["std_test_score"]

y_predicted_dummy=clf.predict(test_t)
mse=mean_squared_error(y_test, y_predicted_dummy)
np.sqrt(mse)




models = []

models.append(('Ridge', Ridge(alpha=0.01)))
models.append(('Lasso', Lasso(alpha=0.01)))
models.append(('ElasticNet', ElasticNet(alpha=0.001, l1_ratio=0.95)))
models.append(('KNN', KNeighborsRegressor(n_neighbors=5)))
models.append(('CART', DecisionTreeRegressor(max_depth= 4, min_samples_split= 50)))
models.append(('RF', RandomForestRegressor()))
models.append(('SVR', SVR(C=1000)))
models.append(('GBM', GradientBoostingRegressor()))
models.append(("XGBoost", XGBRegressor()))
models.append(("LightGBM", LGBMRegressor()))
models.append(("CatBoost", CatBoostRegressor(verbose = False)))

for name, model in models:
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(name,rmse)