import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
plt.style.use('ggplot')
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from scipy.stats import skew,norm
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
import lightgbm as lgb
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train
train.dtypes
train['SalePrice'].describe()
sns.distplot(train['SalePrice'])
#skewness and kurtosis
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())
train.describe()
corrmat = train.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
k = 11
cols = train.corr().nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale = 1.25)
plt.figure(figsize = (30,9))
hm = sns.heatmap(cm, cbar=True, annot=True, \
                 square=True, fmt='.2f', annot_kws={'size': 14}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
sns.distplot(train['1stFlrSF'])
sns.distplot(train['GrLivArea'])
plt.figure(figsize=(12,6))
plt.scatter(x=train.GrLivArea, y=train.SalePrice)
plt.xlabel("GrLivArea", fontsize=13)
plt.ylabel("SalePrice", fontsize=13)
plt.ylim(0,800000)
train.drop(train[(train["GrLivArea"]>4000)&(train["SalePrice"]<300000)].index,inplace=True)
plt.figure(figsize=(12,6))
plt.scatter(x=train.OverallQual, y=train.SalePrice)
plt.xlabel("OverallQual", fontsize=13)
plt.ylabel("SalePrice", fontsize=13)
train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace=True)
full=pd.concat([train,test], ignore_index=True)
full.drop(['Id'],axis=1, inplace=True)
full.shape
# Some of the non-numeric predictors are stored as numbers; we convert them into strings 
full['MSSubClass'] = full['MSSubClass'].apply(str)
full['YrSold'] = full['YrSold'].astype(str)
full['MoSold'] = full['MoSold'].astype(str)
# missing data
total = full.isnull().sum()
total = total[total>0]
percent = (full.isnull().sum()/full.isnull().count())
percent = percent[percent>0]
dtypes = full.dtypes
nulls = np.sum(full.isnull())
dtypes2 = dtypes.loc[(nulls != 0)]
missing_data = pd.concat([total,percent,dtypes2], axis = 1).sort_values(by=0,ascending=False)
missing_data
# Before getting hands dirty with missing values, I will creat dummy variables for all non-missing categorical features.
full_missing_col = [col for col in full.columns if full[col].isnull().any()]                                  
full_predictors = full.drop(full_missing_col, axis=1)
full_category_cols= [cname for cname in full_predictors.columns if 
full_predictors[cname].dtype == "object"]
full_category_cols
# create numerical columns list for future use.
full_num_cols= [cname for cname in full.columns if full[cname].dtype != "object"]
full_num_cols
full = pd.get_dummies(full,columns=full_category_cols)
for col in ("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
           "GarageType", "GarageFinish", "GarageQual", "GarageCond",
           "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
            "BsmtFinType2", "MasVnrType"):
    full[col] = full[col].fillna("None")
for col in ("GarageArea", "GarageCars", "BsmtFinSF1", 
           "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea",
           "BsmtFullBath", "BsmtHalfBath"):
    full[col] = full[col].fillna(0)
train.groupby(['Neighborhood'])[['LotFrontage']].agg(['mean','median','count'])
train["LotFrontage"] = train.groupby(["Neighborhood"])[["LotFrontage"]].transform(lambda x: x.fillna(x.median())) 
test["LotFrontage"] =  test.groupby(["Neighborhood"])[['LotFrontage']].transform(lambda x: x.fillna(x.median()))
full["LotFrontage"] = pd.concat([train["LotFrontage"],test["LotFrontage"]],axis =0,ignore_index=True)
test["LotFrontage"].describe()
full["LotFrontage"].describe()
sns.stripplot(x="PoolQC", y="SalePrice", data=full, size = 5, jitter = True)
full.drop('PoolQC', axis=1, inplace=True)
sns.stripplot(x="MiscFeature", y="SalePrice", data=full, size = 5, jitter = True)
# There are only a low number of houses in this area with any miscalleanous features.
full.drop(['MiscFeature'],axis=1, inplace=True)
sns.stripplot(x="Alley", y="SalePrice", data=full, size = 5, jitter = True)
full['Alley'] = full['Alley'].map({"None":0, "Grvl":1, "Pave":2})
full.head(3)
sns.stripplot(x="Fence", y="SalePrice", data=full, size = 5, jitter = True)
full['Fence'] = full['Fence'].map({"None":0, "MnWw":1, "MnPrv":2, "GdPrv":3, "GdWo":4})
full.head(3)
sns.stripplot(x="FireplaceQu", y="SalePrice", data=full, size = 5, jitter = True)
# this is a categorical feature with order, I will replace the values by hand.
full['FireplaceQu'] = full['FireplaceQu'].map({"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
full['FireplaceQu'].unique()
sns.stripplot(x="GarageCond", y="SalePrice", data=full, size = 5, jitter = True)
full['GarageCond'] = full['GarageCond'].map({"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
sns.stripplot(x="GarageQual", y="SalePrice", data=full, size = 5, jitter = True)
full['GarageQual'] = full['GarageQual'].map({"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
sns.stripplot(x="GarageFinish", y="SalePrice", data=full, size = 5, jitter = True)
full['GarageFinish'] = full['GarageFinish'].map({"None":0, "Unf":1, "RFn":2, "Fin":3})
sns.stripplot(x="GarageType", y="SalePrice", data=full, size = 5, jitter = True)
full['GarageType'] = full['GarageType'].map({"None":0,"2Types":1,"Attchd":2,"Basment":3,"BuiltIn":4,"CarPort":5,"Detchd":6})
sns.stripplot(x="BsmtExposure", y="SalePrice", data=full, size = 5, jitter = True)
full['BsmtExposure'] = full['BsmtExposure'].map({"None":0, "No":1, "Mn":2, "Av":3, "Gd":4})
sns.stripplot(x="BsmtCond", y="SalePrice", data=full, size = 5, jitter = True)
full['BsmtCond'] = full['BsmtCond'].map({"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
sns.stripplot(x="BsmtQual", y="SalePrice", data=full, size = 5, jitter = True)
full['BsmtQual'] = full['BsmtQual'].map({"None":0, "Fa":1, "TA":2, "Gd":3, "Ex":4})
sns.stripplot(x="BsmtFinType2", y="SalePrice", data=full, size = 5, jitter = True)
full['BsmtFinType2'] = full['BsmtFinType2'].map({"None":0, "Unf":1, "LwQ":2, "Rec":3, "BLQ":4, "ALQ":5,"GLQ":6})
sns.stripplot(x="BsmtFinType1", y="SalePrice", data=full, size = 5, jitter = True)
full['BsmtFinType1'] = full['BsmtFinType1'].map({"None":0, "Unf":1, "LwQ":2, "Rec":3, "BLQ":4, "ALQ":5,"GLQ":6})
sns.stripplot(x="MasVnrType", y="SalePrice", data=full, size = 5, jitter = True)
full = pd.get_dummies(full, columns = ["MasVnrType"], prefix="MasVnrType")
sns.stripplot(x="MSZoning", y="SalePrice", data=full, size = 5, jitter = True)
full = pd.get_dummies(full, columns = ["MSZoning"], prefix="MSZoning")
plt.subplots(figsize =(15, 5))
plt.subplot(1, 2, 1)
g = sns.countplot(x = "Utilities", data = train).set_title("Utilities - Training")
plt.subplot(1, 2, 2)
g = sns.countplot(x = "Utilities", data = test).set_title("Utilities - Test")
# Since there is only one category for Utilities feature, therefore I will drop this feature.
full = full.drop(['Utilities'], axis=1)
# 'GarageYrBlt' is missing 164 values. I assume 'GarageYrBlt' equal to YearBuilt time.Thus these two columns will be highly correlated. Therefore, I delete the 'GarageYrBlt'.
full['GarageYrBlt'] = full['GarageYrBlt'].fillna(full['YearBuilt'])
for col in ( "GarageArea", "GarageCars", "BsmtFinSF1", 
           "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea",
           "BsmtFullBath", "BsmtHalfBath"):
    full[col] = full[col].fillna(0)
full['Electrical'] = full['Electrical'].fillna(full['Electrical'].mode()[0])
full['KitchenQual'] = full['KitchenQual'].fillna(full['KitchenQual'].mode()[0])
full['Exterior1st'] = full['Exterior1st'].fillna(full['Exterior1st'].mode()[0])
full['Exterior2nd'] = full['Exterior2nd'].fillna(full['Exterior2nd'].mode()[0])
full['SaleType'] = full['SaleType'].fillna(full['SaleType'].mode()[0])
full["Functional"] = full["Functional"].fillna(full['Functional'].mode()[0])
full = pd.get_dummies(full,columns=['Electrical','KitchenQual','Exterior1st','Exterior2nd','SaleType',"Functional"])
# check missing data again
total = full.isnull().sum()
total = total[total>0]
percent = (full.isnull().sum()/full.isnull().count())
percent = percent[percent>0]
dtypes = full.dtypes
nulls = np.sum(full.isnull())
dtypes2 = dtypes.loc[(nulls != 0)]
missing_data = pd.concat([total,percent,dtypes2], axis = 1).sort_values(by=0,ascending=False)
missing_data
# Adding total sqfootage feature 
full['TotalSF'] = full['TotalBsmtSF'] + full['1stFlrSF'] + full['2ndFlrSF']
# Check skewness for all numerical variables
skew_features = full[full_num_cols + ['TotalSF']].apply(lambda x: skew(x)).sort_values(ascending=False)
skews = pd.DataFrame({'skew':skew_features})
# Box-Cox Transformation
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

high_skew = skew_features[abs(skew_features) > 0.75]
skew_index = high_skew.index

for i in skew_index:
    full[i]= boxcox1p(full[i], boxcox_normmax(full[i]+1))

skew_features2 = full[full_num_cols].apply(lambda x: skew(x)).sort_values(ascending=False)
skews2 = pd.DataFrame({'skew':skew_features2})
skews2
# logistic for SalePrice
plt.subplot(1, 2, 1)
sns.distplot(train.SalePrice, kde=False, fit = norm)

plt.subplot(1, 2, 2)
sns.distplot(np.log(train.SalePrice + 1), kde=False, fit = norm)
plt.xlabel('Log SalePrice')
train["SalePrice"] = np.log1p(train["SalePrice"])
full = full.drop(['SalePrice'],axis=1)
# use robustscaler since maybe there are other outliers.
scaler = RobustScaler()
n_train=train.shape[0]

X = full[:n_train]
test_X = full[n_train:]
y= train.SalePrice

X_scaled = scaler.fit(X).transform(X)
test_X_scaled = scaler.transform(test_X)
print(X.shape)
print(test_X.shape)
print (y.shape)
#Validation function
n_folds = 5
def rmse_cv(model,X,y):
    kf = KFold(n_folds, shuffle=True, random_state=40).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))
    return rmse
models = [LinearRegression(),Ridge(),Lasso(alpha=0.01,max_iter=10000),DecisionTreeRegressor(random_state=1),RandomForestRegressor(),GradientBoostingRegressor(),SVR(),LinearSVR(),
          ElasticNet(alpha=0.001,max_iter=10000),SGDRegressor(max_iter=1000,tol=1e-3),BayesianRidge(),KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
          ExtraTreesRegressor(),XGBRegressor()]
names = ["LR", "Ridge", "Lasso", "DT","RF", "GBR", "SVR", "LinSVR", "Ela","SGD","Bay","Ker","Extra","Xgb"]
for name, model in zip(names, models):
    score = rmse_cv(model, X_scaled, y)
    print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))
class grid():
    def __init__(self,model):
        self.model = model
    
    def grid_get(self,X,y,param_grid):
        grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X,y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])

grid(Lasso()).grid_get(X_scaled,y,{'alpha': [.0001, .0003, .0004,.0005, .0006,.0007, .0009, 
          .01, 0.05, 0.1]})
grid(Ridge()).grid_get(X_scaled,y,{'alpha':[.0001, .0003, .0005, .0007, .0009, 
          .01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 20, 30, 50, 60, 70, 80]})
grid(Ridge()).grid_get(X_scaled,y,{'alpha':[13,13.1,13.2,13.3,13.4,13.5,13.6,13.7]})
grid(ElasticNet()).grid_get(X_scaled,y,{'alpha':[0.005,0.0007,0.0008,0.0009],'l1_ratio':[0.5,0.6,0.7,0.8]})
grid(SVR()).grid_get(X_scaled,y,{'C':[11,13,15],'kernel':["rbf"],"gamma":[0.0003,0.0004],"epsilon":[0.008,0.009]})
param_grid={'alpha':[0.1,0.15,0.2,0.25,0.3,0.4], 'kernel':["polynomial"], 'degree':[1,2,3,4],'coef0':[0.8,1]}
grid(KernelRidge()).grid_get(X_scaled,y,param_grid)
import xgboost as xgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
score = rmse_cv(model_xgb,X_scaled,y)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
import lightgbm as lgb
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmse_cv(model_lgb,X_scaled,y)
print("Lgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
class AverageWeight(BaseEstimator, RegressorMixin):
    def __init__(self,mod,weight):
        self.mod = mod
        self.weight = weight
        
    def fit(self,X,y):
        self.models_ = [clone(x) for x in self.mod]
        for model in self.models_:
            model.fit(X,y)
        return self
    
    def predict(self,X):
        w = list()
        pred = np.array([model.predict(X) for model in self.models_])
        # for every data point, single model prediction times weight, then add them together
        for data in range(pred.shape[1]):
            single = [pred[model,data]*weight for model,weight in zip(range(pred.shape[0]),self.weight)]
            w.append(np.sum(single))
        return w
lasso = Lasso(alpha=0.0005,max_iter=10000)
ridge = Ridge(alpha=13.1)
svr = SVR(gamma= 0.0004,kernel='rbf',C=15,epsilon=0.008)
ker = KernelRidge(alpha=0.1 ,kernel='polynomial',degree=2, coef0=1)
ela = ElasticNet(alpha=0.0007,l1_ratio=0.7,max_iter=10000)
bay = BayesianRidge()
# assign weights based on their gridsearch score
w1 = 0.3
w2 = 0.05
w3 = 0.15
w4 = 0.1
w5 = 0.3
w6 = 0.1
weight_avg = AverageWeight(mod = [lasso,ridge,svr,ker,ela,bay],weight=[w1,w2,w3,w4,w5,w6])
score = rmse_cv(weight_avg,X_scaled,y)
print(score.mean())
a = Imputer().fit_transform(X_scaled)
b = Imputer().fit_transform(y.values.reshape(-1,1)).ravel()
from mlxtend.regressor import StackingCVRegressor
stack_gen = StackingCVRegressor(regressors=(ridge,ela, 
                                            ker, svr,bay), 
                               meta_regressor=lasso,
                               use_features_in_secondary=False)
stack_gen_model = stack_gen.fit(a, b)
score = rmse_cv(stack_gen_model,a,b)
print(score.mean())
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
stack_train_pred = stack_gen_model.predict(X_scaled)
c = Imputer().fit_transform(test_X_scaled)
stack_pred = np.expm1(stack_gen_model.predict(c))
print(rmsle(stack_train_pred, y))
model_xgb.fit(X_scaled, y)
xgb_train_pred = model_xgb.predict(X_scaled)
xgb_pred = np.expm1(model_xgb.predict(test_X_scaled))
print(rmsle(xgb_train_pred, y))
model_lgb.fit(X_scaled, y)
lgb_train_pred = model_lgb.predict(X_scaled)
lgb_pred = np.expm1(model_lgb.predict(test_X_scaled))
print(rmsle(lgb_train_pred, y))
ensemble_train = stack_train_pred*0.7 + xgb_train_pred*0.15 + lgb_train_pred*0.15
print(rmsle(ensemble_train, y))
ensemble = stack_pred*0.8 + xgb_pred*0.1 + lgb_pred*0.1
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': ensemble})
my_submission.to_csv('submission.csv', index=False)



