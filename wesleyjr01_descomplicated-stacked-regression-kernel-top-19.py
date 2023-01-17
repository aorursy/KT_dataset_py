#Import some usefull libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from keras import regularizers
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import math
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))
#Importing the dataset to train our models, and the dataset to make predictions
df_train = pd.read_csv('../input/train.csv')
df_pred = pd.read_csv('../input/test.csv')
#DATA INFORMATION - Lets see how the training data looks like
df_train.head(3)
#DATA INFORMATION - Lets see how the prediction data looks like
df_pred.head(3)
#I will define here some variables to use in the course of work.
copy_df_train = df_train.copy()
copy_df_pred = df_pred.copy()
target = 'SalePrice'
identifier = 'Id'
test_size = 0.3
feature_transform = 1 #Flag variable
Norm_Features = 1 #Flag variable
#X_train : Will be the train data Matrix used to train our models
#X_teste = Will be the test data Matrix used to test our models
#X_pred = Will be the data Matrix used to make predicition and write them down to csv.
### Outliers ###
var = 'GrLivArea'
data = pd.concat([df_train[target], df_train[var]], axis=1)
data.plot.scatter(x=var, y=target, ylim=(0,800000));

#It looks like there are two point pretty off the curve SalePrice/GrLivArea, lets remove them
df_train = df_train.drop(df_train.index[df_train.sort_values(by=['GrLivArea'],ascending=False)[:2].index])
#Lets take a look at it without these outliers:
var = 'GrLivArea'
data = pd.concat([df_train[target], df_train[var]], axis=1)
data.plot.scatter(x=var, y=target, ylim=(0,800000));
###Merge Datasets Before Preprocessing###
#Define some usefull variables
train_Id = df_train[identifier]
pred_Id = df_pred[identifier]
train_target = df_train[target]
# Merge Train and Prediction Dataset into df_merge. Drop 'Id' and 'SalePrice' columns before merge
df_merge = pd.concat([df_train.drop([target,identifier],1),df_pred.drop(identifier,1)],axis=0)
df_merge.info()

#Take a look of the missing variables values of the df_merge DataFrame
total = df_merge.isnull().sum().sort_values(ascending=False)
percent = (df_merge.isnull().sum()/df_merge.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(40)
###Dealing with missing data###
#One approach to deal with missing data is to drop the colums with a lot of data 
#missing, wich in our case is PoolQC, MiscFeature, Alley and Fence, and then drop
#some exmaple lines of the rest NaN's values. (which I dont like very much, we might miss a lot of information by doing that)

#Another approach is to perform Imputation into the numerical features, 
#and get_dummies into the non-numeric variables.

# Separate object and non-object features
qualitative_features = [f for f in df_merge.dropna().columns if df_merge.dropna().dtypes[f] == 'object']
quantitative_features = [f for f in df_merge.dropna().columns if df_merge.dropna().dtypes[f] != 'object']
    
############# NUMERIC FEATURES #############
print('\nIs there any NaN value of numeric features in the dataset before Imputing?:',df_merge[quantitative_features].isnull().sum().any())
df_merge[quantitative_features].isnull().sum().sort_values(ascending=False)
df_merge['LotFrontage'] = df_merge['LotFrontage'].fillna(0)
df_merge['GarageYrBlt'] = df_merge['GarageYrBlt'].fillna(0)
df_merge['MasVnrArea'] = df_merge['MasVnrArea'].fillna(0)

for i in quantitative_features:
    df_merge[i] = df_merge[i].fillna(df_merge[i].mean())
    
## Normalization of Numeric Features ###
df_merge[quantitative_features] = (df_merge[quantitative_features]-df_merge[quantitative_features].min())/(
        df_merge[quantitative_features].max()-df_merge[quantitative_features].min())
## Normalization of Numeric Features ###

print('\nIs there any NaN value of numeric features in the dataset after Imputing?:',df_merge[quantitative_features].isnull().sum().any())
############# NUMERIC FEATURES #############

############# NON-NUMERIC FEATURES #############
print('\nIs there any NaN value of non-numeric features in the dataset before Imputing?:',df_merge[qualitative_features].isnull().sum().any())
df_merge[qualitative_features].isnull().sum().sort_values(ascending=False)
df_merge['PoolQC'] = df_merge['PoolQC'].fillna('None')
df_merge['MiscFeature'] = df_merge['MiscFeature'].fillna('None')
df_merge['Alley'] = df_merge['Alley'].fillna('None')
df_merge['BsmtQual'] = df_merge['BsmtQual'].fillna('None')
df_merge['BsmtCond'] = df_merge['BsmtCond'].fillna('None')
df_merge['BsmtExposure'] = df_merge['BsmtCond'].fillna('None')
df_merge['BsmtFinType1'] = df_merge['BsmtFinType1'].fillna('None')
df_merge['BsmtFinType2'] = df_merge['BsmtFinType2'].fillna('None')
df_merge['FireplaceQu'] = df_merge['FireplaceQu'].fillna('None')
df_merge['GarageType'] = df_merge['GarageType'].fillna('None')
df_merge['GarageFinish'] = df_merge['GarageFinish'].fillna('None')
df_merge['GarageCond'] = df_merge['GarageCond'].fillna('None')
df_merge['GarageQual'] = df_merge['GarageQual'].fillna('None')
df_merge['Fence'] = df_merge['Fence'].fillna('None')
df_merge['MasVnrType'] = df_merge['MasVnrType'].fillna('None')
df_merge["Functional"] = df_merge["Functional"].fillna("Typ")
df_merge.drop('Utilities',axis=1,inplace=True)
last_qlfeat = ['MSZoning','KitchenQual','Exterior2nd','Exterior1st','Electrical','SaleType']
qualitative_features = [f for f in df_merge.dropna().columns if df_merge.dropna().dtypes[f] == 'object'] #List of Non-Numeric Features
quantitative_features = [f for f in df_merge.dropna().columns if df_merge.dropna().dtypes[f] != 'object'] #List of Numeric Features

for i in last_qlfeat:# Mode for the rest of non-numeric features
    df_merge[i] =  df_merge[i].fillna(df_merge[i].mode()[0])

print('\nIs there any NaN value  in the dataset after Imputing and get_dummies?:',df_merge.isnull().sum().any())
############# NON-NUMERIC FEATURES #############

### Labeling of Categorical Variables With Dummies
df_merge = pd.get_dummies(data=df_merge, columns=qualitative_features)
#### FEATURE TRANSFORM - LOG1
#There are some skewed variables which we can perform some log transformation
sns.distplot(df_train[target]) #SalePrice Distribution Plot
#SalePrice looks like this before log transformation
### FEATURE TRANSFORM - LOG1 of SalePrice ###
if feature_transform:
    train_target = np.log1p(train_target)


#SalePrice look like that after log transformation, much closer to a Normal Distr.
sns.distplot(np.log1p(df_train[target]))
# Restore datraframes df_train and df_pred
df_train = df_merge[:len(df_train)]
df_train[target] = train_target
df_pred = df_merge[len(df_train):]
df_train.info()
# Correlation Matrix to get a feeling of the correlations between variables on the dataset
corrmat = df_train.corr()
k = 15 #number of variables for heatmap
cols = corrmat.nlargest(k, target)[target].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(12, 9))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
### Division between X_train,X_test,y_train,y_test
df_train = shuffle(df_train) #shuffle data before division
train_target = df_train[target] # Just for code readibility
predictors = df_train.drop([target], axis=1)
X_train, X_test, y_train, y_test = train_test_split(predictors, 
                                                    train_target,
                                                    train_size=1-test_size, 
                                                    test_size=test_size, 
                                                random_state=0)
X_pred = df_pred
### Lasso Model ###
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

lasso = make_pipeline(RobustScaler(),LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1, 3, 6, 10, 30, 60, 100], 
                max_iter = 50000, cv = 10))
lasso.fit(X_train, y_train)

y_train_las = lasso.predict(X_train)
y_test_las = lasso.predict(X_test)
las_prediction = lasso.predict(X_pred)
### Ridge Model ###
#RobustScaler() to make the model robust against outliers
ridge = make_pipeline(RobustScaler(),RidgeCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1, 3, 6, 10, 30, 60, 100]))
ridge.fit(X_train, y_train)

y_train_rdg = ridge.predict(X_train)
y_test_rdg = ridge.predict(X_test)
rdg_prediction = ridge.predict(X_pred)
### XGBoost Model through GridSearch ###
xgb_model = XGBRegressor()
params = {'n_estimators':[1000],'learning_rate':[0.1,0.05],
'max_depth':[3,4,5]}
grid = GridSearchCV(xgb_model, params)
grid.fit(X_train, y_train)
y_train_gridXGB = grid.best_estimator_.predict(X_train)
y_test_gridXGB = grid.best_estimator_.predict(X_test)
gridXGB_prediction = grid.best_estimator_.predict(X_pred)
### XGBoost Model without GridSearch ###
xgb_model = XGBRegressor(n_estimators=10000, learning_rate=0.05)

xgb_model.fit(X_train, y_train, early_stopping_rounds=5, 
             eval_set=[(X_test, y_test)], verbose=False)
  
y_train_xgb = xgb_model.predict(X_train)
y_test_xgb = xgb_model.predict(X_test)
xgb_prediction = xgb_model.predict(X_pred)
### ElasticNetRegression ###
from sklearn.linear_model import ElasticNetCV, ElasticNet
#ElasticNet Regressor
elnr = make_pipeline(RobustScaler(),ElasticNetCV(l1_ratio=[0.2,0.65],alphas=[0.001,0.005],n_jobs=-1,max_iter=3000))

elnr.fit(X_train,y_train)

y_train_eln = elnr.predict(X_train)
y_test_eln = elnr.predict(X_test)
eln_prediction = elnr.predict(X_pred)
### GradientBoost ###
from sklearn.ensemble import GradientBoostingRegressor
#With huber loss that makes it robust to outliers
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                    min_samples_split=10, 
                                   loss='huber', random_state =5)

GBoost.fit(X_train,y_train)

GBoost_train = GBoost.predict(X_train)
GBoost_test = GBoost.predict(X_test)
GBoost_prediction = GBoost.predict(X_pred)
### KernelRidge ###
from sklearn.kernel_ridge import KernelRidge
KR = KernelRidge(alpha=1.0)
KR.fit(X_train, y_train) 

y_train_KR = KR.predict(X_train)
y_test_KR = KR.predict(X_test)
KR_prediction = KR.predict(X_pred)
### LightGBM ###
import lightgbm as lgb
# other scikit-learn modules
estimator = lgb.LGBMRegressor(num_leaves=31)

param_grid = {
    'learning_rate': [0.001,0.005, 0.01],
    'n_estimators': [100,1000]}

gbm = GridSearchCV(estimator, param_grid)
gbm.fit(X_train, y_train)

y_train_gbm = gbm.predict(X_train)
y_test_gbm = gbm.predict(X_test)
gbm_prediction = gbm.predict(X_pred)
### Results Compilation and Plots ###
if feature_transform:
    y_train = np.expm1(y_train)
    y_test = np.expm1(y_test)

    
    y_train_las = np.expm1(y_train_las)
    y_test_las = np.expm1(y_test_las)
    las_prediction = np.expm1(las_prediction)
    
    y_train_rdg = np.expm1(y_train_rdg)
    y_test_rdg = np.expm1(y_test_rdg)
    rdg_prediction = np.expm1(rdg_prediction)
    
    y_train_xgb = np.expm1(y_train_xgb)
    y_test_xgb = np.expm1(y_test_xgb)
    xgb_prediction = np.expm1(xgb_prediction)
    
    y_train_gridXGB = np.expm1(y_train_gridXGB)
    y_test_gridXGB = np.expm1(y_test_gridXGB)
    gridXGB_prediction = np.expm1(gridXGB_prediction)

    
    y_train_eln = np.expm1(y_train_eln)
    y_test_eln = np.expm1(y_test_eln)
    eln_prediction = np.expm1(eln_prediction)
    
    GBoost_train = np.expm1(GBoost_train)
    GBoost_test = np.expm1(GBoost_test)
    GBoost_prediction = np.expm1(GBoost_prediction)
    
    y_train_KR = np.expm1(y_train_KR)
    y_test_KR = np.expm1(y_test_KR)
    KR_prediction = np.expm1(KR_prediction)
    
    y_train_gbm = np.expm1(y_train_gbm)
    y_test_gbm = np.expm1(y_test_gbm)
    gbm_prediction = np.expm1(gbm_prediction)
else:
    pass
feature_transform=0


df_resultado_treinamento = pd.DataFrame(
        {'x_train':list(X_train.values),
         'y_train':list(y_train),
         'y_train_lasso':y_train_las,
         'y_train_ridge':y_train_rdg,
         'y_train_xgb':y_train_xgb,
         'y_train_gridXGB':y_train_gridXGB,
         'y_train_eln':y_train_eln,
         'GBoost_train':GBoost_train,
         'y_train_KR':y_train_KR,
         'y_train_gbm':y_train_gbm,
                })
    
df_resultado_val = pd.DataFrame(
        {'x_val':list(X_test.values),
         'y_val':list(y_test),  
         'y_test_lasso':y_test_las,
         'y_test_ridge':y_test_rdg,
         'y_test_xgb':y_test_xgb,
         'y_test_gridXGB':y_test_gridXGB,
         'y_test_eln':y_test_eln,
         'GBoost_test':GBoost_test,
         'y_test_KR':y_test_KR,
         'y_test_gbm':y_test_gbm,
                })
    

plt.figure(1)

df_resultado_treinamento = df_resultado_treinamento.sort_values('y_train',ascending=1)
plt.plot(list(df_resultado_treinamento['y_train']),'r-.',label='y_train')
plt.plot(list(df_resultado_treinamento['y_train_lasso']),'g-.',label='LassoTrain')
plt.plot(list(df_resultado_treinamento['y_train_ridge']),'y-.',label='RidgeTrain')
plt.plot(list(df_resultado_treinamento['y_train_xgb']),'k-.',label='XGB')
plt.legend()


plt.figure(2)
df_resultado_val = df_resultado_val.sort_values('y_val',ascending=1)
plt.plot(list(df_resultado_val['y_val']),'r-.',label='y_test')
plt.plot(list(df_resultado_val['y_test_lasso']),'g-.',label='LassoTest')
plt.plot(list(df_resultado_val['y_test_ridge']),'y-.',label='RidgeTest')
plt.plot(list(df_resultado_val['y_test_xgb']),'k-.',label='XGB')
plt.legend()
plt.show()


## Models Eval
#### Root Mean Square Logarithmic Error ###
def logRMSE(y_test, y_pred) : 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))

print("\nLasso logRMSE on Training set :", logRMSE(y_train,y_train_las))
print("Ridge logRMSE on Training set :", logRMSE(y_train,y_train_rdg))
print("XGB logRMSE on Training set :", logRMSE(y_train,y_train_xgb))
print("gridXGB logRMSE on Training set :", logRMSE(y_train,y_train_gridXGB))



print("\nLasso logRMSE on Test set :", logRMSE(y_test,y_test_las))
print("Ridge logRMSE on Test set :", logRMSE(y_test,y_test_rdg))
print("XGB logRMSE on Test set :", logRMSE(y_test,y_test_xgb))
print("gridXGB logRMSE on Training set :", logRMSE(y_test,y_test_gridXGB))
print("eln logRMSE on Test set :", logRMSE(y_test,y_test_eln))
print("GBoost logRMSE on Test set :", logRMSE(y_test,GBoost_test))
print("KR logRMSE on Test set :", logRMSE(y_test,y_test_KR))
print("LightGBM logRMSE on Test set :", logRMSE(y_test,y_test_gbm))



print("StackedModel logRMSE on Test set :", logRMSE(y_test,(y_test_xgb+y_test_rdg+y_test_las + y_test_gridXGB
                                                            +y_test_eln + GBoost_test +y_test_KR +y_test_gbm)/(len(df_resultado_val.columns)-2)))
######################################### Results Compilation and Plots #################################################  


#########################################################################################################################
# Writing dataset for submission
df_out = pd.DataFrame(
        {identifier:pred_Id,
        target: (xgb_prediction + las_prediction + rdg_prediction + gridXGB_prediction
                   +eln_prediction + GBoost_prediction +KR_prediction +gbm_prediction )/(len(df_resultado_val.columns)-2)
                })

df_out = pd.DataFrame(
        {identifier:pred_Id,
        target: (xgb_prediction + las_prediction + rdg_prediction + gridXGB_prediction
                   +eln_prediction + GBoost_prediction +KR_prediction +gbm_prediction )/(len(df_resultado_val.columns)-2)
                })
df_out.to_csv('House_Prices_submition.csv', sep=',', index = False)