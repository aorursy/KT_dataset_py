import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_squared_error, accuracy_score, r2_score

from sklearn.impute import SimpleImputer



pd.set_option('display.max_columns',100)

pd.set_option('display.max_rows',100)
home_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',index_col=0)

home_data
print(home_data.info())
home_data.shape
home_data.describe().round(3)
target_var_name = 'SalePrice'

target_var = pd.DataFrame(home_data[target_var_name]).set_index(home_data.index)

home_data.drop(target_var_name, axis=1, inplace=True)

target_var
print(target_var.describe().round(decimals=2))

sns.distplot(target_var)

plt.title('Distribution of SalePrice')

plt.show()
num_feature = home_data.select_dtypes(exclude=['object']).columns

home_data_num_feature = home_data[num_feature].set_index(home_data.index)
home_data_num_feature.describe().round(3)
fig = plt.figure(figsize=(12,20))

plt.title('Numerical Feature (before dropping identified outliers)')

for i in range(len(home_data_num_feature.columns)):

    fig.add_subplot(9,4,i+1)

    sns.distplot(home_data_num_feature.iloc[:,i].dropna(),kde_kws={'bw':0.1})

    plt.xlabel(home_data_num_feature.columns[i])



plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(12,20))

plt.title('Numerical Feature (before dropping identified outliers)')

for i in range(len(home_data_num_feature.columns)):

    fig.add_subplot(9,4,i+1)

    sns.scatterplot(home_data_num_feature.iloc[:,i], target_var.iloc[:,0])

    plt.xlabel(home_data_num_feature.columns[i])



plt.tight_layout()

plt.show()
correlation = home_data_num_feature.corr()



f, ax = plt.subplots(figsize=(14,12))

plt.title('Correlation of numerical attributes', size=16)

sns.heatmap(correlation>0.8)

plt.show()

y_corr = pd.DataFrame(home_data_num_feature.corrwith(target_var.SalePrice),columns=["Correlation with target variable"])

# plt.hist(y_corr)
y_corr_sorted= y_corr.sort_values(by=['Correlation with target variable'],ascending=False)

y_corr_sorted
fig = plt.figure(figsize=(6,10))

plt.title('Correlation with target variable')

a=sns.barplot(y_corr_sorted.index,y_corr_sorted.iloc[:,0],data=y_corr)

a.set_xticklabels(labels=y_corr_sorted.index,rotation=90)

plt.tight_layout()

plt.show()
[(y_corr_sorted<0.1) & (y_corr_sorted>-0.1)]
cat_feature = home_data.select_dtypes(include=['object']).columns

home_data_cat_feature = home_data[cat_feature]
fig = plt.figure(figsize=(18,50))

plt.title('Distribution of Categorical Feature')

for i in range(len(home_data_cat_feature.columns)):

    fig.add_subplot(15,3,i+1)

    sns.countplot(home_data_cat_feature.iloc[:,i])

    plt.xlabel(home_data_cat_feature.columns[i])

    plt.xticks(rotation=90)



plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(18,80))

plt.title('Numerical Feature (before dropping identified outliers)')

for i in range(len(home_data_cat_feature.columns)):

    fig.add_subplot(15,3,i+1)

    sns.boxplot(x=home_data_cat_feature.iloc[:,i],y=target_var['SalePrice'])

    plt.xlabel(home_data_cat_feature.columns[i])

    plt.xticks(rotation=90)



plt.tight_layout()

plt.show()
##look into sorting by median
home_data_num_feature=home_data_num_feature.drop(['GarageYrBlt','1stFlrSF','TotRmsAbvGrd','GarageArea'],axis=1)

home_data_num_feature.columns
home_data_num_feature=home_data_num_feature.drop(['PoolArea','MoSold','3SsnPorch','BsmtFinSF2','BsmtHalfBath','MiscVal','LowQualFinSF','YrSold','OverallCond','MSSubClass'],axis=1)
home_data_num_feature=home_data_num_feature.drop(home_data_num_feature[home_data_num_feature['LotFrontage']>300].index)

print(len(home_data_num_feature))

home_data_num_feature=home_data_num_feature.drop(home_data_num_feature[(home_data_num_feature['GrLivArea']>4000) & (target_var['SalePrice']<300000)].index)

print(len(home_data_num_feature))



home_data_num_feature=home_data_num_feature.drop(home_data_num_feature[home_data_num_feature['BsmtFinSF1']>4000].index)

print(len(home_data_num_feature))



home_data_num_feature=home_data_num_feature.drop(home_data_num_feature[home_data_num_feature['LotArea']>100000].index)

print(len(home_data_num_feature))



home_data_num_feature=home_data_num_feature.drop(home_data_num_feature[home_data_num_feature['TotalBsmtSF']>6000].index)

print(len(home_data_num_feature))
fig = plt.figure(figsize=(12,12))

plt.title('Numerical Feature (after dropping identified outliers)')

for i in range(len(home_data_num_feature.columns)):

    fig.add_subplot(6,4,i+1)

    sns.scatterplot(home_data_num_feature.iloc[:,i], target_var.iloc[:,0])

    plt.xlabel(home_data_num_feature.columns[i])



plt.tight_layout()

plt.show()
home_data_num_feature.count()
from sklearn.impute import SimpleImputer

imp = SimpleImputer()

home_data_num_feature = pd.DataFrame(imp.fit_transform(home_data_num_feature),columns=home_data_num_feature.columns,index=home_data_num_feature.index)

home_data_num_feature.count()
home_data_cat_feature.count()
home_data_cat_feature=home_data_cat_feature.drop(['Alley','PoolQC','Fence','MiscFeature','FireplaceQu'],axis=1)
imp = SimpleImputer(strategy="most_frequent")

home_data_cat_feature=pd.DataFrame(imp.fit_transform(home_data_cat_feature),columns=home_data_cat_feature.columns,index=home_data_cat_feature.index)

home_data_cat_feature
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(drop='first',sparse=False)

enc.fit(home_data_cat_feature)

home_data_cat_feature_dummies=enc.transform(home_data_cat_feature)

home_data_cat_feature_dummies = pd.DataFrame(home_data_cat_feature_dummies,columns=enc.get_feature_names(),index=home_data_cat_feature.index)

home_data_cat_feature_dummies
X=pd.merge(home_data_num_feature,home_data_cat_feature_dummies,how='left',left_index=True,right_index =True)



# X=pd.concat([home_data_num_feature,home_data_cat_feature_dummies],axis=1)

y=target_var.loc[X.index]
X
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline



r2 = make_scorer(r2_score)

rmse = make_scorer(mean_squared_error,greater_is_better=False,squared=False)



cv_list={}

cv_rmse={}

cv_r2={}

cv_best_mse={}
model_name = "LinearRegression"

model=LinearRegression()



param_grid = [{model_name+'__fit_intercept':[True,False]}]
pipeline = Pipeline([(model_name, model)])





reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)

reg.fit(X,y.to_numpy())





#Record the best grid search paramters into the list.

cv_list[model_name]=reg

cv_rmse[model_name]=reg.best_score_



#print out the best param and best score

print(model_name)

print('best training param:',reg.best_params_)

print('best training score rmse', reg.best_score_)

print('\n')
from sklearn.linear_model import Lasso



model_name = "Lasso"

model=Lasso()



param_grid = [  {model_name+'__'+'alpha': [2**-5,2**-3,2**-1,2**1,2**3,2**5,2**7,2**9,2**11,2**13,2**15]}]
pipeline = Pipeline([(model_name, model)])





reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)

reg.fit(X,y.to_numpy())





#Record the best grid search paramters into the list.

cv_list[model_name]=reg

cv_rmse[model_name]=reg.best_score_



#print out the best param and best score

print(model_name)

print('best training param:',reg.best_params_)

print('best training score rmse', reg.best_score_)

print('\n')
from sklearn.linear_model import Ridge



model_name = "Ridge"

model=Ridge()



param_grid = [{model_name+'__'+'alpha': [2**-5,2**-3,2**-1,2**1,2**3,2**5,2**7,2**9,2**11,2**13,2**15]}]
pipeline = Pipeline([(model_name, model)])





reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)

reg.fit(X,y.to_numpy())





#Record the best grid search paramters into the list.

cv_list[model_name]=reg

cv_rmse[model_name]=reg.best_score_



#print out the best param and best score

print(model_name)

print('best training param:',reg.best_params_)

print('best training score rmse', reg.best_score_)

print('\n')
from sklearn.tree import DecisionTreeRegressor



model_name='DecisionTreeRegressor'

model=DecisionTreeRegressor()



param_grid = [{model_name+'__'+'splitter': ['best','random'],

              model_name+'__'+'max_depth':np.arange(1,20)

              }]
pipeline = Pipeline([(model_name, model)])





reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)

reg.fit(X,y.to_numpy())





#Record the best grid search paramters into the list.

cv_list[model_name]=reg

cv_rmse[model_name]=reg.best_score_



#print out the best param and best score

print(model_name)

print('best training param:',reg.best_params_)

print('best training score rmse', reg.best_score_)

print('\n')
from sklearn.ensemble import BaggingRegressor

from sklearn.tree import DecisionTreeRegressor



model_name = 'BaggingDecisionTreeRegressor'

model=BaggingRegressor(DecisionTreeRegressor())



param_grid = [{model_name+'__'+'base_estimator__splitter': ['best','random'],

              model_name+'__'+'base_estimator__max_depth':np.arange(1,30)

              }]
pipeline = Pipeline([(model_name, model)])





reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)

reg.fit(X,y.to_numpy().ravel())





#Record the best grid search paramters into the list.

cv_list[model_name]=reg

cv_rmse[model_name]=reg.best_score_



#print out the best param and best score

print(model_name)

print('best training param:',reg.best_params_)

print('best training score rmse', reg.best_score_)

print('\n')
from sklearn.ensemble import RandomForestRegressor



model_name='RandomForestRegressor'

model=RandomForestRegressor()



param_grid = [{model_name+'__'+'max_depth' : np.arange(1,100,2)}]
pipeline = Pipeline([(model_name, model)])





reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)

reg.fit(X,y.to_numpy().ravel())





#Record the best grid search paramters into the list.

cv_list[model_name]=reg

cv_rmse[model_name]=reg.best_score_



#print out the best param and best score

print(model_name)

print('best training param:',reg.best_params_)

print('best training score rmse', reg.best_score_)

print('\n')
from sklearn.ensemble import AdaBoostRegressor



model_name='AdaBoostRegressor'

model=AdaBoostRegressor()



param_grid = [{model_name+'__'+'learning_rate' : [0.001,0.01,0.1,1,10,100]}]
pipeline = Pipeline([(model_name, model)])





reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)

reg.fit(X,y.to_numpy().ravel())





#Record the best grid search paramters into the list.

cv_list[model_name]=reg

cv_rmse[model_name]=reg.best_score_



#print out the best param and best score

print(model_name)

print('best training param:',reg.best_params_)

print('best training score rmse', reg.best_score_)

print('\n')
from sklearn.ensemble import GradientBoostingRegressor



model_name='GradientBoostingRegressor'

model=GradientBoostingRegressor()



param_grid = [{model_name+'__'+'loss' : ['ls','lad','huber','quantile'],model_name+'__'+'learning_rate' : [0.01,0.1,1,10],model_name+'__'+'criterion':['friedman_mse', 'mse']}]

pipeline = Pipeline([(model_name, model)])





reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)

reg.fit(X,y.to_numpy().ravel())





#Record the best grid search paramters into the list.

cv_list[model_name]=reg

cv_rmse[model_name]=reg.best_score_



#print out the best param and best score

print(model_name)

print('best training param:',reg.best_params_)

print('best training score rmse', reg.best_score_)

print('\n')
from xgboost import XGBRegressor

model_name='XGBoost'

model=XGBRegressor()



param_grid = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'objective':['reg:linear'],

              'learning_rate': [.03, 0.05, .07], #so called `eta` value

              'max_depth': [3,4,5],

              'min_child_weight': [4],

              'silent': [1],

              'subsample': [0.7],

              'colsample_bytree': [0.7],

              'n_estimators': [500]}



xgb_dt=GridSearchCV(model, param_grid,n_jobs=-1,cv=5,scoring=rmse)

xgb_dt.fit(X,y)



cv_list[model_name]=xgb_dt.best_estimator_

cv_rmse[model_name]=xgb_dt.best_score_



print(xgb_dt.best_estimator_)

print(xgb_dt.best_score_)
# from sklearn.svm import SVR



# model_name = "SVR"

# model=SVR()



# param_grid = [

#   {model_name+'__'+'C': [0.1,1], model_name+'__'+'kernel': ['linear','poly','rbf','sigmoid'],

#    model_name+'__'+'gamma':['auto']

#   }]



# pipeline = Pipeline([(model_name, model)])



# reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=2)

# reg.fit(X,y.to_numpy().ravel())



# #Record the best grid search paramters into the list.

# cv_list[model_name]=reg

# cv_rmse[model_name]=reg.best_score_



# #print out the best param and best score

# print(model_name)

# print('best training param:',reg.best_params_)

# print('best training score rmse', reg.best_score_)

# print('\n')
score = abs(pd.DataFrame.from_dict(cv_rmse,orient='index',columns=['CV Score']))

score = score.sort_values('CV Score')

score
test_data_path = '../input/house-prices-advanced-regression-techniques/test.csv'

X_test_set = pd.read_csv(test_data_path,index_col=0)

X_test_set.shape
test_home_data_num_feature = X_test_set[home_data_num_feature.columns]

test_home_data_num_feature.describe().round(2)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

test_home_data_num_feature = pd.DataFrame(imp.fit_transform(test_home_data_num_feature),columns=test_home_data_num_feature.columns,index=test_home_data_num_feature.index)

# test_norm_home_data_num_feature = pd.DataFrame(scaler.fit_transform(test_home_data_num_feature),columns=test_home_data_num_feature.columns,index=test_home_data_num_feature.index)

test_home_data_num_feature
test_home_data_cat_feature = X_test_set[home_data_cat_feature.columns]

test_home_data_cat_feature.describe().round(2)



imp = SimpleImputer(strategy="most_frequent")

test_home_data_cat_feature=pd.DataFrame(imp.fit_transform(test_home_data_cat_feature),columns=test_home_data_cat_feature.columns,index=test_home_data_cat_feature.index)

test_home_data_cat_feature



test_home_data_cat_feature_dummies = enc.transform(test_home_data_cat_feature)



test_home_data_cat_feature_dummies = pd.DataFrame(test_home_data_cat_feature_dummies,columns=enc.get_feature_names(),index=test_home_data_cat_feature.index)

test_home_data_cat_feature_dummies
X_test = pd.concat([test_home_data_num_feature,test_home_data_cat_feature_dummies],axis=1)

X_test
predict=cv_list['LinearRegression'].predict(X_test)

output = pd.DataFrame({'SalePrice': predict[:,0]},index=X_test.index)

output.to_csv('LinearRegression.csv', index=True)

predict=cv_list['Lasso'].predict(X_test)

output = pd.DataFrame({'SalePrice': predict},index=X_test.index)

output.to_csv('Lasso.csv', index=True)
predict=cv_list['Ridge'].predict(X_test)

output = pd.DataFrame({'SalePrice': predict[:,0]},index=X_test.index)

output.to_csv('Ridge.csv', index=True)
predict=cv_list['DecisionTreeRegressor'].predict(X_test)

output = pd.DataFrame({'SalePrice': predict},index=X_test.index)

output.to_csv('DecisionTreeRegressor.csv', index=True)
predict=cv_list['BaggingDecisionTreeRegressor'].predict(X_test)

output = pd.DataFrame({'SalePrice': predict},index=X_test.index)

output.to_csv('BaggingDecisionTreeRegressor.csv', index=True)
predict=cv_list['RandomForestRegressor'].predict(X_test)

output = pd.DataFrame({'SalePrice': predict},index=X_test.index)

output.to_csv('RandomForestRegressor.csv', index=True)
predict=cv_list['AdaBoostRegressor'].predict(X_test)

output = pd.DataFrame({'SalePrice': predict},index=X_test.index)

output.to_csv('AdaBoostRegressor.csv', index=True)
predict=cv_list['GradientBoostingRegressor'].predict(X_test)

output = pd.DataFrame({'SalePrice': predict},index=X_test.index)

output.to_csv('GradientBoostingRegressor.csv', index=True)
predict=xgb_dt.predict(X_test)

output = pd.DataFrame({'SalePrice': predict},index=X_test.index)

output.to_csv('xgb.csv', index=True)
test_score = {'LinearRegression':17183.86239,'Lasso':16379.19466,'Ridge':16107.36134,'DecisionTreeRegressor':24232.59348,'BaggingDecisionTreeRegressor':17949.15006,'RandomForestRegressor':16163.46606,'AdaBoostRegressor':22434.87007,'GradientBoostingRegressor':15517.90164,'XGBoost':13745.37874}
test_score = pd.DataFrame.from_dict(test_score,orient='index')

score['Test Score']=test_score

score