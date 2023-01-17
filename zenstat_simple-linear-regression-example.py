import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# list all directory 

import os

print(os.listdir("../input"))
#load train and test 

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
#check sample data 

train.head(10)
#get training data set dimensions

print("train data shape", train.shape)

#get testing data set dimensions

print("test data shape", test.shape)
#list of columns 

print("\ncolumn in training data set\n\n",train.columns.values)

print("\ncolumn in testing data set\n\n",test.columns.values)
print("extra columns found in training dataset", set(train.columns.values)-set(test.columns.values))
#check  summary of dependent variable

train.SalePrice.describe()


import matplotlib.pyplot as plt 

plt.style.use(style='ggplot')

plt.rcParams['figure.figsize']=(10,6)

#check skewness of sale price 

print("Skewness : ", train.SalePrice.skew())

plt.hist(train.SalePrice, color='blue')

plt.show()

#distribution is a bit skewed to the left 
#change sale price to natural log 

print("Skewness after log: ", np.log(train.SalePrice).skew())

plt.hist(np.log(train.SalePrice), color='blue')

plt.show()

target= np.log(train.SalePrice)

#natural log transformation of target var changes it to normal distribution 
#find numeric features 

num_features = train.select_dtypes(include=[np.number])

#check data types of these 

num_features.dtypes
#check correlation of numeric variables 

corr = num_features.corr()

#top 5 highly correlated vars 

print(corr['SalePrice'].sort_values(ascending=False)[:5],'\n')

#bottom 5 highly correlated vars 

print(corr['SalePrice'].sort_values(ascending=False)[-5:],'\n')
#check unique values of feature OverallQual

train.OverallQual.unique()
#check first overall quality variable with SalePirce

qual_pivot = train.pivot_table(index='OverallQual', 

                               values='SalePrice', 

                               aggfunc=np.mean)

display(qual_pivot)
#create pivot for overall quality 

qual_pivot.plot(kind='bar', color='green')

plt.xlabel('Overall Quality')

plt.ylabel('Mean Sale Price')

plt.xticks(rotation=0)

plt.show()
#create pivot for Gr Living area 

plt.scatter(x=train['GrLivArea'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('Above grade(ground) living area square feet')

plt.show()
plt.scatter(x=train['GarageArea'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()
train = train[train['GarageArea'] < 1150]

plt.scatter(x=train['GarageArea'], y =np.log(train.SalePrice))

plt.xlim(-200,1600) #adjusting to same scale 

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()
nuls = pd.DataFrame(train.isnull().sum().sort_values(ascending =False)[:25])

nuls.columns = ['Null Count']

nuls.index.name = 'Feature'

nuls
print ("Unique values are:", train.MiscFeature.unique())
catgr  = train.select_dtypes(exclude=[np.number])

catgr.describe()
print('originals')

print(train.Street.value_counts(),"\n")
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)

test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
print("Encoded:")

print(train.enc_street.value_counts())
# One more variable, Garage car capacity

train.GarageCars.value_counts().plot(kind='bar', color='green')

plt.xlabel('Garage Car Capacity')

plt.ylabel('Counts')

plt.xticks(rotation=0)

plt.show()
condition_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)

condition_pivot.plot(kind='bar', color='skyblue')

plt.xlabel('Sale Condition')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()

#encoding steps

def encode_condition(x) : 

    return 1 if x =='Partial' else 0

train['enc_condition'] = train.SaleCondition.apply(encode_condition)

test['enc_condition'] = test.SaleCondition.apply(encode_condition)
condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)

condition_pivot.plot(kind='bar', color='gray')

plt.xlabel('Encoded Sale Condition')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()
#update missing values 

train = train.fillna(train.mean())

test = test.fillna(test.mean())
#interpolate missing values 

dt = train.select_dtypes(include=[np.number]).interpolate().dropna()

#check if all cols have zero null values 

sum(dt.isnull().sum()!=0)
#change y to natural log 

y = np.log(train.SalePrice)

#drop original dependent var and id 

X = dt.drop(['Id','SalePrice'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
#Hyper paramteter tuning  example 

import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

#Hyper parameter tuning example 

gbm = xgb.XGBRegressor()

reg_cv = GridSearchCV(gbm, {"colsample_bytree":[1.0],"min_child_weight":[1.0,1.2]

                            ,'max_depth': [3,4,6], 'n_estimators': [500,1000]}, verbose=1)

reg_cv.fit(X_train,y_train)

reg_cv.best_params_

###########

gbm = xgb.XGBRegressor(**reg_cv.best_params_)

gbm.fit(X_train,y_train)

##############

submit= pd.DataFrame()

submit['Id'] = test.Id

test_features = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()

preds = gbm.predict(test_features)

final_preds = np.exp(preds)

print('Original preds :\t', preds[:5])

print('Final preds :\t', final_preds[:5])

submit['SalePrice'] = final_preds

#final submission  

submit.to_csv('xgb_hyper_param_subm.csv', index=False)

print('XGB submission using hyper param tuning code  created')
#1. linear regression 

from sklearn import linear_model

lr = linear_model.LinearRegression()

model = lr.fit(X_train, y_train)

#r square 

print("R-Square : " ,model.score(X_test,y_test))

#rmse 

preds = model.predict(X_test)

from sklearn.metrics import mean_squared_error

print ('RMSE: ', mean_squared_error(y_test, preds))
#Adding simple XGB output and test 

import warnings

warnings.filterwarnings("ignore")

from xgboost import XGBRegressor

#

def xgb_regressor(learn_rate):

    #instance of XGB regressor 

    xgbmodel = XGBRegressor(n_estimators=1000, learning_rate=learn_rate)

    xgbmodel.fit(X_train, y_train, verbose=False)

    # make predictions

    predictions = xgbmodel.predict(X_test)

    from sklearn.metrics import mean_absolute_error

    print('{:^20}'.format('Learning Rate:')+ '{:^5}'.format(str(learn_rate)) +'{:^5}'.format("\tMAE: ")+'{:<20}'.format(str(mean_absolute_error( y_test,predictions))) +'{:^5}'.format("\tRMSE: ")+'{:<20}'.format(str(mean_squared_error( y_test,predictions)) +'{:^5}'.format("\tR^2: ")+'{:<20}'.format(xgbmodel.score(X_test,y_test)))  )

    

xgb_regressor(0.04) #experimented with .03 -.09, .04 looks best 



#using best learning rate xgb_regressor(0.04) and updating same code for submission 

def xgb_regressor_updated(learn_rate):

    #instance of XGB regressor 

    print('***********Final run with best learning rate*************')

    xgbmodel = XGBRegressor(n_estimators=1000, learning_rate=learn_rate)

    xgbmodel.fit(X_train, y_train, verbose=False)

    # make predictions

    predictions = xgbmodel.predict(X_test)

    from sklearn.metrics import mean_absolute_error

    print('{:^20}'.format('Learning Rate:')+ '{:^5}'.format(str(learn_rate)) +'{:^5}'.format("\tMAE: ")+'{:<20}'.format(str(mean_absolute_error( y_test,predictions))) +'{:^5}'.format("\tRMSE: ")+'{:<20}'.format(str(mean_squared_error( y_test,predictions)) +'{:^5}'.format("\tR^2: ")+'{:<20}'.format(xgbmodel.score(X_test,y_test)))  )

    #test 

    submit= pd.DataFrame()

    submit['Id'] = test.Id

    test_features = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()

    preds = xgbmodel.predict(test_features)

    final_preds = np.exp(preds)

    print('Original preds :\t', preds[:5])

    print('Final preds :\t', final_preds[:5])

    submit['SalePrice'] = final_preds

    #final submission  

    submit.to_csv('xgb_submit.csv', index=False)

    print('XGB submission file created')



#test and create xgb submission 

xgb_regressor_updated(0.04)
plt.scatter(preds, y_test, alpha=.75, color='g')

plt.xlabel('predicted price')

plt.ylabel('actual sale price ')

plt.title('Linear regression ')

plt.show()
#Regularization 

for i in range (-3, 3):

    alpha = 10**i

    rm = linear_model.Ridge(alpha=alpha)

    ridge_model = rm.fit(X_train, y_train)

    preds_ridge = ridge_model.predict(X_test)

    plt.scatter(preds_ridge, y_test, alpha=.75, color='g')

    plt.xlabel('Predicted Price')

    plt.ylabel('Actual Price')

    plt.title('Ridge Regularization with alpha = {}'.format(alpha))

    overlay = 'R^2 is: {}\nRMSE is: {}'.format(ridge_model.score(X_test, y_test),

                                               mean_squared_error(y_test, preds_ridge))

    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')

    plt.show()
submit= pd.DataFrame()

submit['Id'] = test.Id

#select features 

test_features = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()

preds = model.predict(test_features)

#unlog/exp the prediction  

final_preds = np.exp(preds)

print('Original preds :\n', preds[:5])

print('Final preds :\n', final_preds[:5])

submit['SalePrice'] = final_preds

#final submission  

submit.to_csv('test_submit.csv', index=False)
#variables not used 

print("Vars not used : \n", set(test.columns.values)-set(X.columns.values))