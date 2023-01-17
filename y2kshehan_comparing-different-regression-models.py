#Importing Libraries

import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt # plotting library

import missingno as msno # plotting missing data

import seaborn as sns # plotting library

from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict

from sklearn import metrics

from sklearn.preprocessing import Imputer #for handling missing data

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor



#Importing the dataset

dataset = pd.read_csv('../input/train.csv')



#Show first 5 rows

dataset.head()

#setup  Matplotlib (magic function) . Plots will render within the notebook itself

%matplotlib inline  

dataset.hist(bins = 50 , figsize = (20,20))

plt.show()
#checking missing values by column

dataset.isnull().sum()
#Visualizing missing data

msno.matrix(df=dataset, figsize=(20,14), color=(0.5,0,0))
dataset = dataset.drop(['Id','LotFrontage','Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis=1)

#Get column count

len(dataset.columns)
dataset = dataset.dropna(thresh=70) 

#Lets visualize missing data again

msno.matrix(df=dataset, figsize=(20,14), color=(0.5,0,0))
#Separating Independent & Dependant Varibles

X = dataset.iloc[:,0:-1]

y = dataset.iloc[:,-1] #Dependant Varible (SalePrice)

X.head() #show first 5 records

y[0:5] #show first 5 records
#One Hot Encoding

X = pd.get_dummies(data = X , columns=['MSZoning' ,'Street','LotShape','LandContour','Utilities','LotConfig','LandSlope'

                                                   ,'Neighborhood','Condition1','Condition2','BldgType','HouseStyle',

                                                   'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual',

                                                   'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

                                                   'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','SaleType','SaleCondition',

                                                   'KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive'] , drop_first = True )



X.head() #show first 5 records
#Converting dataframes to numpy arrays

#X = X.values

#y = y.values
#Filling the missing data

#imputer =  Imputer(missing_values = 'NaN' , strategy = 'most_frequent' , axis = 0)

#imputer = imputer.fit(X[:,:])

#X[:,:] = imputer.transform(X[:,:])



#Alternative for Imputer class. 

X = X.fillna(X.median())
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2 , random_state = 0)
lin_reg = LinearRegression()

lin_reg.fit(X_train,y_train)



#Predicting the SalePrices using test set 

y_pred_lr = lin_reg.predict(X_test)



#Mutiple Linear Regression Accuracy with test set

accuracy_lf = metrics.r2_score(y_test, y_pred_lr)

print('Mutiple Linear Regression Accuracy: ', accuracy_lf)



#Predicting the SalePrice using cross validation (KFold method)

y_pred_kf_lr = cross_val_predict(lin_reg, X, y, cv=10 )



#Mutiple Linear Regression Accuracy with cross validation (KFold method)

accuracy_lf = metrics.r2_score(y, y_pred_kf_lr)

print('Cross-Predicted(KFold) Mutiple Linear Regression Accuracy: ', accuracy_lf)
poly_reg = PolynomialFeatures(degree = 2)

X_poly = poly_reg.fit_transform(X)

lin_reg_pl = LinearRegression()



#Predicting the SalePrice using cross validation (KFold method)

y_pred_pl = cross_val_predict(lin_reg_pl, X_poly, y, cv=10 )

#Polynominal Regression Accuracy with cross validation

accuracy_pl = metrics.r2_score(y, y_pred_pl)

print('Cross-Predicted(KFold) Polynominal Regression Accuracy: ', accuracy_pl)
dt_regressor = DecisionTreeRegressor(random_state = 0)

dt_regressor.fit(X_train,y_train)



#Predicting the SalePrices using test set 

y_pred_dt = dt_regressor.predict(X_test)



#Decision Tree Regression Accuracy with test set

print('Decision Tree Regression Accuracy: ', dt_regressor.score(X_test,y_test))



#Predicting the SalePrice using cross validation (KFold method)

y_pred_dt = cross_val_predict(dt_regressor, X, y, cv=10 )

#Decision Tree Regression Accuracy with cross validation

accuracy_dt = metrics.r2_score(y, y_pred_dt)

print('Cross-Predicted(KFold) Decision Tree Regression Accuracy: ', accuracy_dt)
rf_regressor = RandomForestRegressor(n_estimators = 300 ,  random_state = 0)

rf_regressor.fit(X_train,y_train)



#Predicting the SalePrices using test set 

y_pred_rf = rf_regressor.predict(X_test)



#Random Forest Regression Accuracy with test set

print('Random Forest Regression Accuracy: ', rf_regressor.score(X_test,y_test))



#Predicting the SalePrice using cross validation (KFold method)

y_pred_rf = cross_val_predict(rf_regressor, X, y, cv=10 )



#Random Forest Regression Accuracy with cross validation

accuracy_rf = metrics.r2_score(y, y_pred_rf)

print('Cross-Predicted(KFold) Random Forest Regression Accuracy: ', accuracy_rf)
ranking = np.argsort(-rf_regressor.feature_importances_)

f, ax = plt.subplots(figsize=(15, 100))

sns.barplot(x=rf_regressor.feature_importances_[ranking], y=X_train.columns.values[ranking], orient='h')

ax.set_xlabel("feature importance")

plt.tight_layout()

plt.show()
X_train = X_train.iloc[:,ranking[:30]]

X_test = X_test.iloc[:,ranking[:30]]

lin_reg = LinearRegression()

lin_reg.fit(X_train,y_train)



#Predicting the SalePrices using test set 

y_pred_lr = lin_reg.predict(X_test)



#Mutiple Linear Regression Accuracy with test set

accuracy_lf = metrics.r2_score(y_test, y_pred_lr)

print('Mutiple Linear Regression Accuracy: ', accuracy_lf)



#Predicting the SalePrice using cross validation (KFold method)

y_pred_kf_lr = cross_val_predict(lin_reg, X, y, cv=10 )



#Mutiple Linear Regression Accuracy with cross validation (KFold method)

accuracy_lf = metrics.r2_score(y, y_pred_kf_lr)

print('Cross-Predicted(KFold) Mutiple Linear Regression Accuracy: ', accuracy_lf)