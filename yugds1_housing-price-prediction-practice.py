# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head(5)
test.head(5)
print(len(train))
print(len(test))
train.describe()

train['SalePrice'].describe()
correlation = train.corr()
correlation = correlation['SalePrice'].sort_values(ascending=False)
correlation
train_null = pd.isnull(train).sum()
test_null = pd.isnull(test).sum()

null = pd.concat([train_null,test_null],axis=1,keys=['Training','Testing'])

null_count = null[null.sum(axis=1)>100]
null_count
meaningful_null = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]

for i in meaningful_null:
    train[i].fillna("None",inplace=True)
    test[i].fillna("None",inplace=True)
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")

train_null = pd.isnull(train).sum()
test_null = pd.isnull(test).sum()

null = pd.concat([train_null,test_null],axis=1,keys=['Training','Testing'])

null_count = null[null.sum(axis=1) > 100]
null_count
train.drop(['LotFrontage'],axis=1,inplace=True)
test.drop(['LotFrontage'],axis=1,inplace=True)

null_count = null[null.sum(axis=1) <= 100]
null_count
null_many = null[null.sum(axis=1) > 200]  #a lot of missing values
null_few = null[(null.sum(axis=1) > 0) & (null.sum(axis=1) < 200)]  #few missing values

train["GarageYrBlt"].fillna(train["GarageYrBlt"].median(), inplace=True)
test["GarageYrBlt"].fillna(test["GarageYrBlt"].median(), inplace=True)
train["MasVnrArea"].fillna(train["MasVnrArea"].median(), inplace=True)
test["MasVnrArea"].fillna(test["MasVnrArea"].median(), inplace=True)
train["MasVnrType"].fillna("None", inplace=True)
test["MasVnrType"].fillna("None", inplace=True)
types_train = train.dtypes #type of each feature in data: int, float, object
num_train = types_train[(types_train == int) | (types_train == float)] #numerical values are either type int or float
cat_train = types_train[types_train == object] #categorical values are type object

#we do the same for the test set
types_test = test.dtypes
num_test = types_test[(types_test == int) | (types_test == float)]
cat_test = types_test[types_test == object]
numerical_values_train = list(num_train.index)
numerical_values_test = list(num_test.index)
#numerical_values_train

fill_num =[]
for i in numerical_values_train:
    if i in (null_few.index):
        fill_num.append(i)
        
print(fill_num)

for i in fill_num:
    train[i].fillna(train[i].median(),inplace=True)
    test[i].fillna(test[i].median(),inplace=True)
categorical_values_train = list(cat_train.index)
categorical_values_test = list(cat_test.index)

fill_cat = []
for i in categorical_values_train:
    if i in (null_few.index):
        fill_cat.append(i)

print(fill_cat)
def most_common_term(lst):
    lst = list(lst)
    return max(set(lst), key=lst.count)
#most_common_term finds the most common term in a series

most_common = ["Electrical", "Exterior1st", "Exterior2nd", "Functional", "KitchenQual", "MSZoning", "SaleType", "Utilities", "MasVnrType"]

counter = 0
for i in fill_cat:
    most_common[counter] = most_common_term(train[i])
    counter += 1

    most_common_dictionary = {fill_cat[0]: [most_common[0]], fill_cat[1]: [most_common[1]], fill_cat[2]: [most_common[2]], fill_cat[3]: [most_common[3]],
                          fill_cat[4]: [most_common[4]], fill_cat[5]: [most_common[5]], fill_cat[6]: [most_common[6]], fill_cat[7]: [most_common[7]],
                          fill_cat[8]: [most_common[8]]}
most_common_dictionary

counter = 0
for i in fill_cat:  
    train[i].fillna(most_common[counter], inplace=True)
    test[i].fillna(most_common[counter], inplace=True)
    counter += 1
training_null = pd.isnull(train).sum()
testing_null = pd.isnull(test).sum()

null = pd.concat([training_null, testing_null], axis=1, keys=["Training", "Testing"])
null[null.sum(axis=1) > 0]
sns.distplot(train['SalePrice'])
sns.distplot(np.log(train['SalePrice']))
train["TransformedPrice"] = np.log(train['SalePrice'])

#transform categorical values to representative number
for i in categorical_values_train:
    feature_set = set(train[i])
    for j in feature_set:
        feature_list = list(feature_set)
        train.loc[train[i] == j, i] = feature_list.index(j)

for i in categorical_values_test:
    feature_set2 = set(test[i])
    for j in feature_set2:
        feature_list2 = list(feature_set2)
        test.loc[test[i] == j, i] = feature_list2.index(j)
train.head()
test.head()
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import cross_val_score, KFold
X_train = train.drop(["Id","SalePrice","TransformedPrice"],axis=1).values
Y_train = train["TransformedPrice"].values
X_test = test.drop("Id",axis=1).values

from sklearn.model_selection import train_test_split #to create validation data set

X_training, X_valid, y_training, y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=0) #X_valid and y_valid are the validation sets
#fitting the model
linreg = LinearRegression()
linreg.fit(X_training,y_training)

lin_pred = linreg.predict(X_valid)
r2_lin = r2_score(y_valid, lin_pred)
rmse_lin = np.sqrt(mean_squared_error(y_valid, lin_pred))
print("R^2 Score: " + str(r2_lin))
print("RMSE Score: " + str(rmse_lin))


scores_lin = cross_val_score(linreg, X_training, y_training, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_lin)))
lasso = Lasso()
lasso.fit(X_training,y_training)

lasso_pred = lasso.predict(X_valid)
r2_lasso = r2_score(y_valid,lasso_pred)
rmse_lasso = np.sqrt(mean_squared_error(y_valid,lasso_pred))
print("R^2 Score: "+ str(r2_lasso))
print("RMSE Score: "+ str(rmse_lasso))

scores_lasso = cross_val_score(lasso,X_training,y_training,cv=10,scoring="r2")
print("cross validation score: "+str(np.mean(scores_lasso)))
ridge = Ridge()
ridge.fit(X_training,y_training)

ridge_pred = ridge.predict(X_valid)
r2_ridge = r2_score(y_valid,ridge_pred)
rmse_ridge = np.sqrt(mean_squared_error(y_valid,ridge_pred))
print("R^2 Score: "+ str(r2_ridge))
print("RMSE Score: "+ str(rmse_ridge))

scores_ridge = cross_val_score(ridge,X_training,y_training,cv=10,scoring="r2")
print("cross validation score: "+str(np.mean(scores_ridge)))


dtree = DecisionTreeRegressor()
dtree.fit(X_training,y_training)

dtree_pred = dtree.predict(X_valid)
r2_dtree = r2_score(y_valid,dtree_pred)
rmse_dtree = np.sqrt(mean_squared_error(y_valid,dtree_pred))
print("R^2 Score: "+ str(r2_dtree))
print("RMSE Score: "+ str(rmse_dtree))

scores_dtree = cross_val_score(dtree,X_training,y_training,cv=10,scoring="r2")
print("cross validation score: "+str(np.mean(scores_dtree)))
rf = RandomForestRegressor()
rf.fit(X_training,y_training)

rf_pred = rf.predict(X_valid)
r2_rf = r2_score(y_valid,rf_pred)
rmse_rf = np.sqrt(mean_squared_error(y_valid,rf_pred))
print("R^2 Score: "+ str(r2_rf))
print("RMSE Score: "+ str(rmse_rf))

scores_rf = cross_val_score(rf,X_training,y_training,cv=10,scoring="r2")
print("cross validation score: "+str(np.mean(scores_rf)))
model_performances = pd.DataFrame({
    "Model" : ["Linear Regression", "Ridge", "Lasso", "Decision Tree Regressor", "Random Forest Regressor"],
    "Best Score" : [np.mean(scores_lin),  np.mean(scores_lasso), np.mean(scores_ridge), np.mean(scores_dtree), np.mean(scores_rf)],
    "R Squared" : [str(r2_lin)[0:5], str(r2_ridge)[0:5], str(r2_lasso)[0:5], str(r2_dtree)[0:5], str(r2_rf)[0:5]],
    "RMSE" : [str(rmse_lin)[0:8], str(rmse_ridge)[0:8], str(rmse_lasso)[0:8], str(rmse_dtree)[0:8], str(rmse_rf)[0:8]]
})
model_performances.round(4)

print("Sorted by Best Score:")
model_performances.sort_values(by="Best Score", ascending=False)
print("Sorted by R Squared:")
model_performances.sort_values(by="R Squared", ascending=False)
print("Sorted by RMSE:")
model_performances.sort_values(by="RMSE", ascending=True)
rf.fit(X_train,Y_train)
predict = np.exp(rf.predict(X_test))
submission = pd.DataFrame({"Id":test["Id"],"SalePrice":predict})

print(submission.shape)