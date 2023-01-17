# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# regression models

from sklearn.preprocessing import StandardScaler # for feature scaling

from sklearn.pipeline import Pipeline # for using pipeline

from sklearn.linear_model import LinearRegression # for linear regression

from sklearn.linear_model import ElasticNet

from sklearn.linear_model import Ridge # for ridge regression

from sklearn.linear_model import Lasso # for lasso regression

from sklearn.svm import SVR # for support vector regression

from sklearn.tree import DecisionTreeRegressor # for decisiton tree regression

from sklearn.ensemble import RandomForestRegressor # for random forest regression

# hyptertuning

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint as sp_randint

# extra

import pandas_profiling

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from collections import Counter

from IPython.core.display import display, HTML

sns.set_style('darkgrid')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
insurance_dataset = pd.read_csv("/kaggle/input/insurance/insurance.csv")
insurance_dataset.head()
pandas_profiling.ProfileReport(insurance_dataset)
insurance_dataset.info()
plt.figure(figsize=(10,10))

sns.distplot(insurance_dataset.age)

plt.show()
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)

sns.barplot(x=insurance_dataset.sex.value_counts().index , y= insurance_dataset.sex.value_counts().values)

plt.subplot(1,2,2)

sns.boxenplot(x="sex",y="charges",data=insurance_dataset)

plt.show()
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)

sns.distplot(insurance_dataset.bmi)

plt.subplot(1,2,2)

sns.boxenplot(x="children",y="charges",data=insurance_dataset)

plt.show()
plt.figure(figsize=(10,10))

sns.distplot(insurance_dataset.children)

plt.show()
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)

sns.barplot(x=insurance_dataset.smoker.value_counts().index , y= insurance_dataset.smoker.value_counts().values)

plt.subplot(1,2,2)

sns.boxenplot(x="smoker",y="charges",data=insurance_dataset)

plt.show()
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)

sns.barplot(x=insurance_dataset.region.value_counts().index , y= insurance_dataset.region.value_counts().values)

plt.subplot(1,2,2)

sns.boxenplot(x="region",y="charges",data=insurance_dataset)

plt.show()
plt.figure(figsize=(25,10))

sns.boxenplot(x="age" , y="charges", data = insurance_dataset)

plt.show()
plt.figure(figsize=(25,10))

sns.barplot(x="age" , y="charges", data = insurance_dataset)

plt.show()
plt.figure(figsize=(10,10))

sns.scatterplot(x="charges",y="age",data=insurance_dataset,hue="smoker")

plt.show()
insurance_dataset.isnull().sum()
insurance_dataset = pd.get_dummies(insurance_dataset)

insurance_dataset.head()
X =insurance_dataset.drop("charges",axis =1).values

y =insurance_dataset["charges"].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 40)
print("Shape of X_train: ",X_train.shape)

print("Shape of X_test: ", X_test.shape)

print("Shape of y_train: ",y_train.shape)

print("Shape of y_test",y_test.shape)
plt.figure(figsize=(10,10))

sns.pairplot(insurance_dataset,vars=['age','bmi','children','charges'],hue='smoker_yes')

plt.plot()
Linear_regression = LinearRegression()

Linear_regression.fit(X_train,y_train)
# K-fold Cross validation 

# cv = number of folds of the dataset

cv_result = cross_val_score(estimator=Linear_regression,X=X,y=y,cv=10)

print("CV(K-fold) result(mean of 10) :",cv_result.mean())



# Pridicting on Train set result

# calculate R2 score

y_pridicted_train = Linear_regression.predict(X_train)

R2_score_linearregression_train = r2_score(y_train,y_pridicted_train)

print("R2 Score : ",R2_score_linearregression_train)



# Pridicting on Test set result

y_pridicted_test = Linear_regression.predict(X_test)

R2_score_linearregression_test = r2_score(y_test,y_pridicted_test)

print("R2 Score : ",R2_score_linearregression_test)



# Root mean square error

from math import sqrt

rmse = sqrt(mean_squared_error(y_test, y_pridicted_test))

print("RMSE : ",rmse)
# convert to 1-D Array

#y_train.ravel()
Linear_model_graph = pd.DataFrame({"y_train":y_train.ravel(),"y_pridicted_train":y_pridicted_train.ravel()})

plt.figure(figsize=(20,5))

sns.lineplot(x=Linear_model_graph.index[:40],y=Linear_model_graph.y_train[:40])

sns.lineplot(x=Linear_model_graph.index[:40],y=Linear_model_graph.y_pridicted_train[:40])

plt.xlabel("value")

plt.ylabel("index")

plt.title("original and pridicted value")

plt.show()
poly_model_graph = pd.DataFrame({"y_train":y_test.ravel(),"y_pridicted_train":y_pridicted_test.ravel()})

plt.figure(figsize=(20,5))

sns.scatterplot(x=poly_model_graph.index,y=poly_model_graph.y_train,marker='+')

sns.scatterplot(x=poly_model_graph.index,y=poly_model_graph.y_pridicted_train,marker='o')

plt.xlabel("value")

plt.ylabel("index")

plt.title("original and pridicted value")

plt.show()
from sklearn.preprocessing import PolynomialFeatures # for adding polynomial features

poly_regression = PolynomialFeatures(degree=2)

X_train_poly = poly_regression.fit_transform(X_train)

poly_regression.fit(X_train_poly,y_train)



regressor_poly2 = LinearRegression()

regressor_poly2.fit(X_train_poly,y_train)
# predicting cross validation 

cv_regressot_poly2 = cross_val_score(estimator=regressor_poly2,X=poly_regression.fit_transform(X),y=y,cv=10)



# R2 score train

y_priediction_poly2_train  = regressor_poly2.predict(X_train_poly)

r2_score_y_train = r2_score(y_train,y_priediction_poly2_train)

print("r2_score_y_train : ", r2_score_y_train )



# R2 score test

y_priediction_poly2_test  = regressor_poly2.predict(poly_regression.fit_transform(X_test))

r2_score_y_test = r2_score(y_test,y_priediction_poly2_test)

print("r2_score_y_test : ",r2_score_y_test)



# root mean square error

from math import sqrt 

rmse_test = sqrt(mean_squared_error(y_priediction_poly2_test,y_test))

print("rmse_test :",rmse_test)
poly_model_graph = pd.DataFrame({"y_train":y_test.ravel(),"y_pridicted_train":y_priediction_poly2_test.ravel()})

plt.figure(figsize=(20,5))

sns.lineplot(x=poly_model_graph.index[:40],y=poly_model_graph.y_train[:40])

sns.lineplot(x=poly_model_graph.index[:40],y=poly_model_graph.y_pridicted_train[:40])

plt.xlabel("value")

plt.ylabel("index")

plt.title("original and pridicted value")

plt.show()
poly_model_graph = pd.DataFrame({"y_train":y_test.ravel(),"y_pridicted_train":y_priediction_poly2_test.ravel()})

plt.figure(figsize=(20,5))

sns.scatterplot(x=poly_model_graph.index,y=poly_model_graph.y_train,marker='+')

sns.scatterplot(x=poly_model_graph.index,y=poly_model_graph.y_pridicted_train,marker='o')

plt.xlabel("value")

plt.ylabel("index")

plt.title("original and pridicted value")

plt.show()
def ridge_regression(X_data,y_data,X_test,y_test, alpha):

    

    #fit ridge regression 

    ridge_reg = Ridge(alpha=alpha,normalize=True)

    ridge_reg.fit(X_data,y_data)

    

    y_pred_train = ridge_reg.predict(X_data)

    y_pred_test = ridge_reg.predict(X_test)

    

#     #plot graph

    

#     poly_model_graph = pd.DataFrame({"y_data":y_data.ravel(),"y_pridicted_train":y_pred_train.ravel()})

#     plt.figure(figsize=(20,5))

#     sns.lineplot(x=poly_model_graph.index[:80],y=poly_model_graph.y_data[:80])

#     sns.lineplot(x=poly_model_graph.index[:80],y=poly_model_graph.y_pridicted_train[:80])

#     plt.xlabel("value")

#     plt.ylabel("index")

#     plt.title("original and pridicted value in train dataset at alpha = {}".format(alpha))

#     plt.show()

   

#     poly_model_graph = pd.DataFrame({"y_data":y_test.ravel(),"y_pridicted_train":y_pred_test.ravel()})

#     plt.figure(figsize=(20,5))

#     sns.lineplot(x=poly_model_graph.index[:80],y=poly_model_graph.y_data[:80])

#     sns.lineplot(x=poly_model_graph.index[:80],y=poly_model_graph.y_pridicted_train[:80])

#     plt.xlabel("value")

#     plt.ylabel("index")

#     plt.title("original and pridicted value in train dataset at alpha = {}".format(alpha))

#     plt.show()

    

    #check errors

    print("\n\nfor alpha = {}".format(alpha))

    

    # R2 score train

    r2_score_y_train = r2_score(y_train,y_pred_train)

    print("r2_score_y_train : ", r2_score_y_train )



    # R2 score test

    r2_score_y_test = r2_score(y_test,y_pred_test)

    print("r2_score_y_test : ",r2_score_y_test)



    # root mean square error

    from math import sqrt 

    rmse_test = sqrt(mean_squared_error(y_pred_test,y_test))

    print("rmse_test :",rmse_test)

    rmse_test = sqrt(mean_squared_error(y_pred_train,y_data))

    print("rmse_train :",rmse_test)

    

    return [r2_score_y_train,r2_score_y_test,rmse_test] , ridge_reg.coef_
r2_train_score_r = []

r2_test_score_r = []

rms_score_r = []

cofficient_r = []

for Alpha in [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]:

    temp , coff = ridge_regression(X_train,y_train,X_test,y_test,Alpha)

    r2_train_score_r.append(temp[0])

    r2_test_score_r.append(temp[1])

    rms_score_r.append(temp[2])

    cofficient_r.append(coff)
graph = pd.DataFrame({"Alpha":["1e-15", "1e-10", "1e-8", "1e-4","1e-3","1e-2"," 1"," 5"," 10"," 20"],"R2 score for train":r2_train_score_r,

                      "R2 score for test":r2_test_score_r,"Rmse":rms_score_r})

plt.figure(figsize=(20,10))

plt.subplot(2,1,1)

sns.lineplot(x=graph.Alpha,y=graph["R2 score for train"],label="R2 score for train")

sns.lineplot(x=graph.Alpha,y=graph["R2 score for test"],label="R2 score for test")

plt.subplot(2,1,2)

sns.lineplot(x=graph.Alpha,y=graph["Rmse"],label="RMSE")

plt.show()
#disply cofficient

cofficient_dataframe = pd.DataFrame()
def lasso_regression(X_train,y_train,X_test,y_test,alpha):

    lasso_reg = Lasso(alpha=alpha,normalize=True)

    lasso_reg.fit(X_train,y_train)

    

    y_pred_train = lasso_reg.predict(X_train)

    

    y_pred_test = lasso_reg.predict(X_test)

    

    #check errors

    print("\n\nfor alpha = {}".format(alpha))

    

    # R2 score train

    r2_score_y_train = r2_score(y_train,y_pred_train)

    print("r2_score_y_train : ", r2_score_y_train )



    # R2 score test

    r2_score_y_test = r2_score(y_test,y_pred_test)

    print("r2_score_y_test : ",r2_score_y_test)



    # root mean square error

    from math import sqrt 

    rmse_test = sqrt(mean_squared_error(y_pred_test,y_test))

    print("rmse_test :",rmse_test)

    rmse_test = sqrt(mean_squared_error(y_pred_train,y_train))

    print("rmse_train :",rmse_test)

    return [r2_score_y_train,r2_score_y_test,rmse_test] , lasso_reg.coef_
r2_train_score_l = []

r2_test_score_l = []

rms_score_l = []

cofficient_l = []

for Alpha in [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]:

    temp , coff = lasso_regression(X_train,y_train,X_test,y_test,Alpha)

    r2_train_score_l.append(temp[0])

    r2_test_score_l.append(temp[1])

    rms_score_l.append(temp[2])

    cofficient_l.append(coff)
graph_1 = pd.DataFrame({"Alpha":["1e-15", "1e-10", "1e-8", "1e-4","1e-3","1e-2"," 1"," 5"," 10"," 20"],"R2 score for train":r2_train_score_l,

                      "R2 score for test":r2_test_score_l,"Rmse":rms_score_l})

plt.figure(figsize=(20,10))

plt.subplot(2,1,1)

sns.lineplot(x=graph_1.Alpha,y=graph_1["R2 score for train"],label="R2 score for train")

sns.lineplot(x=graph_1.Alpha,y=graph_1["R2 score for test"],label="R2 score for test")

plt.subplot(2,1,2)

sns.lineplot(x=graph_1.Alpha,y=graph_1["Rmse"],label="RMSE")

plt.show()
plt.figure(figsize=(10,5))

sns.lineplot(x=graph_1.Alpha,y=r2_train_score_l,label="r2_train_score_lasso")

sns.lineplot(x=graph_1.Alpha,y=r2_train_score_r,label="r2_train_score_ridge")

plt.show()
plt.figure(figsize=(10,5))

sns.lineplot(x=graph_1.Alpha,y=r2_test_score_l,label="r2_test_score_lasso")

sns.lineplot(x=graph_1.Alpha,y=r2_test_score_r,label="r2_test_score_ridge")

plt.show()
plt.figure(figsize=(10,5))

sns.lineplot(x=graph_1.Alpha,y=rms_score_l,label="Rmse lesso")

sns.lineplot(x=graph_1.Alpha,y=rms_score_r,label="Rmse ridge")

plt.show()
Elastic_regression = ElasticNet()

Elastic_regression.fit(X_train,y_train)



y_predicted_train = Elastic_regression.predict(X_train)



y_predicted_test = Elastic_regression.predict(X_test)



# R2 score train

r2_score_y_train = r2_score(y_train,y_predicted_train)

print("r2_score_y_train : ", r2_score_y_train )



# R2 score test

r2_score_y_test = r2_score(y_test,y_predicted_test)

print("r2_score_y_test : ",r2_score_y_test)



# root mean square error

from math import sqrt 

rmse_test = sqrt(mean_squared_error(y_predicted_test,y_test))

print("rmse_test :",rmse_test)

rmse_test = sqrt(mean_squared_error(y_predicted_train,y_train))

print("rmse_train :",rmse_test)