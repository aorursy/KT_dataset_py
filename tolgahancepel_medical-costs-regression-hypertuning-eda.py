# data science

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# regression models

from sklearn.preprocessing import StandardScaler # for feature scaling

from sklearn.pipeline import Pipeline # for using pipeline

from sklearn.linear_model import LinearRegression # for linear regression

from sklearn.preprocessing import PolynomialFeatures # for adding polynomial features

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

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from collections import Counter

from IPython.core.display import display, HTML

sns.set_style('darkgrid')
dataset = pd.read_csv('../input/insurance/insurance.csv')

dataset.head()
dataset.count()
dataset.describe()
dataset.isnull().sum()
corr = dataset.corr()

#Plot figsize

fig, ax = plt.subplots(figsize=(10, 8))

#Generate Heat Map, allow annotations and place floats in map

sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f")

#Apply xticks

plt.xticks(range(len(corr.columns)), corr.columns);

#Apply yticks

plt.yticks(range(len(corr.columns)), corr.columns)

#show plot

plt.show()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['age'], ax = axes[0])

axes[0].set_xlabel('Age', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.scatterplot(x = 'charges', y = 'age', data = dataset, hue = 'smoker', ax = axes[1])

axes[1].set_xlabel('Charges', fontsize=14)

axes[1].set_ylabel('Age', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()



plt.show()
f, axe = plt.subplots(1,1,figsize=(20,4))

sns.boxenplot(x = 'age', y = 'charges', data = dataset, ax = axe)

axe.set_xlabel('Age', fontsize=14)

axe.set_ylabel('Charges', fontsize=14)

plt.show()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['bmi'], ax = axes[0])

axes[0].set_xlabel('Body Mass Index (BMI)', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.scatterplot(x = 'charges', y = 'bmi', data = dataset, hue = 'sex',ax = axes[1])

axes[1].set_xlabel('Charges', fontsize=14)

axes[1].set_ylabel('Body Mass Index (BMI)', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()



plt.show()
sex_list = Counter(dataset['sex'])

labels = sex_list.keys()

sizes = sex_list.values()



f, axes = plt.subplots(1,2,figsize=(14,4))



sns.countplot(dataset['sex'], ax = axes[0], palette="Set1")

axes[0].set_xlabel('Sex', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.boxenplot(x = 'sex', y = 'charges', data = dataset, hue = 'sex', ax = axes[1])

axes[1].set_xlabel('Charges', fontsize=14)

axes[1].set_ylabel('Sex', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(0.6,1), loc=1, borderaxespad=0.)



plt.show()
children_list = Counter(dataset['children'])

labels = children_list.keys()

sizes = children_list.values()



f, axes = plt.subplots(1,2,figsize=(14,4))



sns.countplot(dataset['children'], ax = axes[0], palette="Set1")

axes[0].set_xlabel('Children', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.boxplot(x = 'children', y = 'charges', data = dataset, ax = axes[1])

axes[1].set_xlabel('Children', fontsize=14)

axes[1].set_ylabel('Charges', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()



plt.show()
smoker_list = Counter(dataset['smoker'])

labels = smoker_list.keys()

sizes = smoker_list.values()



f, axes = plt.subplots(1,2,figsize=(14,4))



sns.countplot(dataset['smoker'], ax = axes[0], palette="Set1")

axes[0].set_xlabel('Smoker', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.violinplot(x = 'smoker', y = 'charges', data = dataset, hue = 'smoker', ax = axes[1])

axes[1].set_xlabel('Smoker', fontsize=14)

axes[1].set_ylabel('Charges', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()



plt.show()
region_list = Counter(dataset['region'])

labels = region_list.keys()

sizes = region_list.values()



f, axes = plt.subplots(1,2,figsize=(14,4))



sns.countplot(dataset['region'], ax = axes[0], palette="Set1")

axes[0].set_xlabel('Region', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.swarmplot(x = 'region', y = 'charges', data = dataset, hue = 'region', ax = axes[1])

axes[1].set_xlabel('Region', fontsize=14)

axes[1].set_ylabel('Charges', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)



plt.show()
dataset = pd.get_dummies(dataset)
dataset.head()
X = dataset.drop('charges', axis = 1).values

y = dataset['charges'].values.reshape(-1,1)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
print("Shape of X_train: ",X_train.shape)

print("Shape of X_test: ", X_test.shape)

print("Shape of y_train: ",y_train.shape)

print("Shape of y_test",y_test.shape)
# Creating the linear regressor

regressor_linear = LinearRegression()

regressor_linear.fit(X_train, y_train)
# Predicting Cross Validation Score the Test set results

cv_linear = cross_val_score(estimator = regressor_linear, X = X, y = y, cv = 10)



# Predicting R2 Score the Train set results

y_pred_linear_train = regressor_linear.predict(X_train)

r2_score_linear_train = r2_score(y_train, y_pred_linear_train)



# Predicting R2 Score the Test set results

y_pred_linear_test = regressor_linear.predict(X_test)

r2_score_linear_test = r2_score(y_test, y_pred_linear_test)



# Predicting RMSE the Test set results

rmse_linear = (np.sqrt(mean_squared_error(y_test, y_pred_linear_test)))

print("CV: ", cv_linear.mean())

print('R2_score (train): ', r2_score_linear_train)

print('R2_score (test): ', r2_score_linear_test)

print("RMSE: ", rmse_linear)
# Creating the polynomial features and regressor

poly_reg = PolynomialFeatures(degree = 2)

X_poly = poly_reg.fit_transform(X)

X_train_poly = poly_reg.fit_transform(X_train)

poly_reg.fit(X_train_poly, y_train)



regressor_poly2 = LinearRegression()

regressor_poly2.fit(X_train_poly, y_train)
# Predicting Cross Validation Score the Test set results

cv_poly2 = cross_val_score(estimator = regressor_poly2, X = X_poly, y = y, cv = 10)



# Predicting R2 Score the Train set results

y_pred_poly2_train = regressor_poly2.predict(poly_reg.fit_transform(X_train))

r2_score_poly2_train = r2_score(y_train, y_pred_poly2_train)



# Predicting R2 Score the Test set results

y_pred_poly2_test = regressor_poly2.predict(poly_reg.fit_transform(X_test))

r2_score_poly2_test = r2_score(y_test, y_pred_poly2_test)



# Predicting RMSE the Test set results

rmse_poly2 = (np.sqrt(mean_squared_error(y_test, y_pred_poly2_test)))

print('CV: ', cv_poly2.mean())

print('R2_score (train): ', r2_score_poly2_train)

print('R2_score (test): ', r2_score_poly2_test)

print("RMSE: ", rmse_poly2)
steps = [

    ('scalar', StandardScaler()),

    ('poly', PolynomialFeatures(degree=2)),

    ('model', Ridge())

]



ridge_pipe = Pipeline(steps)
# Applying Grid Search to find the best model and the best parameters

# step 1: alpha:[200, 230, 250,265, 270, 275, 290, 300, 500] -> 200

# step 2: alpha:[10,50,100,150,200] -> 50

# step 3: alpha: np.arange(30, 75, 1) -> 43



parameters =  {  'model__alpha' : [43],

                 'model__fit_intercept' : [True],

                 'model__tol' : [0.0001],

                 'model__solver' : ['auto'],

                'model__random_state': [42] 

}

regressor_ridge = GridSearchCV(ridge_pipe, parameters, iid=False, cv=10)

regressor_ridge = regressor_ridge.fit(X, y.ravel())
print(regressor_ridge.best_score_)

print(regressor_ridge.best_params_)
# Predicting Cross Validation Score the Test set results

cv_ridge = regressor_ridge.best_score_



# Predicting R2 Score the Test set results

y_pred_ridge_train = regressor_ridge.predict(X_train)

r2_score_ridge_train = r2_score(y_train, y_pred_ridge_train)



# Predicting R2 Score the Test set results

y_pred_ridge_test = regressor_ridge.predict(X_test)

r2_score_ridge_test = r2_score(y_test, y_pred_ridge_test)



# Predicting RMSE the Test set results

rmse_ridge = (np.sqrt(mean_squared_error(y_test, y_pred_ridge_test)))

print('CV: ', cv_ridge.mean())

print('R2_score (train): ', r2_score_ridge_train)

print('R2_score (test): ', r2_score_ridge_test)

print("RMSE: ", rmse_ridge)
steps = [

    ('scalar', StandardScaler()),

    ('poly', PolynomialFeatures(degree=2)),

    ('model', Lasso())

]



lasso_pipe = Pipeline(steps)
# Applying Grid Search to find the best model and the best parameters

# step 1: alpha:np.arange(0.01, 1, 0.005) -> 0.9949



parameters =  {  'model__alpha' : [0.9949],

                 'model__fit_intercept' : [True],

                 'model__tol' : [0.0001],

                 'model__max_iter' : [5000],

                'model__random_state': [42] 

}

regressor_lasso = GridSearchCV(lasso_pipe, parameters, iid=False, cv=10, n_jobs = -1, verbose = 4)

regressor_lasso = regressor_lasso.fit(X, y.ravel())
# Predicting Cross Validation Score

cv_lasso = regressor_lasso.best_score_



# Predicting R2 Score the Test set results

y_pred_lasso_train = regressor_lasso.predict(X_train)

r2_score_lasso_train = r2_score(y_train, y_pred_lasso_train)



# Predicting R2 Score the Test set results

y_pred_lasso_test = regressor_lasso.predict(X_test)

r2_score_lasso_test = r2_score(y_test, y_pred_lasso_test)



# Predicting RMSE the Test set results

rmse_lasso = (np.sqrt(mean_squared_error(y_test, y_pred_lasso_test)))

print('CV: ', cv_lasso.mean())

print('R2_score (train): ', r2_score_lasso_train)

print('R2_score (test): ', r2_score_lasso_test)

print("RMSE: ", rmse_lasso)
# Feature Scaling

sc_X = StandardScaler()

sc_y = StandardScaler()

X_scaled = sc_X.fit_transform(X)

y_scaled = sc_y.fit_transform(y.reshape(-1,1))
# Creating the SVR regressor

regressor_svr = SVR()
# Applying Grid Search to find the best model and the best parameters

parameters =  { 'kernel' : ['rbf', 'sigmoid'],

                 'gamma' : [0.001, 0.01, 0.1, 1, 'scale'],

                 'tol' : [0.0001],

                 'C': [0.001, 0.01, 0.1, 1, 10, 100] }

regressor_svr = GridSearchCV(estimator = regressor_svr,

                           param_grid = parameters,

                           cv = 10,

                           verbose = 4,

                           iid = True,

                           n_jobs = -1)

regressor_svr = regressor_svr.fit(X_scaled, y_scaled.ravel())
print(regressor_svr.best_params_)

print(regressor_svr.best_score_)
# Predicting Cross Validation Score

cv_svr = regressor_svr.best_score_



# Predicting R2 Score the Train set results

y_pred_svr_train = sc_y.inverse_transform(regressor_svr.predict(sc_X.transform(X_train)))

r2_score_svr_train = r2_score(y_train, y_pred_svr_train)



# Predicting R2 Score the Test set results

y_pred_svr_test = sc_y.inverse_transform(regressor_svr.predict(sc_X.transform(X_test)))

r2_score_svr_test = r2_score(y_test, y_pred_svr_test)



# Predicting RMSE the Test set results

rmse_svr = (np.sqrt(mean_squared_error(y_test, y_pred_svr_test)))

print('CV: ', cv_svr.mean())

print('R2_score (train): ', r2_score_svr_train)

print('R2_score (test): ', r2_score_svr_test)

print("RMSE: ", rmse_svr)
# Creating the Decision Tree regressor

regressor_dt = DecisionTreeRegressor(random_state = 42)
# Applying Grid Search to find the best model and the best parameters

parameters = [ { "max_depth": np.arange(1,21),

              "min_samples_leaf": [1, 5, 10, 20, 50, 100],

              "min_samples_split": np.arange(2, 11),

              "criterion": ["mse"],

              "random_state" : [42]}

            ]

regressor_dt = GridSearchCV(estimator = regressor_dt,

                           param_grid  = parameters,

                           cv = 10,

                           verbose = 4,

                           iid = False,

                           n_jobs = -1)

regressor_dt = regressor_dt.fit(X_scaled, y_scaled)
print(regressor_dt.best_params_)

print(regressor_dt.best_score_)
# Predicting Cross Validation Score

cv_dt = regressor_dt.best_score_



# Predicting R2 Score the Train set results

y_pred_dt_train = sc_y.inverse_transform(regressor_dt.predict(sc_X.transform(X_train)))

r2_score_dt_train = r2_score(y_train, y_pred_dt_train)



# Predicting R2 Score the Test set results

y_pred_dt_test = sc_y.inverse_transform(regressor_dt.predict(sc_X.transform(X_test)))

r2_score_dt_test = r2_score(y_test, y_pred_dt_test)



# Predicting RMSE the Test set results

rmse_dt = (np.sqrt(mean_squared_error(y_test, y_pred_dt_test)))

print('CV: ', cv_dt.mean())

print('R2_score (train): ', r2_score_dt_train)

print('R2_score (test): ', r2_score_dt_test)

print("RMSE: ", rmse_dt)
# Creating the Random Forest regressor

regressor_rf = RandomForestRegressor()
# RANDOM SEARCH - STEP 1 - TOOK 17.6 MINUTES

#parameters =  { "n_estimators": [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],

#                "max_features": ["auto", "sqrt"],

#                "max_depth": np.linspace(10, 110, num = 11),

#                "min_samples_split": [2, 5, 10],

#                "min_samples_leaf": [1, 2, 4],

#                "bootstrap": [True, False],

#                "criterion": ["mse"],

#                "random_state" : [42] }

#            

#regressor_rf = RandomizedSearchCV(estimator = regressor_rf,

#                                  param_distributions = parameters,

#                                  n_iter = 100,

#                                  cv = 10,

#                                  random_state=42,

#                                  verbose = 4,

#                                  n_jobs = -1)

#regressor_rf = regressor_rf.fit(X_scaled, y.ravel())

#

# Best Parameters and Score:

# {'random_state': 42, 'n_estimators': 1200, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 50.0, 'criterion': 'mse', 'bootstrap': True}

# 0.8541297253461337
# GRID SEARCH - STEP 2 - TOOK 1.5 MINUTES

#parameters =  { "n_estimators": [1200],

#                "max_features": ["auto"],

#                "max_depth": [50],

#                "min_samples_split": [7,10,13],

#                "min_samples_leaf": [4,7,10],

#                "bootstrap": [True],

#                "criterion": ["mse"],

#                "random_state" : [42] }

#            

#regressor_rf = GridSearchCV(estimator = regressor_rf,

#                                  param_grid = parameters,

#                                  cv = 10,

#                                  verbose = 4,

#                                  n_jobs = -1)

#regressor_rf = regressor_rf.fit(X_scaled, y.ravel())

#

# Best Parameters and Score:

# {'bootstrap': True, 'criterion': 'mse', 'max_depth': 50, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 7, 'n_estimators': 1200, 'random_state': 42}

# 0.8587579970188238
# Applying RandomSearch and GridSearch to find the best model and the best parameters

parameters =  { "n_estimators": [1200],

                "max_features": ["auto"],

                "max_depth": [50],

                "min_samples_split": [7],

                "min_samples_leaf": [10],

                "bootstrap": [True],

                "criterion": ["mse"],

                "random_state" : [42] }

            

regressor_rf = GridSearchCV(estimator = regressor_rf,

                                  param_grid = parameters,

                                  cv = 10,

                                # verbose = 4,

                                  n_jobs = -1)

regressor_rf = regressor_rf.fit(X_scaled, y.ravel())
print(regressor_rf.best_params_)

print(regressor_rf.best_score_)
from sklearn.metrics import r2_score



# Predicting Cross Validation Score

cv_rf = regressor_rf.best_score_



# Predicting R2 Score the Train set results

y_pred_rf_train = regressor_rf.predict(sc_X.transform(X_train))

r2_score_rf_train = r2_score(y_train, y_pred_rf_train)



# Predicting R2 Score the Test set results

y_pred_rf_test = regressor_rf.predict(sc_X.transform(X_test))

r2_score_rf_test = r2_score(y_test, y_pred_rf_test)



# Predicting RMSE the Test set results

rmse_rf = (np.sqrt(mean_squared_error(y_test, y_pred_rf_test)))

print('CV: ', cv_rf.mean())

print('R2_score (train): ', r2_score_rf_train)

print('R2_score (test): ', r2_score_rf_test)

print("RMSE: ", rmse_rf)
models = [('Linear Regression', rmse_linear, r2_score_linear_train, r2_score_linear_test, cv_linear.mean()),

          ('Polynomial Regression (2nd)', rmse_poly2, r2_score_poly2_train, r2_score_poly2_test, cv_poly2.mean()),

          ('Ridge Regression', rmse_ridge, r2_score_ridge_train, r2_score_ridge_test, cv_ridge.mean()),

          ('Lasso Regression', rmse_lasso, r2_score_lasso_train, r2_score_lasso_test, cv_lasso.mean()),

          ('Support Vector Regression', rmse_svr, r2_score_svr_train, r2_score_svr_test, cv_svr.mean()),

          ('Decision Tree Regression', rmse_dt, r2_score_dt_train, r2_score_dt_test, cv_dt.mean()),

          ('Random Forest Regression', rmse_rf, r2_score_rf_train, r2_score_rf_test, cv_rf.mean())   

         ]
predict = pd.DataFrame(data = models, columns=['Model', 'RMSE', 'R2_Score(training)', 'R2_Score(test)', 'Cross-Validation'])

predict
f, axe = plt.subplots(1,1, figsize=(18,6))



predict.sort_values(by=['Cross-Validation'], ascending=False, inplace=True)



sns.barplot(x='Cross-Validation', y='Model', data = predict, ax = axe, palette='viridis')

#axes[0].set(xlabel='Region', ylabel='Charges')

axe.set_xlabel('Cross-Validaton Score', size=16)

axe.set_ylabel('Model')

axe.set_xlim(0,1.0)

axe.set_xticks(np.arange(0, 1.1, 0.1))

plt.show()
f, axes = plt.subplots(2,1, figsize=(14,10))



predict.sort_values(by=['R2_Score(training)'], ascending=False, inplace=True)



sns.barplot(x='R2_Score(training)', y='Model', data = predict, palette='Blues_d', ax = axes[0])

#axes[0].set(xlabel='Region', ylabel='Charges')

axes[0].set_xlabel('R2 Score (Training)', size=16)

axes[0].set_ylabel('Model')

axes[0].set_xlim(0,1.0)

axes[0].set_xticks(np.arange(0, 1.1, 0.1))



predict.sort_values(by=['R2_Score(test)'], ascending=False, inplace=True)



sns.barplot(x='R2_Score(test)', y='Model', data = predict, palette='Reds_d', ax = axes[1])

#axes[0].set(xlabel='Region', ylabel='Charges')

axes[1].set_xlabel('R2 Score (Test)', size=16)

axes[1].set_ylabel('Model')

axes[1].set_xlim(0,1.0)

axes[1].set_xticks(np.arange(0, 1.1, 0.1))



plt.show()
predict.sort_values(by=['RMSE'], ascending=False, inplace=True)



f, axe = plt.subplots(1,1, figsize=(18,6))

sns.barplot(x='Model', y='RMSE', data=predict, ax = axe)

axe.set_xlabel('Model', size=16)

axe.set_ylabel('RMSE', size=16)



plt.show()