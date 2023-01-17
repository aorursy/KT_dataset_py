import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from collections import Counter

from IPython.core.display import display, HTML

sns.set_style('darkgrid')
from sklearn.datasets import load_boston

boston_dataset = load_boston()

dataset = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)
dataset.head()
dataset['MEDV'] = boston_dataset.target
dataset.head()
dataset.isnull().sum()
X = dataset.iloc[:, 0:13].values

y = dataset.iloc[:, 13].values.reshape(-1,1)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 25)
print("Shape of X_train: ",X_train.shape)

print("Shape of X_test: ", X_test.shape)

print("Shape of y_train: ",y_train.shape)

print("Shape of y_test",y_test.shape)
corr = dataset.corr()

#Plot figsize

fig, ax = plt.subplots(figsize=(10, 10))

#Generate Heat Map, allow annotations and place floats in map

sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".2f")

#Apply xticks

plt.xticks(range(len(corr.columns)), corr.columns);

#Apply yticks

plt.yticks(range(len(corr.columns)), corr.columns)

#show plot

plt.show()
sns.pairplot(dataset)

plt.show()
from sklearn.linear_model import LinearRegression

regressor_linear = LinearRegression()

regressor_linear.fit(X_train, y_train)
from sklearn.metrics import r2_score



# Predicting Cross Validation Score the Test set results

cv_linear = cross_val_score(estimator = regressor_linear, X = X_train, y = y_train, cv = 10)



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
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 2)

X_poly = poly_reg.fit_transform(X_train)

poly_reg.fit(X_poly, y_train)

regressor_poly2 = LinearRegression()

regressor_poly2.fit(X_poly, y_train)
from sklearn.metrics import r2_score



# Predicting Cross Validation Score the Test set results

cv_poly2 = cross_val_score(estimator = regressor_poly2, X = X_train, y = y_train, cv = 10)



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
from sklearn.linear_model import Ridge

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures



steps = [

    ('scalar', StandardScaler()),

    ('poly', PolynomialFeatures(degree=2)),

    ('model', Ridge(alpha=3.8, fit_intercept=True))

]



ridge_pipe = Pipeline(steps)

ridge_pipe.fit(X_train, y_train)
from sklearn.metrics import r2_score



# Predicting Cross Validation Score the Test set results

cv_ridge = cross_val_score(estimator = ridge_pipe, X = X_train, y = y_train.ravel(), cv = 10)



# Predicting R2 Score the Test set results

y_pred_ridge_train = ridge_pipe.predict(X_train)

r2_score_ridge_train = r2_score(y_train, y_pred_ridge_train)



# Predicting R2 Score the Test set results

y_pred_ridge_test = ridge_pipe.predict(X_test)

r2_score_ridge_test = r2_score(y_test, y_pred_ridge_test)



# Predicting RMSE the Test set results

rmse_ridge = (np.sqrt(mean_squared_error(y_test, y_pred_ridge_test)))

print('CV: ', cv_ridge.mean())

print('R2_score (train): ', r2_score_ridge_train)

print('R2_score (test): ', r2_score_ridge_test)

print("RMSE: ", rmse_ridge)
from sklearn.linear_model import Lasso

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures



steps = [

    ('scalar', StandardScaler()),

    ('poly', PolynomialFeatures(degree=2)),

    ('model', Lasso(alpha=0.012, fit_intercept=True, max_iter=3000))

]



lasso_pipe = Pipeline(steps)

lasso_pipe.fit(X_train, y_train)
from sklearn.metrics import r2_score



# Predicting Cross Validation Score

cv_lasso = cross_val_score(estimator = lasso_pipe, X = X_train, y = y_train, cv = 10)



# Predicting R2 Score the Test set results

y_pred_lasso_train = lasso_pipe.predict(X_train)

r2_score_lasso_train = r2_score(y_train, y_pred_lasso_train)



# Predicting R2 Score the Test set results

y_pred_lasso_test = lasso_pipe.predict(X_test)

r2_score_lasso_test = r2_score(y_test, y_pred_lasso_test)



# Predicting RMSE the Test set results

rmse_lasso = (np.sqrt(mean_squared_error(y_test, y_pred_lasso_test)))

print('CV: ', cv_lasso.mean())

print('R2_score (train): ', r2_score_lasso_train)

print('R2_score (test): ', r2_score_lasso_test)

print("RMSE: ", rmse_lasso)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_y = StandardScaler()

X_scaled = sc_X.fit_transform(X_train)

y_scaled = sc_y.fit_transform(y_train.reshape(-1,1))
# Fitting the SVR Model to the dataset

from sklearn.svm import SVR

regressor_svr = SVR(kernel = 'rbf', gamma = 'scale')

regressor_svr.fit(X_scaled, y_scaled.ravel())
from sklearn.metrics import r2_score



# Predicting Cross Validation Score

cv_svr = cross_val_score(estimator = regressor_svr, X = X_scaled, y = y_scaled.ravel(), cv = 10)



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
# Fitting the Decision Tree Regression Model to the dataset

from sklearn.tree import DecisionTreeRegressor

regressor_dt = DecisionTreeRegressor(random_state = 0)

regressor_dt.fit(X_train, y_train)
from sklearn.metrics import r2_score



# Predicting Cross Validation Score

cv_dt = cross_val_score(estimator = regressor_dt, X = X_train, y = y_train, cv = 10)



# Predicting R2 Score the Train set results

y_pred_dt_train = regressor_dt.predict(X_train)

r2_score_dt_train = r2_score(y_train, y_pred_dt_train)



# Predicting R2 Score the Test set results

y_pred_dt_test = regressor_dt.predict(X_test)

r2_score_dt_test = r2_score(y_test, y_pred_dt_test)



# Predicting RMSE the Test set results

rmse_dt = (np.sqrt(mean_squared_error(y_test, y_pred_dt_test)))

print('CV: ', cv_dt.mean())

print('R2_score (train): ', r2_score_dt_train)

print('R2_score (test): ', r2_score_dt_test)

print("RMSE: ", rmse_dt)
# Fitting the Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor

regressor_rf = RandomForestRegressor(n_estimators = 500, random_state = 0)

regressor_rf.fit(X_train, y_train.ravel())
from sklearn.metrics import r2_score



# Predicting Cross Validation Score

cv_rf = cross_val_score(estimator = regressor_rf, X = X_scaled, y = y_train.ravel(), cv = 10)



# Predicting R2 Score the Train set results

y_pred_rf_train = regressor_rf.predict(X_train)

r2_score_rf_train = r2_score(y_train, y_pred_rf_train)



# Predicting R2 Score the Test set results

y_pred_rf_test = regressor_rf.predict(X_test)

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



sns.barplot(x='Cross-Validation', y='Model', data = predict, ax = axe)

#axes[0].set(xlabel='Region', ylabel='Charges')

axe.set_xlabel('Cross-Validaton Score', size=16)

axe.set_ylabel('Model')

axe.set_xlim(0,1.0)

plt.show()
f, axes = plt.subplots(2,1, figsize=(14,10))



predict.sort_values(by=['R2_Score(training)'], ascending=False, inplace=True)



sns.barplot(x='R2_Score(training)', y='Model', data = predict, palette='Blues_d', ax = axes[0])

#axes[0].set(xlabel='Region', ylabel='Charges')

axes[0].set_xlabel('R2 Score (Training)', size=16)

axes[0].set_ylabel('Model')

axes[0].set_xlim(0,1.0)



predict.sort_values(by=['R2_Score(test)'], ascending=False, inplace=True)



sns.barplot(x='R2_Score(test)', y='Model', data = predict, palette='Reds_d', ax = axes[1])

#axes[0].set(xlabel='Region', ylabel='Charges')

axes[1].set_xlabel('R2 Score (Test)', size=16)

axes[1].set_ylabel('Model')

axes[1].set_xlim(0,1.0)



plt.show()
predict.sort_values(by=['RMSE'], ascending=False, inplace=True)



f, axe = plt.subplots(1,1, figsize=(18,6))

sns.barplot(x='Model', y='RMSE', data=predict, ax = axe)

axe.set_xlabel('Model', size=16)

axe.set_ylabel('RMSE', size=16)



plt.show()