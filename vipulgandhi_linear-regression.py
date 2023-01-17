# imports

import pandas as pd

import numpy as np

import seaborn as sns

import statsmodels.api as sm

import statsmodels.formula.api as smf

from statsmodels.tools.eval_measures import rmse

from sklearn import metrics

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



# allow plots to appear directly in the notebook

%matplotlib inline



# Supress Warnings

import warnings

warnings.filterwarnings('ignore')
# Advertising data set contains information about money spent on advertisement (TV, Radio and Newspaper) and their generated Sales.



df_advertising = pd.read_csv("../input/advertising-dataset/advertising.csv")

df_advertising.head(2)
# shape of the DataFrame

df_advertising.shape
# visualize the relationship between the features and the target using scatterplots

sns.pairplot(df_advertising, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size = 4, aspect = 1)
sns.pairplot(df_advertising[['TV','Radio','Newspaper']])
sns.heatmap(df_advertising[['TV','Radio','Newspaper']].corr(), annot = True)
# This function will regress y on x and then draw a scatterplot of the residuals.



sns.residplot(x = df_advertising['TV'], y = df_advertising["Sales"], lowess = True)
Ststsmodels_model = smf.ols(formula='Sales ~ TV', data = df_advertising)

Ststsmodels_result = Ststsmodels_model.fit()



# print the coefficients

Ststsmodels_result.params
### SCIKIT-LEARN ###



X = df_advertising[['TV']]

y = df_advertising[["Sales"]]



SkLearn_model = LinearRegression()

SkLearn_result = SkLearn_model.fit(X, y)



# print the coefficients

print(SkLearn_result.intercept_)

print(SkLearn_result.coef_)
# manually calculate the prediction

Sales = 6.97482149 + 0.05546477*50

Sales * 1000
### STATSMODELS ###



X_new = pd.DataFrame({'TV': [50]})



# predict for a new observation

Sales = Ststsmodels_result.predict(X_new)

Sales * 1000
### SCIKIT-LEARN ###



# predict for a new observation

Sales = SkLearn_result.predict(np.array(50).reshape(1,-1))

Sales * 1000
sns.pairplot(df_advertising, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=4, aspect = 1, kind='reg')
Ststsmodels_residual = Ststsmodels_result.resid

ax = sm.qqplot(Ststsmodels_residual, fit = True, line = "45")
### STATSMODELS ###



# print the confidence intervals for the model coefficients

Ststsmodels_result.conf_int()
### STATSMODELS ###



# print the p-values for the model coefficients

Ststsmodels_result.pvalues
### STATSMODELS ###



# print a summary of the fitted model

Ststsmodels_result.summary()
### SCIKIT-LEARN ###



# print the R-squared value for the model

SkLearn_result.score(X, y)
### STATSMODELS ###



# create a fitted model with all three features

Ststsmodels_model = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=df_advertising)

Ststsmodels_result = Ststsmodels_model.fit()



# print the coefficients

Ststsmodels_result.params
### SCIKIT-LEARN ###



feature_cols = ['TV', 'Radio', 'Newspaper']

X = df_advertising[feature_cols]

y = df_advertising[["Sales"]]



# instantiate and fit

SkLearn_model = LinearRegression()

SkLearn_result = SkLearn_model.fit(X, y)



# print the coefficients

print(SkLearn_result.intercept_)

print(SkLearn_result.coef_)
### STATSMODELS ###



# print a summary of the fitted model

Ststsmodels_result.summary()
# only include TV and Radio in the model



# instantiate and fit model

Ststsmodels_model = smf.ols(formula='Sales ~ TV + Radio', data=df_advertising)

Ststsmodels_result = Ststsmodels_model.fit()



# print a summary of the fitted model

Ststsmodels_result.summary()
# exclude Newspaper

X = df_advertising[['TV', 'Radio']]

y = df_advertising.Sales



# Split data

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)



# Instantiate model

lm2 = LinearRegression()



# Fit model

lm2.fit(X_train, y_train)



# Predict

y_pred = lm2.predict(X_test)



# RMSE

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# include Newspaper

X = df_advertising[['TV', 'Radio', 'Newspaper']]

y = df_advertising.Sales



# Split data

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)





# Instantiate model

lm2 = LinearRegression()



# Fit Model

lm2.fit(X_train, y_train)



# Predict

y_pred = lm2.predict(X_test)



# RMSE

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.linear_model import Ridge



ridgeReg = Ridge(alpha=0.1, normalize=True)



ridgeReg.fit(X_train,y_train)



y_pred = ridgeReg.predict(X_test)



# RMSE

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(ridgeReg.intercept_)

print(ridgeReg.coef_)
ridgeReg = Ridge(alpha=0.9, normalize=True)



ridgeReg.fit(X_train,y_train)



y_pred = ridgeReg.predict(X_test)



# RMSE

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(ridgeReg.intercept_)

print(ridgeReg.coef_)
from sklearn.linear_model import Lasso



lassoReg = Lasso(alpha=0.1, normalize=True)



lassoReg.fit(X_train,y_train)



y_pred = lassoReg.predict(X_test)



# RMSE

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(lassoReg.intercept_)

print(lassoReg.coef_)
lassoReg = Ridge(alpha=0.9, normalize=True)



lassoReg.fit(X_train,y_train)



y_pred = lassoReg.predict(X_test)



# RMSE

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(lassoReg.intercept_)

print(lassoReg.coef_)
from sklearn.linear_model import ElasticNet



elsticNetReg = ElasticNet( l1_ratio=0.2, normalize=True)



elsticNetReg.fit(X_train,y_train)



y_pred = elsticNetReg.predict(X_test)



# RMSE

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(elsticNetReg.intercept_)

print(elsticNetReg.coef_)