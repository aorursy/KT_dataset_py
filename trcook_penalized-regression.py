#Load Packages

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics import mean_squared_error

%matplotlib inline 

# Linear Regression-Ordinary Least Squares

from sklearn.linear_model import LinearRegression

# Ridge Regression

from sklearn.linear_model import Ridge

# functions to cross validation to optimize hyperparameters

from sklearn.model_selection import GridSearchCV

# Lasso Regression-Least Absolute Shrinkage and Selection Operator

from sklearn.linear_model import Lasso

# ElasticNet Regression

from sklearn.linear_model import ElasticNet

#LARS Regression Model- Least Angle Regression model

from sklearn import linear_model



# using statsmodels

import statsmodels.formula.api as smf # smf -- statsmodels formula
#Load Data

#load the Boston House datasets.

#The diabetes dataset consists of 13 economic variables

#measure on 506 houses, and an median value (MEDV) as the target:

dataset = pd.read_csv("../input/week-2-data/boston_house_prices.csv", sep=",", header=1)

dataset.head()
#In a imported data set, you would break the data into two objects dataname.features (your PV) and dataname.target (your target/DV)

#There are a lot of ways to do this- here is one (it will give a warning)

#View our .data and .target

dataset.target=dataset['MEDV'] 

dataset.features=dataset.drop(['MEDV'], axis=1)                         

print(dataset.target.shape)

print(dataset.features.shape)


# fit a linear regression model to the data

model_LR = LinearRegression(normalize=True)

model_LR.fit(dataset.features, dataset.target)

print(model_LR)

# make predictions

expected_LR = dataset.target

predicted_LR = model_LR.predict(dataset.features)

# summarize the fit of the model

print("Coef", model_LR.intercept_, model_LR.coef_)

print("MSE", mean_squared_error(expected_LR, predicted_LR))
fig, ax = plt.subplots()

plt.plot(model_LR.coef_, label='LR')

plt.axhline(linewidth=4, color='r') # for reference

legend = ax.legend(loc='lower right', shadow=True)

plt.show()


# fit a ridge regression model to the data

model_RG = Ridge(alpha=2)

model_RG.fit(dataset.features, dataset.target)

print(model_RG)

# make predictions

expected_RG = dataset.target

predicted_RG= model_RG.predict(dataset.features)

# summarize the fit of the model

print("Coef", model_RG.intercept_, model_RG.coef_)

print("MSE", mean_squared_error(expected_RG, predicted_RG))
fig, ax = plt.subplots()

plt.plot(model_LR.coef_, label='LR')

plt.plot(model_RG.coef_, label='Ridge')

plt.axhline(linewidth=4, color='r') # for reference

legend = ax.legend(loc='lower right', shadow=True)

plt.show()


param_grid = {"alpha": [.01,.1, .5, 1, 2]}

#param_grid={"alpha": [1,10,1]} this does a range 1 through 10 changes by a factor of 1. 

#param_grid={"alpha": [.01,1,.05]} this does a range 1 through 1 changes by a factor of .05



# run grid search

grid_search = GridSearchCV(model_RG, param_grid=param_grid,n_jobs=-1,cv=5)

grid_search.fit(dataset.features, dataset.target)

print("Grid Scores", grid_search.cv_results_)

print("Best", grid_search.best_params_)                                   


# fit a LASSO model to the data

model_LAS = Lasso(alpha=1)

model_LAS.fit(dataset.features, dataset.target)

print(model_LAS)

# make predictions

expected_LAS = dataset.target

predicted_LAS = model_LAS.predict(dataset.features)

# summarize the fit of the model

print("Coef", model_LAS.intercept_,model_LAS.coef_)

print("MSE", mean_squared_error(expected_LAS, predicted_LAS))
fig, ax = plt.subplots()

plt.plot(model_LR.coef_, label='LR')

plt.plot(model_RG.coef_, label='Ridge')

plt.plot(model_LAS.coef_, label='Lasso')

plt.axhline(linewidth=4, color='r') # for reference

legend = ax.legend(loc='lower right', shadow=True)

plt.show()




# fit a model to the data

model_EN = ElasticNet(alpha=2)

model_EN.fit(dataset.features, dataset.target)

print(model_EN)

# make predictions

expected_EN = dataset.target

predicted_EN = model_EN.predict(dataset.features)

# summarize the fit of the model

print("Coef", model_EN.intercept_, model_EN.coef_)

print("MSE", mean_squared_error(expected_EN, predicted_EN))
#Plot all the model coefficients

fig, ax = plt.subplots()

plt.plot(model_LR.coef_, label='LR')

plt.plot(model_RG.coef_, label='Ridge')

plt.plot(model_LAS.coef_, label='Lasso')

plt.plot(model_EN.coef_, label='ElasticNet')

plt.axhline(linewidth=4, color='r') # for reference

legend = ax.legend(loc='lower right', shadow=True)

plt.show()


model_LAR = linear_model.Lars(n_nonzero_coefs=1)

model_LAR.fit(dataset.features, dataset.target)

print(model_LAR)

# make predictions

expected_LAR = dataset.target

predicted_LAR = model_LAR.predict(dataset.features)

# summarize the fit of the model

mse_LAR = np.mean((predicted_LAR-expected_LAR)**2)

print("Coef", model_LAR.intercept_, model_LAR.coef_)

print("MSE", mean_squared_error(expected_LAR, predicted_LAR))
#Plot all the model coefficients

fig, ax = plt.subplots()

plt.plot(model_LR.coef_, label='LR')

plt.plot(model_RG.coef_, label='Ridge')

plt.plot(model_LAS.coef_, label='Lasso')

plt.plot(model_EN.coef_, label='ElasticNet')

plt.plot(model_LAR.coef_, label='LARS')

plt.axhline(linewidth=4, color='r') # for reference

legend = ax.legend(loc='lower right', shadow=True)

plt.show()






# Load data

url = 'http://vincentarelbundock.github.io/Rdatasets/csv/HistData/Guerry.csv'

dat = pd.read_csv(url)





# Fit regression model (using the natural log of one of the regressors)

results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dat).fit()



# Inspect the results

print(results.summary())
#USING the STATSMODEL instead of skLearn with the diabetes dataset



# Fit regression model 

results1 = smf.ols('dataset.target~ dataset.features', data=dataset).fit()



# Inspect the results

print(results1.summary())
#extra individually

print('Parameters: ', results.params)

print('Standard errors: ', results.bse)

print('Predicted values: ', results.predict())