# Ridge Regression

import numpy as np

from sklearn import datasets

from sklearn.linear_model import Ridge

# import lasso

from sklearn.linear_model import Lasso

#import elasticnet

from sklearn.linear_model import ElasticNet
# load the diabetes datasets

dataset = datasets.load_diabetes()
# fit a ridge regression model to the data

model = Ridge(alpha=0.1)

model.fit(dataset.data, dataset.target)

print(model)

# make predictions

expected = dataset.target

predicted = model.predict(dataset.data)

# summarize the fit of the model

mse = np.mean((predicted-expected)**2)

print(mse)

print(model.score(dataset.data, dataset.target))
# Lasso Regression

# fit a LASSO model to the data



model = Lasso(alpha=0.1)

model.fit(dataset.data, dataset.target)

print(model)

# make predictions

expected = dataset.target

predicted = model.predict(dataset.data)

# summarize the fit of the model

mse = np.mean((predicted-expected)**2)

print(mse)

print(model.score(dataset.data, dataset.target))
# ElasticNet Regression

# fit a model to the data



model = ElasticNet(alpha=0.1)

model.fit(dataset.data, dataset.target)

print(model)

# make predictions

expected = dataset.target

predicted = model.predict(dataset.data)

# summarize the fit of the model

mse = np.mean((predicted-expected)**2)

print(mse)

print(model.score(dataset.data, dataset.target))