# Importing the libraries

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import pandas as pd
import seaborn as sns
import keras as K
import keras.layers as Dense
import keras.models as Sequential
import keras.optimizers as Adam

import numpy as np
# reading the data
data = pd.read_csv('../input/avocado-prices/avocado.csv')
data.head()
# Seperating the prices to be predicted
y = data.AveragePrice	
data.drop(['AveragePrice'], axis=1, inplace=True)

# Splitting the data into training and test datasets
from sklearn.model_selection import train_test_split

trainflights, testflights, ytrain, ytest = train_test_split(data, y, train_size=0.8,
                                                            test_size=0.2, random_state=0)


s = (trainflights.dtypes == 'object')
object_cols = list(s[s].index)

n = (trainflights.dtypes == ('float64','int64'))
numerical_cols = list(n[n].index)
# Checking the columns containing categorical columns:
print(object_cols)
# Using One Hot Encoder to make the categorical columns usable

oneHot = OneHotEncoder(handle_unknown = 'ignore', sparse=False)
oneHottrain = pd.DataFrame(oneHot.fit_transform(trainflights[object_cols]))
oneHottest = pd.DataFrame(oneHot.transform(testflights[object_cols]))

# Reattaching index since OneHotEncoder removes them:
oneHottrain.index = trainflights.index
oneHottest.index = testflights.index 

# Dropping the old categorical columns:
cattraincol = trainflights.drop(object_cols, axis=1)
cattestcol = testflights.drop(object_cols, axis=1)

# Concatenating the new columns:
trainflights = pd.concat([cattraincol, oneHottrain], axis=1)
testflights = pd.concat([cattestcol, oneHottest], axis=1)
# Scaling the values

trainf = trainflights.values
testf = testflights.values

minmax = MinMaxScaler()

trainflights = minmax.fit_transform(trainf)
testflights = minmax.transform(testf)

# defining a way to find Mean Absolute Percentage Error:
def PercentError(preds, ytest):
  error = abs(preds - ytest)

  errorp = np.mean(100 - 100*(error/ytest))

  print('the accuracy is:', errorp)
# Implementing the algo:
model = RandomForestRegressor(n_estimators=100,
                              random_state=0, verbose=1)

# Fitting the data to random forest regressor:
model.fit(trainflights, ytrain)

# Predicting the test dataset:
preds = model.predict(testflights)
PercentError(preds, ytest)
# Using Linear Regression:

LinearModel = LinearRegression()
LinearModel.fit(trainflights, ytrain)
# Predicting on the test dataset:
LinearPredictions = LinearModel.predict(testflights)
PercentError(LinearPredictions, ytest)
