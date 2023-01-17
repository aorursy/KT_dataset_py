# libs imported
import numpy as np
import pandas as pd
# import module we'll need to import our custom module
from shutil import copyfile
# copy our file into the working directory (make sure it has .py suffix)
copyfile(src = "../input/visuals/visuals.py", dst = "../working/visuals.py")
# import all our functions
import visuals as vs
# import os
# print(os.listdir("../input/visuals"))
# python version
from sys import version_info
print(version_info)
# import data
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv('../input/boston-house-prices/housing.csv', delim_whitespace=True, names=columns)
print(data.shape)
print(data.head(5))
# features and prices
features = data.drop('MEDV', axis=1)
prices = data['MEDV']
print(features.head(5))
print(prices.head(5))
# base info
max_price = np.max(prices)
min_price = np.min(prices)
mean_price = np.mean(prices)
median_price = np.median(prices)
print("max:%.2f  min:%.2f  mean:%.2f  midain:%.2f" %(max_price, min_price, mean_price, median_price))
# data split and regroup
from sklearn.model_selection import train_test_split
train_features, test_features, train_prices, test_prices = train_test_split(features, prices, test_size=0.2, random_state=0)
# Learning performance with decision tree
vs.ModelLearning(train_features, train_prices)
vs.ModelComplexity(train_features, train_prices)
# build model
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV

def fit_model(train_features, train_labels):
    kf = KFold(n_splits=10)
    dt = DecisionTreeRegressor()
    sc = make_scorer(r2_score)
    params = {'max_depth': range(1, 11)}
    grid = GridSearchCV(dt, params, sc, cv=kf)
    grid.fit(train_features, train_labels)
    return grid.best_estimator_
# train
optimal_reg = fit_model(train_features, train_prices)
print ("Parameter 'max_depth' is {} for the optimal model.".format(optimal_reg.get_params()['max_depth']))
# show performance
predictions = optimal_reg.predict(test_features)
r2_score = r2_score(predictions, test_prices)
print("r2 score is %.2f" %r2_score)