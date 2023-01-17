import pandas as pd

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt
df_train = pd.read_csv('../input/train.csv', index_col=0)
df_test = pd.read_csv('../input/test.csv', index_col=0)
years = 2018 - df_train['YearBuilt']
years = pd.DataFrame(years)

years = years.rename(columns={"YearBuilt":"Years"})
x_train = pd.concat([df_train, years], axis=1)
# notUsedFeatures = ['MSSubClass', 'MSZoning', 'LotFrontage', 'Stree', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl']

x_train = x_train[['LotArea', '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'KitchenAbvGr', 'PoolArea', 'Years']]
x_train = np.matrix(x_train)
x_train
y_train = df_train[['SalePrice']]

y_train = np.matrix(y_train)
def initialize_parameters(x):

    theta = np.random.rand(1, x.shape[1])

    theta = np.matrix(theta)

    return theta
def h(x, theta):

    y_predict = x * theta.T

    return y_predict
def cost_function(h, y):

    m = y.shape[0]

    total_loss = np.square(np.subtract(h, y)).sum(axis=0)

    cost = (1./(2. * m)) * total_loss

    cost = np.asscalar(cost)

    return cost
def fit(x, y, num_iterations=100, learning_rate=0.001):

    theta = initialize_parameters(x)

    m = x.shape[0]

    costs = []

    for i in range(num_iterations):

        for j in range(x.shape[1]):

            theta[0, j] = theta[0, j] - learning_rate * ((1.0/m)*((np.multiply(np.subtract(h(x, theta), y),(x[:, j]))).sum(axis = 0)))

        y_predict = h(x, theta)

        costs.append(cost_function(y_predict, y))

        if i % 1000 == 0:

            print("Cost after {} iterations: {}".format(i, costs[i]))

    return theta, costs
x_train.shape, y_train.shape
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_train_norm = scaler.fit_transform(x_train)
x_train_norm = np.matrix(x_train_norm)

x_train_norm = np.insert(x_train_norm, 0, 1, axis = 1)
x_train_norm.shape
theta, costs = fit(x_train_norm, y_train, num_iterations=8000, learning_rate=0.0001)
y_predict = h(x_train_norm, theta)
theta