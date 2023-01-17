# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt





data = pd.read_csv('../input/train.csv')

data.columns



X_train = np.array([data['LotArea']]).T

Y_train = np.array([data['SalePrice']]).T

X_train = X_train[0:2]

Y_train = Y_train[0:2]

m = data.shape[0]

learning_rate = 0.01

print(X_train.shape)

print(Y_train.shape)
# Initialize the weights (theta)

W = np.random.rand((X_train.shape[0]), 1)

b = 1

print(W.shape)
def train(X, W, Y, b, iterations, learning_rate=0.01):

    cost = 0

    for i in range(iterations):

        print("Iteration : " + str(i))

        # Calculate the hypthesis 

        H = np.dot(W.T,X)

#         print("Hypothesis Function")

#         print(H)

    

        # Compute cost

        cost = (1/2*m) * np.sum((H-Y)**2)

#         print("Cost")

#         print(cost)

        

        # Gradient Descent

#         print("Weights Before Update : ")

#         print(W)

        W = W - learning_rate * 1/m * np.sum((H-Y)*X)

#         print("Weights After Update :")

#         print(W)

#          print(W.shape)

        print(np.sum((H-Y)))

    return cost
cost = train(X_train, W, Y_train, b, 100, 0.001)

print("Cost : " + str(cost))