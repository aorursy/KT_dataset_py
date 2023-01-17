import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn

%matplotlib inline



print(np.__version__)

print(pd.__version__)

import sys

print(sys.version)

import sklearn

print(sklearn.__version__)
x = np.linspace(-6, 6, num = 1000)

plt.figure(figsize = (12,8))

plt.plot(x, 1 / (1 + np.exp(-x))); # Sigmoid Function

plt.title("Sigmoid Function");
tmp = [0, 0.4, 0.6, 0.8, 1.0]
tmp
np.round(tmp)
np.array(tmp) > 0.7
dataset = [[-2.0011, 0],

           [-1.4654, 0],

           [0.0965, 0],

           [1.3881, 0],

           [3.0641, 0],

           [7.6275, 1],

           [5.3324, 1],

           [6.9225, 1],

           [8.6754, 1],

           [7.6737, 1]]
coef = [-0.806605464, 0.2573316]
for row in dataset:

    yhat = 1.0 / (1.0 + np.exp(- coef[0] - coef[1] * row[0]))

    print("yhat {0:.4f}, yhat {1}".format(yhat, round(yhat)))