# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import math


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import sys
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def f(x, eps):
    sum = 1
    k = 1
    term = 1
    if x >= 0.6 or x <= 0.6:
        eps = eps / 10
    while True:
        delta = (2 * k - 1) / (2 * k) * (x ** 2)
        term = term * delta
        sum += term
        k += 1
        if term <= eps:
            return sum
x_values = []
y_values = []
for x in np.linspace(-.95, .95, 39):
    x_values.append(round(x, 2))
    y_values.append(f(round(x, 2), 10 ** (-4)))

results = pd.DataFrame({'x': x_values, 'y': y_values})
y_built_in = [1 / math.sqrt(1 - (round(x, 2) ** 2)) for x in np.linspace(-.95, .95, 39)]
absolute_error = [abs(y_built_in[i] - y_values[i]) for i in range(len(y_values))]
results2 = pd.DataFrame({'x': x_values,
                         'y_my_function': y_values,
                         'y_built_in': y_built_in,
                         'abs_error': absolute_error,
                         'accuracy': 10 ** (-4)})
results.to_csv('results.csv', header=None, index=False)
results2.to_csv('results2.csv', header=None, index=False)
plt.plot(results.x, results.y)
plt.scatter(results.x, y_built_in)
plt.plot(results.x, results.y)
plt.show()
