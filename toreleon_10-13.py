import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data = pd.DataFrame({

    'Student': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],

    'Beer': [5, 2, 9, 8, 3, 7, 3, 5, 3, 5, 4, 6, 5, 7, 1, 7],

    'BAC': [0.10, 0.03, 0.19, 0.12, 0.04, 0.095, 0.07, 0.06, 0.02, 0.05, 0.07, 0.10, 0.085, 0.09, 0.01, 0.05]

    

})
data
plt.figure(figsize = (12, 8))

plt.scatter('Beer', 'BAC', data = data)

plt.title('Scatter of Beers and BAC')

plt.xlabel('Beer')

plt.ylabel('BAC')

plt.show()
X = data['Beer'].values

y = data['BAC'].values
def LinearRegressionModel(X, y):

    meanX = np.mean(X)

    meany = np.mean(y)

    theta = np.sum((X - meanX)*(y - meany))/np.sum((X - meanX)**2)

    bias = np.mean(meany - meanX*theta)

    return theta, bias
theta, bias = LinearRegressionModel(X, y)

y_predict = X*theta + bias
def R2_score(y, y_pred):

    meany = np.mean(y)

    return 1 - np.sum((y - y_pred)**2)/np.sum((y - meany)**2)
R2_score(y, y_predict)