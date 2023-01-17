# the author is emilyhorsmanBasic





# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
def sigmoid(z):

    return 1. / (1 + np.exp(-z));
def z(theta, x):

    assert theta.shape[1] == 1

    assert theta.shape[0] == x.shape[1]  # Theta should have as many rows as x has features.

    return np.dot(x, theta); # 计算矩阵乘积
 # 测试np.dot

a = np.array([[1,2],[3,4]]);

b = np.array([[4,1],[2,2]]);

print ("a.T*b is:",np.dot(a.T,b));

print ("a*b.T is:",np.dot(a,b.T));



a = np.array([1,1]);

b = np.array([2,3]);



print(a.shape);

print(b.T.shape);

print ("a*b is: ",np.dot(a,b.T)); # 一维向量的内积
def hypothesis(theta, x):

    return sigmoid(z(theta, x));
def cost(theta, x, y):

    # 最后结果n*1维的向量

    assert x.shape[1] == theta.shape[0]  # x has a column for each feature, theta has a row for each feature.

    assert x.shape[0] == y.shape[0]  # One row per sample.

    assert y.shape[1] == 1

    assert theta.shape[1] == 1

    

    h = hypothesis(theta, x);

    cost = -1/len(x) * np.sum( np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h)));

    

    return cost;
def gradient_descent(theta, x, y, learning_rate):

    h = hypothesis(theta, x);

    theta = theta - learning_rate / len(x) * np.dot(x.T, (h - y));

    return theta;
def minimize(theta, x, y, iterations, learning_rate):

    costs = []

    for _ in range(iterations):

        theta = gradient_descent(theta, x, y, learning_rate);

        costs.append(cost(theta, x, y));

    return theta, costs;
mushroom_data = pd.read_csv("../input/mushrooms.csv").dropna(); # 

mushroom_x = pd.get_dummies(mushroom_data.drop('class', axis=1))

mushroom_x['bias'] = 1

mushroom_x = mushroom_x.values

mushroom_y = (np.atleast_2d(mushroom_data['class']).T == 'p').astype(int)



x_train, x_test, y_train, y_test = train_test_split(mushroom_x, mushroom_y, train_size=0.85, test_size=0.15);

print ("x_train, y_train");

print (x_train.shape, y_train.shape);



candidate = np.atleast_2d([ np.random.uniform(-1, 1, 118) ]).T

theta, costs = minimize(candidate, x_train, y_train, 1200, 1.2)

plt.plot(range(len(costs)), costs)

plt.show()

print(costs[-1])



predictions = x_test.dot(theta) > 0

#print (predictions);

len(list(filter(lambda x: x[0] == x[1], np.dstack((predictions, y_test))[:,0]))) / len(predictions) # 正确率