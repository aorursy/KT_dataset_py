## just for practice

## the author is Mircea Stanciu



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
def computeCost(x, y, theta):

    h = x.dot(theta); # calculate hypothesis

    cost = sum(pow(h-y, 2)) / (2*m); # cost function

    return cost;
def gradientDescent(x, y, theta, alpha, iterations):

    computed_theta = theta;

    

    for i in range(0, iterations):

        h = x.dot(computed_theta);

        

        computed_theta[0] = computed_theta[0] - alpha * (1/m) * sum(h-y);

        computed_theta[1] = computed_theta[1] - alpha * (1/m) * sum((h-y) * X[:,1]);

        

    return computed_theta;
# step 1 - read data

data = pd.read_csv('../input/ex1data1.txt', header=None);



#print (data.shape[0]); # 输出矩阵的行数

#print (data.shape[1]); # 输出矩阵的列数 



X = data.iloc[:, 0].values # 读取所有行的第1列 [0:4,:0]读取第0,1,2,3这四行，前闭后开集合

y = data.iloc[:, 1].values # 读取所有行的第2列

m = y.size;



# step 2 - plot

plt.scatter(X, y, marker='x'); # 画图x,y为数据 marker为左边点的标记符号

plt.xlabel('Population of City in 10,000s'); # x轴描述

plt.ylabel('Profit in $10,000s'); # y轴描述

plt.show(); # 画出图形



# step 3 - cost function

# add column of ones to X

X = np.concatenate((np.ones((m,1), dtype=np.int), X.reshape(m,1)), axis=1); # 数组拼接

print (X.size);



# compute initial cost

print('Testing the cost function with theta = [0 ; 0]')



J = computeCost(X, y, np.array([0, 0]))

print('Expected cost value (approx): 32.07')

print('Actual cost value: {}\n'.format(J))



print('Testing the cost function with theta = [-1 ; 2]')

J = computeCost(X, y, np.array([-1, 2]))

print('Expected cost value (approx): 54.24')

print('Actual cost value: {}\n'.format(J))



# run gradient descent

theta = np.zeros(2)

alpha = 0.01

iterations = 1500



print('Running Gradient Descent')

theta = gradientDescent(X, y, theta, alpha, iterations)

print('Expected theta value (approx): [-3.6303, 1.1664]')

print('Actual theta value: {}\n'.format(theta))



# plot the linear fit

plt.scatter(X[:,1], y, marker='x', label='Training data')

plt.plot(X[:,1], X.dot(theta), color='r', label='Linear regression')

plt.xlabel('Population of City in 10,000s')

plt.ylabel('Profit in $10,000s')

plt.legend()

plt.show()



# predict values for population sizes of 35,000 and 70,000

predict1 = np.array([1, 3.5]).dot(theta)

print('For population of 35,000 we predict a profit of {}'.format(predict1 * 10000))

      

predict2 = np.array([1, 7]).dot(theta)

print('For population of 70,000 we predict a profit of {}'.format(predict2 * 10000))