import numpy as np

import matplotlib.pyplot as plt



lines = np.loadtxt('../input/data.csv', delimiter=',', dtype='str')

x_total = lines[:, 1:3].astype('float')

y_total = lines[:, 3].astype('float')



pos_index = np.where(y_total == 1)

neg_index = np.where(y_total == 0)

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.show()

print('Data set size:', x_total.shape[0])
from sklearn import linear_model



lr_clf = linear_model.LogisticRegression()

lr_clf.fit(x_total, y_total)

print(lr_clf.coef_[0])

print(lr_clf.intercept_)



y_pred = lr_clf.predict(x_total)

print('accuracy:',(y_pred == y_total).mean())



plot_x = np.linspace(-1.0, 1.0, 100)

plot_y = - (lr_clf.coef_[0][0] * plot_x + lr_clf.intercept_[0]) / lr_clf.coef_[0][1]

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(plot_x, plot_y, c='g')

plt.show()
# 1. finish function my_logistic_regression;

# 2. draw a training curve (the x-axis represents the number of training iterations, and the y-axis represents the training loss for each round);

# 3. draw a pic to show the result of logistic regression (just like the pic in section 2);

from numpy import *

n_iterations = 2000

learning_rate = 0.1

loss_list = []



x_total = lines[:, 0:3].astype('float')

for i in range(0,100):

     x_total[i,0]=1

        

def sigmoid(x):

    x = x.astype(float)

    return 1./(1+np.exp(-x))

def gradAscent(dataMatIn, classLabels):

     dataMatrix = mat(dataMatIn)             #convert to NumPy matrix

     labelMat = mat(classLabels).transpose() #convert to NumPy matrix

     m,n = shape(dataMatrix)

     alpha = 0.1  # 学习率

     maxCycles = 2000

     weights = ones((n,1))

     for k in range(maxCycles):              # heavy on matrix operations

         h = sigmoid(dataMatrix*weights)     # matrix multiplication

         error = (labelMat - h)              # vector subtraction

         temp = dataMatrix.transpose()* error # matrix multiplication

         weights = weights + alpha * temp  # 这里与梯度上升是等价的

     return weights

def my_logistic_regression(x_total, y_total):

     x=gradAscent(x_total,y_total)

     y_pred = sigmoid(mat(x_total)*x)

     for i in range(0,100):

         if y_pred[i]<=0.25: y_pred[i]=0

         else: y_pred[i]=1

     return y_pred

y_pred = my_logistic_regression(x_total, y_total)

co=0

for i in range(100):

    if y_pred[i]==y_total[i]:  co=co+1

print('accuracy:',co/100)



coef=gradAscent(x_total,y_total).getA()

plot_x = np.linspace(-1.0, 1.0, 100)

plot_y = - (coef[1] * plot_x + coef[0]) / coef[2]

plt.scatter(x_total[pos_index, 1], x_total[pos_index, 2], marker='o', c='r')

plt.scatter(x_total[neg_index, 1], x_total[neg_index, 2], marker='x', c='b')

plt.plot(plot_x, plot_y, c='g')

plt.show()