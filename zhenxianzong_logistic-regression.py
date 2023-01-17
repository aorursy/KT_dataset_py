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



#准备工作

import numpy as np

loss_list = []

rate = 0.1

Niterations = 2000



#定义sigmoid函数

def sigmoid(x):

    return 1.0 / (1 + np.exp(-x))



def my_logistic_regression(x_total, y_total):

    q=0

    Samples, Features = np.shape(x_total)

    y_total = y_total.reshape((100, 1))

    W = np.ones((Features, 1))

    #预测步骤

    for l in range(Niterations):

            Out = sigmoid(x_total.dot(W) + q)

            Err = y_total - Out

            loss = -np.multiply(y_total, np.log(Out)) - np.multiply((1 - y_total), np.log(1 - Out))

            loss = np.sum(loss) / Features

            loss_list.append(loss)

            W = W + rate * x_total.T.dot(Err)/Samples

            q = q + rate * np.sum(Err, axis=0, keepdims=True) / Samples

    #矩阵处理步骤

    y_total=sigmoid(x_total.dot(W) + q)

    y_total[y_total < 0.5] = 0

    y_total[y_total >= 0.5] = 1

    #可视化步骤

    m = range(Niterations)

    plt.plot(m, loss_list)

    plt.show()

    plot_x = np.linspace(-1.0, 1.0, 100)

    plot_y = -W[0][0] / W[1][0] * plot_x - q / W[1][0]

    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

    plt.plot(plot_x, plot_y.T, c='g')

    plt.show()

    return y_total.T



y_pred = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())