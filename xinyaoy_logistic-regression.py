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

import math

n_iterations = 2000

learning_rate = 0.1

loss_list = []

def my_logistic_regression(x_total, y_total):

   # TODO

    w1 = 0

    w2 = 0

    b = 0

    y_pred = [1] * 100

    min_y = 0.5

    for i in range(n_iterations):

        Loss = 0

        w1_g = 0

        w2_g = 0

        b_g = 0

        for j in range(100):

            y_pred[j] = (w1*x_total[j][0] + w2*x_total[j][1] + b)

            Loss += (-y_total[j] * math.log(sigmoid(y_pred[j]))-(1-y_total[j])*math.log(1-sigmoid(y_pred[j])))

            w1_g += (sigmoid(y_pred[j])-y_total[j]) * x_total[j][0]

            w2_g += (sigmoid(y_pred[j]) - y_total[j]) * x_total[j][1]

            b_g += (sigmoid(y_pred[j]) - y_total[j])

        loss_list.append(Loss/100)

        w1 -= learning_rate * w1_g/100

        w2 -= learning_rate * w2_g / 100

        b -= learning_rate * b_g / 100

    for j in range(100):

        if sigmoid(y_pred[j]) > min_y:

            y_pred[j] = 1

        else:

            y_pred[j] = 0



    # show the loss

    x = np.linspace(0, n_iterations, n_iterations)

    plt.plot(x, loss_list)

    plt.show()



    # show the result

    pos_index = np.where(y_total == 1)

    neg_index = np.where(y_total == 0)

    plot_x = np.linspace(-1.0, 1.0, 100)

    plot_y = - (w1 * plot_x + b) / w2

    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

    plt.plot(plot_x, plot_y, c='g')

    plt.show()



    return y_pred



def sigmoid(x):

    s = 1 / (1 + np.exp(-x))

    return s



y_pred = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())