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



n_iterations = 2000

learning_rate = 0.1

loss_list = []

#以上三个变量写在了下面的函数中，没有更改变量的值

def my_logistic_regression(x_total, y_total):

    theta = np.zeros([1,3])

    x_total = np.insert(x_total, 0, 1, axis=1)

    

    theta, loss_list = gradient_descent(x_total, y_total)

    

    #plot pictures here

    plot_curve(loss_list)

    plot_regression(theta)

    

    #get the prediction value list

    theta_x = x_total * theta

    prediction = sigmoid(theta_x)

    y_total = []

    for i in prediction:

        if i > 0.5:

            new_y = 1

        else:

            new_y = 0

        y_total.append(new_y)

    # TODO

    return y_total



# do the loss function

def sigmoid(z):

    return 1 / (1 + np.exp(-z))



def loss_funtion(x_total, y_total, weights):

    m, n = np.shape(x_total)

    loss = 0.0

    for i in range(m):

        sum_theta_x = 0.0

        for j in range(n):

            sum_theta_x += x_total[i, j] * weights.T[0, j]

        propability = sigmoid(sum_theta_x)

        loss += -y_total[i, 0] * np.log(propability) - (1 - y_total[i, 0]) * np.log(1 - propability)

    return loss



# do the gradient decent

def gradient_descent(x_total, y_total):

    data_matrix = np.mat(x_total)

    label_matrix = np.mat(y_total).T

    m, n = np.shape(data_matrix)

    weights = np.ones((n, 1))

    learning_rate = 0.1

    n_iterations = 2000

    loss_list = []



    while n_iterations > 0:

        loss = loss_funtion(data_matrix, label_matrix, weights)

        new_weights = weights - learning_rate * data_matrix.T * (sigmoid(data_matrix * weights) - label_matrix)

        new_loss = loss_funtion(data_matrix, label_matrix, new_weights)

        loss_list.append(new_loss)

        weights = new_weights

        n_iterations -=1

    return weights, loss_list



#plot

def plot_regression(theta):

    lines = np.loadtxt('../input/data.csv', delimiter=',', dtype='str')

    x_total = lines[:, 1:3].astype('float')

    y_total = lines[:, 3].astype('float')

    

    pos_index = np.where(y_total == 1)

    neg_index = np.where(y_total == 0)

    

    plot_x = np.linspace(-1.0, 1.0, 100)

    plot_y = (float(theta[0]) + float(theta[1]) * plot_x)/(-float(theta[2]))

    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

    plt.plot(plot_x, plot_y, c='g')

    

def plot_curve(loss_list):

    y_data = loss_list

    x_data = np.linspace(1, 2000, 2000)

    plt.plot(x_data,y_data)

    plt.xlabel('number of training iterations')

    plt.ylabel('training loss for each round')

    plt.show()

    

    

y_pred = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())