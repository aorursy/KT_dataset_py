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



#sigmoid函数

def sigmoid(x):    

    y = 1 / (1 + np.exp(-x))

    return y



def my_logistic_regression(x_total, y_total):

    weight = np.zeros(3)

    x_total_train = np.hstack((x_total, np.ones([x_total.shape[0], 1])))

    

    for i in range(n_iterations):

        p = sigmoid(np.dot(x_total_train, weight))

        loss = (-y_total * np.log(p) - (1 - y_total) * (np.log(1 - p))).mean()  #交叉熵损失

        loss_list.append(loss)

        w_gradient = (x_total_train * np.tile((p - y_total).reshape([-1,1]),3)).mean(axis=0)

        weight = weight - learning_rate * w_gradient

        

    y_pred = np.where(sigmoid(np.dot(x_total_train, weight))>0.5, 1, 0)

    

    return y_pred, weight



y_pred, weight = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())



print(weight)



plt.figure(figsize=(6,10))

plt.subplot(211)

plt.title('traing curve')

plt.xlabel('the number of training iterations')

plt.ylabel('the training loss')

plt.plot(np.arange(n_iterations), loss_list)



plt.subplot(212)

plot_x = np.linspace(-1.0, 1.0, 100)

plot_y = - (weight[0] * plot_x + weight[2]) / weight[1]

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(plot_x, plot_y, c='g')

plt.show()