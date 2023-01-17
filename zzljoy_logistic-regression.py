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
learning_rate = 0.01
loss_list = []
def my_logistic_regression(x_total, y_total):
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    
    n = x_total.shape[0]
    w = -0.5
    b = 0.5
    dif_w = 0
    dif_b = 0
    threshold = 0.5
    for i in range(n_iterations):
        x_pred = w * x_total[:, 0] + b
        y_pred = sigmoid(x_total[:, 1] - x_pred)
        dif_w = sum((y_pred - y_total) * x_total[:, 0]) / n
        dif_b = sum(y_pred - y_total) / n
        loss = sum(- y_total * np.log(y_pred) - (1 - y_total) * np.log(1 - y_pred)) / n
        loss_list.append(loss)
        w = w + dif_w * learning_rate
        b = b + dif_b * learning_rate
    
    y_total = y_pred > threshold
    plot_x = np.linspace(-1.0, 1.0, 100)
    plot_y = w * plot_x + b
    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')
    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')
    plt.plot(plot_x, plot_y, c='g')
    plt.show()
    plot_it = np.linspace(1, n_iterations, 2000)
    plot_loss = np.array(loss_list)
    plt.plot(plot_it, plot_loss, c='b')
    plt.show()
    return y_total

y_pred = my_logistic_regression(x_total, y_total)
print('accuracy:',(y_pred == y_total).mean())

