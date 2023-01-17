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

class MyLogisticRegression (object):

    def __init__(self):

        self.weight = None

        self.intercept = None

        self.loss_list = None

    

    def probability(self, data):

        theta = np.dot(data, self.weight) + self.intercept

        return 1/(1+np.exp(-theta))

    

    def loss(self, data, target):

        y_probability = self.probability(data)

        temp = target * np.log(y_probability) + (1-target) * np.log(1-y_probability)

        return -np.mean(temp)  

    

    def fit(self, data, target, alpha=learning_rate, max_iteration=n_iterations):

        

        assert len(data.shape) == 2

        n = data.shape[1]

        

        self.loss_list = np.zeros(max_iteration)

        self.weight = np.zeros(n)

        self.intercept = np.zeros(1)

        

        for i in range(max_iteration):

            errors = self.probability(data) - target

            for j in range(n):

                self.weight[j] = self.weight[j] - alpha * np.mean(errors * data[:,j])

            self.intercept = self.intercept - alpha * np.mean(errors)

            self.loss_list[i] = self.loss(data, target)

        

    def predict(self, data):

        probs = self.probability(data)

        predict = [int(p >= 0.5) for p in probs]

        return np.asarray(predict)



def my_logistic_regression(x_total, y_total):

    Myitem = MyLogisticRegression()

    Myitem.fit(x_total,y_total,max_iteration=n_iterations)

    loss_list = Myitem.loss_list

    plt.plot(loss_list)

    plt.show()

    y_total = Myitem.predict(x_total)

    plot_y = -(Myitem.weight[0]*plot_x+Myitem.intercept[0])/Myitem.weight[1]

    return y_total,plot_y



y_pred,plot_y = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())



plot_x = np.linspace(-1.0, 1.0, 100)

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(plot_x, plot_y, c='g')

plt.show()